"""Exploratory feasibility pilot for C5 (a LoRA tilt dial). Not a sealed run; nothing here is reported as a result.

    PYTHONPATH=src python3 -m dialedin.pilot_c5 --steps 20

Question: can a low-rank adapter be trained, at 8K tokens on one GPU, to change the tilt of Glimmer's frequency
response, and does the change follow the adapter scale? The pilot measures what the C5 pre-registration needs:
memory and time per step, whether the gradient carries signal, how noisy an 8K tilt estimate is, whether scale 0
reproduces the base model bit for bit, and whether forward-mode autodiff works through Glimmer.

Text: only Moby Dick (book 2701) beyond token 98,304. No sealed run reads past token 32,768 of any book.
  training region: tokens 98,304 to 106,495; held-out region: tokens 131,072 to 139,263.
Directions: training draws a fresh random unit direction each step (seeds 10,000 + step); evaluation uses seeds 1
and 2, never used in training. Output: /workspace/env/pilot_c5.json.

Measurement at 8K (differs from the sealed 32K window; the pilot reports the gap, C5 must gate at 32K):
  T = 8192, window [2048, 8192), W = 6144, periods 2, 4, 16, 64, 256, 1024, 2048 (all divide W).
  Undriven and driven inputs run as one batch of 2, so gradient checkpointing recomputes both the same way.
  The drive is a sum of tones a_k cos(2 pi t / N_k + phi_k) along the direction at the layer 1 output.
  Each head's response is projected onto Hann-windowed cos and sin at each tone's bin (a lock-in, linear and
  differentiable); gain = |driven - undriven projection| / |drive projection|; curve = mean gain over all heads in
  layers 2 to 51; tilt = least-squares slope of ln(gain) on ln(N).
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math
import time

import numpy as np
import torch
from torch import nn

T, T1 = 8192, 2048
W = T - T1
PERIODS = (2, 4, 16, 64, 256, 1024, 2048)
TRAIN_START, EVAL_START, BOOK = 98304, 131072, 2701
EVAL_SEEDS = (1, 2)
TONE_A = 0.1                                   # total amplitude budget, as a fraction of the median residual norm
KL_CHUNK = 1024


# ---------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    """y = base(x) + scale * (alpha / r) * B(A(x)). At scale 0 the branch is skipped, so outputs are bit for bit
    those of the base layer. A and B are float32; the branch runs in the input dtype."""
    scale = 0.0                                # shared by every wrapper: the dial

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base, self.r, self.alpha = base, r, alpha
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))

    def forward(self, x):
        y = self.base(x)
        if LoRALinear.scale == 0.0:
            return y
        d = (x @ self.A.to(x.dtype).T) @ self.B.to(x.dtype).T
        return y + (LoRALinear.scale * self.alpha / self.r) * d


def attach_lora(layers, layer_ids, r, alpha):
    """Wraps every nn.Linear inside the chosen decoder layers. Returns the wrapped module names and the params."""
    names, params = [], []
    for i in layer_ids:
        for name, mod in list(layers[i].named_modules()):
            for cname, child in list(mod.named_children()):
                if isinstance(child, nn.Linear):
                    w = LoRALinear(child, r, alpha)
                    setattr(mod, cname, w)
                    names.append(f"layers.{i}.{name + '.' if name else ''}{cname}")
                    params += [w.A, w.B]
    for p in params:
        p.requires_grad_(True)
    return names, params


# ---------------------------------------------------------------- measurement
def bins(periods):
    return [W // n for n in periods]


def side_bins(k0, tone_bins):
    """The two bins nearest k0 (at distance 2 or more) that are at least 2 from every tone bin, in (0, W/2]."""
    out = []
    for dist in range(2, 64):
        for k in (k0 - dist, k0 + dist):
            if 0 < k <= W // 2 and all(abs(k - b) >= 2 for b in tone_bins) and k not in out:
                out.append(k)
        if len(out) >= 2:
            return out[:2]
    raise ValueError(k0)


def basis(periods, device):
    """Rows: for each period, Hann*cos and Hann*sin at its bin, then at two nearby noise bins clear of every tone."""
    t = torch.arange(W, dtype=torch.float64)
    hann = 0.5 - 0.5 * torch.cos(2 * math.pi * t / W)
    rows = []
    tb = bins(periods)
    for k0 in tb:
        for k in [k0] + side_bins(k0, tb):
            ang = 2 * math.pi * k * t / W
            rows += [hann * torch.cos(ang), hann * torch.sin(ang)]
    return torch.stack(rows).float().to(device)                 # [6P, W]


def tones(periods, amp_total, rng):
    """One amplitude per tone (amp_total / sqrt(P)) and a phase (0 for N = 2, whose bin is Nyquist; else random)."""
    a = amp_total / math.sqrt(len(periods))
    phases = [0.0 if n == 2 else float(rng.uniform(0, 2 * math.pi)) for n in periods]
    t = np.arange(T, dtype=np.float64)
    w = sum(a * np.cos(2 * np.pi * t / n + p) for n, p in zip(periods, phases))
    return w, phases


def input_mags(w, periods, B):
    x = torch.as_tensor(w[T1:], dtype=torch.float64)
    proj = (B.double().cpu() @ x).reshape(len(periods), 3, 2)[:, 0]   # target bin only
    return proj.norm(dim=1)                                             # [P]


class Rig:
    """Hooks on the decoder: drive at the layer 1 output (batch element 1 only), lock-in projections at every
    measured layer's o_proj input, and the median residual norm at layer 1 (batch element 0)."""

    def __init__(self, layers, measured, heads=32):
        self.layers, self.measured, self.heads = layers, list(measured), heads
        self.delta = None
        self.B = None
        self.proj = {}
        self.median_norm = None
        self.handles = [layers[1].register_forward_hook(self._drive)]
        for i in self.measured:
            self.handles.append(layers[i].self_attn.o_proj.register_forward_pre_hook(self._probe(i)))

    def _drive(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self.median_norm = float(hidden[0].detach().float().norm(dim=-1).median())
        if self.delta is None:
            return None
        new = hidden.clone()
        new[1] = (hidden[1].float() + self.delta).to(hidden.dtype)
        return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new

    def _probe(self, i):
        def hook(module, args):
            if self.B is not None:
                x = args[0][:, T1:].float()                                     # [2, W, 4096]
                self.proj[i] = torch.einsum("rw,bwd->brd", self.B, x)           # [2, 6P, 4096]
        return hook

    def curves(self, periods, in_mag):
        """Mean gain over heads, per period; and the mean noise magnitude at the side bins (same units)."""
        P = len(periods)
        g, nz = [], []
        for i in self.measured:
            d = (self.proj[i][1] - self.proj[i][0]).reshape(P, 3, 2, self.heads, -1)
            mag = d.pow(2).sum(dim=(2, 4)).sqrt()                               # [P, 3, heads]
            g.append(mag[:, 0]); nz.append(mag[:, 1:].mean(dim=1))
        g = torch.stack(g, 1).reshape(P, -1).mean(1) / in_mag.to(g[0].device).float()
        nz = torch.stack(nz, 1).reshape(P, -1).mean(1) / in_mag.to(nz[0].device).float()
        return g, nz

    def remove(self):
        for h in self.handles:
            h.remove()


def slope(periods, gains):
    x = torch.log(torch.tensor(periods, dtype=torch.float32, device=gains.device))
    y = torch.log(gains)
    xc = x - x.mean()
    return (xc * (y - y.mean())).sum() / (xc * xc).sum()


def direction(seed, dim, device):
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(dim, generator=g, dtype=torch.float32)
    return (d / d.norm()).to(device)


def run_pass(decoder, rig, ids, periods, w, d, keep_hidden=False):
    """One batch-of-2 pass: element 0 undriven, element 1 driven by w along d. Returns (gains, noise, hidden0)."""
    rig.B = basis(periods, ids.device)
    rig.delta = torch.as_tensor(w, dtype=torch.float32, device=ids.device)[:, None] * d[None, :]
    out = decoder(input_ids=ids.repeat(2, 1), use_cache=False)
    g, nz = rig.curves(periods, input_mags(w, periods, rig.B))
    h = out.last_hidden_state[0] if keep_hidden else None
    return g, nz, h


def scaled_tones(rig_norm, periods, rng):
    return tones(periods, TONE_A * rig_norm, rng)


# ---------------------------------------------------------------- KL
def logits_fn(head, multiplier, softcap):
    def f(h):
        x = head(h)
        if multiplier is not None:
            x = x * multiplier
        if softcap:
            x = torch.tanh(x / softcap) * softcap
        return x.float()
    return f


def kl_to_base(h_adapt, h_base, logits):
    """Mean over positions T1 .. T-1 of KL(base || adapted), chunked; each chunk is checkpointed."""
    from torch.utils.checkpoint import checkpoint
    total = 0.0
    for s in range(T1, T, KL_CHUNK):
        def chunk(ha, hb):
            lp = logits(ha).log_softmax(-1)
            with torch.no_grad():
                lb = logits(hb).log_softmax(-1)
            return (lb.exp() * (lb - lp)).sum(-1).sum()
        total = total + checkpoint(chunk, h_adapt[s:s + KL_CHUNK], h_base[s:s + KL_CHUNK], use_reentrant=False)
    return total / (T - T1)


def checkpoint_layers(layers):
    """Per-layer activation checkpointing (non-reentrant), done by hand so it does not depend on HF's switch.
    Module hooks run in __call__, outside the checkpointed forward; submodule hooks run again on recompute.
    Returns a function that restores the original forwards."""
    from torch.utils.checkpoint import checkpoint
    originals = []
    for layer in layers:
        orig = layer.forward
        originals.append((layer, orig))

        def fwd(*a, _orig=orig, **kw):
            return checkpoint(_orig, *a, use_reentrant=False, **kw)
        layer.forward = fwd

    def undo():
        for layer, orig in originals:
            layer.forward = orig
    return undo


# ---------------------------------------------------------------- pilot
def evaluate(decoder, rig, ids, seeds, scales, norm, dim):
    """No grad. Tilt over all 7 periods, per held-out direction and adapter scale."""
    rows = []
    with torch.no_grad():
        for sd in seeds:
            d = direction(sd, dim, ids.device)
            w, _ = scaled_tones(norm, PERIODS, np.random.default_rng(sd))
            for sc in scales:
                LoRALinear.scale = sc
                g, nz, _ = run_pass(decoder, rig, ids, PERIODS, w, d)
                rows.append({"seed": sd, "scale": sc, "tilt": float(slope(PERIODS, g)),
                             "gains": g.tolist(), "snr": (g / nz).tolist()})
    LoRALinear.scale = 0.0
    return rows


def forward_ad_probe(decoder, layers, measured, ids, d, norm, n_tokens, try_backward, params):
    """Does forward-mode AD run through Glimmer, and does a backward pass through it work? Short input only."""
    import torch.autograd.forward_ad as fwAD
    x = ids[:, :n_tokens]
    out = {"tokens": n_tokens}
    t = torch.arange(n_tokens, dtype=torch.float32, device=ids.device)
    tangent = (TONE_A * norm * torch.cos(2 * math.pi * t / 64))[:, None] * d[None, :]
    seen = {}

    def drive(m, a, o):
        h = o[0] if isinstance(o, tuple) else o
        dual = fwAD.make_dual(h, tangent[None].to(h.dtype))
        return (dual,) + tuple(o[1:]) if isinstance(o, tuple) else dual

    def probe(i):
        def hook(m, a):
            tan = fwAD.unpack_dual(a[0]).tangent
            seen[i] = None if tan is None else tan.float().pow(2).sum()
        return hook

    hs = [layers[1].register_forward_hook(drive)] + [layers[i].self_attn.o_proj.register_forward_pre_hook(probe(i))
                                                       for i in measured]
    try:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with fwAD.dual_level(), torch.set_grad_enabled(try_backward):
            decoder(input_ids=x, use_cache=False)
            got = [v for v in seen.values() if v is not None]
            out["tangent_layers"] = len(got)
            if try_backward and got:
                loss = torch.stack(got).sum()
                loss.backward()
                out["backward_grad_norm"] = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params
                                                                 if p.grad is not None)))
        out["ok"] = True
        out["seconds"] = time.time() - t0
        out["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
    except Exception as e:                                     # the point of the probe is to find out
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {str(e)[:400]}"
    finally:
        for h in hs:
            h.remove()
        for p in params:
            p.grad = None
    return out


def main():
    import yaml
    from . import glimmer
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/workspace/weights/Muse-Glimmer-30B")
    ap.add_argument("--manifest", default="/workspace/data/books/MANIFEST.json")
    ap.add_argument("--spec", default="criteria/sealed/c4-glimmer-structure-and-text-v1.yaml")
    ap.add_argument("--out", default="/workspace/env/pilot_c5.json")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--target-factor", type=float, default=0.7, help="target tilt = factor x the starting tilt")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--periods-per-step", type=int, default=3)
    ap.add_argument("--skip-forward-ad", action="store_true")
    args = ap.parse_args()

    spec = yaml.safe_load(open(args.spec))
    texts = glimmer.load_texts(spec, args.manifest, os.path.dirname(args.manifest))
    torch.manual_seed(0)
    model, decoder, layers, tok, platform = glimmer.load(spec, args.weights)
    torch.use_deterministic_algorithms(False)                  # training backward has nondeterministic kernels
    dev = next(model.parameters()).device
    for p in model.parameters():
        p.requires_grad_(False)
    report = {"args": vars(args), "platform": platform, "notes": __doc__.splitlines()[0]}

    allids = tok(texts[BOOK], add_special_tokens=False)["input_ids"]
    assert len(allids) >= EVAL_START + T, len(allids)
    train_ids = torch.tensor([allids[TRAIN_START:TRAIN_START + T]], device=dev)
    eval_ids = torch.tensor([allids[EVAL_START:EVAL_START + T]], device=dev)
    measured = range(2, len(layers))
    dim = decoder.config.hidden_size
    rig = Rig(layers, measured)
    logits = logits_fn(model.get_output_embeddings(), getattr(decoder.config, "output_multiplier", None),
                       getattr(decoder.config, "final_logit_softcapping", None))

    # 1. base reference before any wrapper: o_proj inputs of one undriven pass, for the scale-0 check
    ref = {}
    hk = layers[10].self_attn.o_proj.register_forward_pre_hook(lambda m, a: ref.setdefault("x", a[0].detach().clone()))
    with torch.no_grad():
        rig.delta, rig.B = None, None
        decoder(input_ids=eval_ids, use_cache=False)
        norm = rig.median_norm
    hk.remove()
    report["median_norm_layer1"] = norm

    names, params = attach_lora(layers, args.layers, args.rank, alpha=2.0 * args.rank)
    report["lora"] = {"modules": names, "params": int(sum(p.numel() for p in params))}
    print(f"== LoRA on {len(names)} linear layers in layers {args.layers}; {report['lora']['params']:,} params")

    # 2. scale 0 reproduces the base bit for bit
    chk = {}
    hk = layers[10].self_attn.o_proj.register_forward_pre_hook(lambda m, a: chk.setdefault("x", a[0].detach().clone()))
    LoRALinear.scale = 0.0
    with torch.no_grad():
        decoder(input_ids=eval_ids, use_cache=False)
    hk.remove()
    report["scale0_bitwise_equal"] = bool(torch.equal(ref["x"], chk["x"]))
    print(f"== scale 0 bit for bit equal to base: {report['scale0_bitwise_equal']}")

    # 3. noise of the 8K tilt before training
    t0 = time.time()
    report["eval_before"] = evaluate(decoder, rig, eval_ids, EVAL_SEEDS, [0.0], norm, dim)
    for r in report["eval_before"]:
        print(f"   before: seed {r['seed']} tilt {r['tilt']:.4f}  min SNR {min(r['snr']):.1f}")
    report["eval_seconds_per_pass"] = (time.time() - t0) / len(EVAL_SEEDS)
    tilt0 = float(np.mean([r["tilt"] for r in report["eval_before"]]))
    target = args.target_factor * tilt0

    # 4. forward-mode AD probe (inference, then backward), on 2,048 tokens
    if not args.skip_forward_ad:
        d0 = direction(1, dim, dev)
        LoRALinear.scale = 1.0
        for bw in (False, True):
            r = forward_ad_probe(decoder, layers, measured, eval_ids, d0, norm, 2048, bw, params)
            report[f"forward_ad_backward_{bw}"] = r
            print(f"== forward AD (backward={bw}): {r}")
        LoRALinear.scale = 0.0

    # 5. training
    undo = checkpoint_layers(layers)
    opt = torch.optim.Adam(params, lr=args.lr)
    rng = np.random.default_rng(0)
    steps = []
    for step in range(args.steps):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        per = sorted(rng.choice(PERIODS, size=args.periods_per_step, replace=False).tolist())
        d = direction(10000 + step, dim, dev)
        w, _ = scaled_tones(norm, per, rng)
        with torch.no_grad():
            LoRALinear.scale, rig.delta, rig.B = 0.0, None, None
            h_base = decoder(input_ids=train_ids, use_cache=False).last_hidden_state[0]
        LoRALinear.scale = 1.0
        g, nz, h_adapt = run_pass(decoder, rig, train_ids, per, w, d, keep_hidden=True)
        tilt = slope(per, g)
        tilt_loss = ((tilt - target) / tilt0) ** 2
        kl = kl_to_base(h_adapt, h_base, logits)
        loss = tilt_loss + args.beta * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None)))
        opt.step()
        rig.delta, rig.B, rig.proj = None, None, {}
        row = {"step": step, "periods": per, "tilt": float(tilt), "tilt_loss": float(tilt_loss), "kl": float(kl),
               "grad_norm": gn, "seconds": time.time() - t0, "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
               "min_snr": float((g / nz).min())}
        steps.append(row)
        print(f"   step {step:>3}: N {per} tilt {row['tilt']:.4f} (target {target:.4f})  kl {row['kl']:.2e}  "
              f"grad {gn:.2e}  {row['seconds']:.1f} s  peak {row['peak_gib']:.1f} GiB", flush=True)
        del h_base, h_adapt, g, nz, tilt, loss
    report["train"] = steps
    report["target_tilt"], report["start_tilt"] = target, tilt0

    # 6. dose response on held-out text and directions
    undo()
    report["eval_after"] = evaluate(decoder, rig, eval_ids, EVAL_SEEDS, [0.0, 0.5, 1.0, 1.5, 2.0], norm, dim)
    for sc in (0.0, 0.5, 1.0, 1.5, 2.0):
        ts = [r["tilt"] for r in report["eval_after"] if r["scale"] == sc]
        print(f"   held out, scale {sc}: tilt {np.mean(ts):.4f}  ({', '.join(f'{x:.4f}' for x in ts)})")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"== pilot written to {args.out}")


if __name__ == "__main__":
    main()
