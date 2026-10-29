"""Exploratory mechanism probe for the C5 dial. Not sealed; nothing here gates or confirms anything.

    PYTHONPATH=src python3 -m dialedin.probe_mechanisms --spec criteria/sealed/c6-glimmer-dial-topic-switch-v2.yaml

Question: when the frozen C5 adapter is turned up (scales 0, 1, 2), which mechanisms in Glimmer change, and which part
of layers 2 to 5 carries the change in lean? Run it after C6 is complete (it reads C6's text region).

Part 1, transfer (SDPA, 8,192 tokens, the pilot's 8K setup): the C5 drive at the layer 1 output, one tone per pass at
0.1 x the median residual norm, periods 2 to 2,048, directions 1 and 2 (never used in training). Gains, SNR and tilt at:
  - the C5 curve (mean gain over every head in layers 2 to 51), as a check that the dial still moves the lean here;
  - each head's output in layers 2 to 5 (o_proj input), rows 1 and 3 of the mechanism table;
  - the attention branch (post_attention_layernorm output) and MLP branch (post_feedforward_layernorm output) of
    layers 2 to 5, rows 1, 6 and 8;
  - the residual after layers 2, 3, 4, 5, 11, 25 and 51, row 9.
Part 2, operating point (eager attention, 4,096 tokens, undriven; statistics over queries 2,048 to 4,095), every layer:
  - per head: mean attention distance (rows 1, 3, 5), entropy (row 2), mass on the first token and peak mass (row 10),
    mass on the last 16 tokens, and the head's share of the summed head output norm (row 3);
  - fast fraction (share of non-DC power at periods of 16 tokens or less) of the gate pre-activation (row 6), of the
    log scale factor of each of the four norms (row 7), and of each branch's input and output (rows 1 and 8).
Layer 3 is the only global layer the adapter touches (row 4). Row 11 is the input text, which the dial cannot change.

Text: the first 3 books of set C, tokens 60,000 onward. Output: results/raw/probe-mech-<utc>/mechanisms.json, labeled
exploratory, and a printed summary.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import torch

from . import glimmer, metrics
from . import pilot_c5 as P

SCALES = (0.0, 1.0, 2.0)
SEEDS = (1, 2)
TRANSFER_TOKENS = 8192
STATS_TOKENS = 4096
STATS_Q0 = 2048
NEAR = 16
FAST_PERIOD = 16
START = 60000
N_BOOKS = 3
ADAPTED = (2, 3, 4, 5)
RESIDUAL_POINTS = (2, 3, 4, 5, 11, 25, 51)
HEAD_CHUNK = 8
PARTS = ("input_layernorm", "self_attn", "post_attention_layernorm", "pre_feedforward_layernorm", "mlp",
         "post_feedforward_layernorm")


# ---------------------------------------------------------------- pure statistics
def fast_fraction(x: torch.Tensor, min_period: int = FAST_PERIOD) -> float:
    """x: [L] or [L, D] over time. Share of the non-DC power (Hann window, mean removed, summed over D) at periods of
    min_period tokens or less."""
    x = x.detach().float()
    if x.dim() == 1:
        x = x[:, None]
    L = x.shape[0]
    x = x - x.mean(0, keepdim=True)
    w = torch.hann_window(L, periodic=True, device=x.device, dtype=x.dtype)
    p = torch.fft.rfft(x * w[:, None], dim=0).abs().pow(2).sum(1)
    k = torch.arange(p.shape[0], device=x.device)
    total = p[1:].sum()
    return float(p[1:][k[1:] * min_period >= L].sum() / total) if total > 0 else float("nan")


def attention_stats(w: torch.Tensor, q0: int) -> dict:
    """w: attention weights [1, H, L, L] (rows sum to 1). Per head, averaged over queries q0 .. L-1."""
    H, L = w.shape[1], w.shape[-1]
    qpos = torch.arange(q0, L, device=w.device)
    dist = (qpos[:, None] - torch.arange(L, device=w.device)[None, :]).clamp_min(0).float()
    near = (dist < NEAR).float()
    out = {k: [] for k in ("mean_distance", "entropy", "first_token_mass", "peak_mass", "near_mass")}
    for h0 in range(0, H, HEAD_CHUNK):
        a = w[0, h0:h0 + HEAD_CHUNK, q0:, :].float()
        out["mean_distance"] += (a * dist).sum(-1).mean(-1).tolist()
        out["entropy"] += (-(a * a.clamp_min(1e-30).log()).sum(-1)).mean(-1).tolist()
        out["first_token_mass"] += a[..., 0].mean(-1).tolist()
        out["peak_mass"] += a.max(-1).values.mean(-1).tolist()
        out["near_mass"] += (a * near).sum(-1).mean(-1).tolist()
        del a
    return out


def point_gains(p0: torch.Tensor, p1: torch.Tensor, n_periods: int, in_mag: torch.Tensor, heads: int = None):
    """Projections [6P, D] undriven and driven. Gain and noise at each period's bin and side bins, divided by the drive's
    own magnitude. With heads, per head: returns [P, heads] arrays."""
    d = (p1 - p0).double().cpu()
    if heads:
        mag = d.reshape(n_periods, 3, 2, heads, -1).pow(2).sum(dim=(2, 4)).sqrt()      # [P, 3, heads]
    else:
        mag = d.reshape(n_periods, 3, 2, -1).pow(2).sum(dim=(2, 3)).sqrt()             # [P, 3]
    m = in_mag.double().cpu().reshape(-1, *([1] * (mag.dim() - 2)))
    return mag[:, 0] / m, mag[:, 1:].mean(1) / m


def tilt(periods, gains) -> float:
    g = np.asarray(gains, dtype=float)
    if not np.all(np.isfinite(g)) or np.any(g <= 0):
        return float("nan")
    return float(P.slope(periods, torch.tensor(g, dtype=torch.float64)))


# ---------------------------------------------------------------- hooks
def first(x):
    return x[0] if isinstance(x, tuple) else x


class Points:
    """Lock-in projections at named module outputs while B is set (part 1)."""

    def __init__(self, layers):
        self.B, self.proj, self.handles = None, {}, []
        for i in ADAPTED:
            self._add(f"L{i}.attn", layers[i].post_attention_layernorm)
            self._add(f"L{i}.mlp", layers[i].post_feedforward_layernorm)
        for i in RESIDUAL_POINTS:
            self._add(f"L{i}.out", layers[i])

    def _add(self, name, module):
        def hook(m, a, o):
            if self.B is not None:
                self.proj[name] = self.B @ first(o)[0, P.T1:].float()
        self.handles.append(module.register_forward_hook(hook))

    def remove(self):
        for h in self.handles:
            h.remove()


class Stats:
    """Undriven operating-point statistics for every layer (part 2). Needs eager attention for the weights."""

    def __init__(self, layers, q0, heads=32):
        self.q0, self.heads, self.rows, self.handles = q0, heads, {}, []
        self.saw_weights = False
        for i, layer in enumerate(layers):
            r = self.rows.setdefault(i, {})
            a = layer.self_attn
            self.handles += [
                a.register_forward_hook(self._attn(r)),
                a.o_proj.register_forward_pre_hook(self._heads(r)),
                a.gate_proj.register_forward_hook(lambda m, x, o, r=r: r.__setitem__("gate_fast", fast_fraction(first(o)[0, self.q0:]))),
                layer.post_attention_layernorm.register_forward_hook(self._branch(r, "attn_out")),
                layer.pre_feedforward_layernorm.register_forward_hook(self._branch(r, "mlp_in")),
                layer.post_feedforward_layernorm.register_forward_hook(self._branch(r, "mlp_out")),
                layer.input_layernorm.register_forward_hook(self._branch(r, "attn_in")),
            ]
            for nm in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
                self.handles.append(getattr(layer, nm).register_forward_pre_hook(self._norm(r, nm)))

    def _attn(self, r):
        def hook(m, a, o):
            w = o[1] if isinstance(o, tuple) and len(o) > 1 else None
            if not torch.is_tensor(w) or w.dim() != 4 or w.shape[-1] != w.shape[-2]:
                raise SystemExit("attention weights were not returned: eager attention is not active")
            self.saw_weights = True
            r.update(attention_stats(w, self.q0))
        return hook

    def _heads(self, r):
        def hook(m, a):
            x = a[0][0, self.q0:].float()
            n = x.reshape(x.shape[0], self.heads, -1).norm(dim=-1).mean(0)
            r["head_output_share"] = (n / n.sum()).tolist()
        return hook

    def _branch(self, r, key):
        def hook(m, a, o):
            y = first(o)[0, self.q0:]
            r[f"{key}_fast"] = fast_fraction(y)
            r[f"{key}_norm"] = float(y.float().norm(dim=-1).mean())
        return hook

    def _norm(self, r, name):
        def hook(m, a):
            x = a[0][0, self.q0:].float()
            log_scale = -0.5 * torch.log(x.pow(2).mean(-1) + 1e-6)       # log of the RMS norm's 1 / rms factor
            r[f"{name}_scale_fast"] = fast_fraction(log_scale)
            r[f"{name}_scale_cv"] = float(log_scale.exp().std() / log_scale.exp().mean())
        return hook

    def take(self):
        out = {i: dict(r) for i, r in self.rows.items()}
        for r in self.rows.values():
            r.clear()
        return out

    def remove(self):
        for h in self.handles:
            h.remove()


def use_eager(model):
    """Switches attention to eager in place (same weights). Tries the transformers API, then the config fields."""
    try:
        model.set_attn_implementation("eager")
        return "set_attn_implementation"
    except Exception:
        for c in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
            if c is not None:
                c._attn_implementation = "eager"
        return "config"


def check_layout(layers):
    missing = [f"layer {i}: {p}" for i, l in enumerate(layers) for p in PARTS if not hasattr(l, p)]
    missing += [f"layer {i}: self_attn.{p}" for i, l in enumerate(layers) for p in ("gate_proj", "o_proj")
                if not hasattr(l.self_attn, p)]
    if missing:
        raise SystemExit("unexpected layer layout: " + "; ".join(missing[:6]))


# ---------------------------------------------------------------- the two parts
def transfer(decoder, layers, ids_list, hidden, heads=32):
    """Part 1. Returns per scale: summed gains and noise per point (averaged later), per-head gains in layers 2 to 5."""
    periods = P.configure(TRANSFER_TOKENS)
    rig = P.Rig(layers, range(2, len(layers)), heads=heads)
    pts = Points(layers)
    B, Bc = P.basis(periods, ids_list[0].device), P.basis(periods, "cpu")
    acc = {}                                   # (scale, point) -> list of (gains[P], noise[P])
    try:
        with torch.no_grad():
            for ids in ids_list:
                P.LoRALinear.scale = 0.0
                rig.B, pts.B = None, None
                P.one_pass(decoder, rig, ids, None)
                norm = rig.median_norm
                for sd in SEEDS:
                    d = P.direction(sd, hidden, ids.device)
                    rng = np.random.default_rng(sd)
                    ws = [P.tones([n], P.TONE_A * norm, rng)[0] for n in periods]
                    for sc in SCALES:
                        P.LoRALinear.scale = sc
                        rig.B, pts.B = B, B
                        pts.proj = {}
                        _, r0 = P.one_pass(decoder, rig, ids, None)
                        q0 = dict(pts.proj)
                        rows = {}
                        for k, w in enumerate(ws):
                            pts.proj = {}
                            delta = torch.as_tensor(w, dtype=torch.float32, device=ids.device)[:, None] * d[None, :]
                            _, r1 = P.one_pass(decoder, rig, ids, delta)
                            im = P.input_mags(w, periods, Bc).clamp_min(1e-30)
                            g, nz = rig.curves(r0, r1, periods, im)
                            rows.setdefault("c5_curve", []).append((float(g[k]), float(nz[k])))
                            for name, p1 in pts.proj.items():
                                g, nz = point_gains(q0[name], p1, len(periods), im)
                                rows.setdefault(name, []).append((float(g[k]), float(nz[k])))
                            for i in ADAPTED:
                                g, nz = point_gains(r0[i], r1[i], len(periods), im, heads=heads)
                                rows.setdefault(f"L{i}.heads", []).append((g[k].tolist(), nz[k].tolist()))
                        for name, v in rows.items():
                            acc.setdefault((sc, name), []).append(v)
    finally:
        rig.remove(); pts.remove()
        P.LoRALinear.scale = 0.0
    out = {}
    for (sc, name), runs in acc.items():
        g = np.array([[x[0] for x in run] for run in runs], dtype=float)       # [runs, P] or [runs, P, heads]
        nz = np.array([[x[1] for x in run] for run in runs], dtype=float)
        gm, nm = g.mean(0), nz.mean(0)
        row = {"gains": gm.tolist(), "snr_min": float(np.min(gm / np.maximum(nm, 1e-30)))}
        row["tilt"] = ([tilt(periods, gm[:, h]) for h in range(gm.shape[1])] if gm.ndim == 2 else tilt(periods, gm))
        out.setdefault(name, {})[f"S{sc:g}"] = row
    return list(periods), out


def operating_point(model, decoder, layers, ids_list, heads=32, q0=STATS_Q0):
    """Part 2, eager attention. Per scale, per layer: statistics averaged over books."""
    how = use_eager(model)
    st = Stats(layers, q0, heads)
    out = {}
    try:
        with torch.no_grad():
            for sc in SCALES:
                P.LoRALinear.scale = sc
                per_book = []
                for ids in ids_list:
                    decoder(input_ids=ids, use_cache=False)
                    per_book.append(st.take())
                out[f"S{sc:g}"] = {i: {k: (np.mean([b[i][k] for b in per_book], axis=0).tolist()
                                         if isinstance(per_book[0][i][k], list)
                                         else float(np.mean([b[i][k] for b in per_book])))
                                     for k in per_book[0][i]} for i in per_book[0]}
    finally:
        st.remove()
        P.LoRALinear.scale = 0.0
    if not st.saw_weights:
        raise SystemExit("no attention weights were seen")
    return how, out


def summarize(periods, tr, op, global_layers) -> dict:
    s = {}
    c = tr["c5_curve"]
    s["c5_curve_tilt"] = {k: c[k]["tilt"] for k in c}
    s["c5_curve_change_S2"] = c["S2"]["tilt"] / c["S0"]["tilt"] - 1
    s["c5_curve_snr_min"] = min(c[k]["snr_min"] for k in c)
    s["tilt_by_point"] = {n: {k: v["tilt"] for k, v in tr[n].items()} for n in tr if not n.endswith(".heads") and n != "c5_curve"}
    s["snr_min_by_point"] = {n: min(v["snr_min"] for v in tr[n].values()) for n in tr}
    dist0 = [d for i in ADAPTED for d in op["S0"][i]["mean_distance"]]
    dtilt = [b - a for i in ADAPTED for a, b in zip(tr[f"L{i}.heads"]["S0"]["tilt"], tr[f"L{i}.heads"]["S2"]["tilt"])]
    ok = [(x, y) for x, y in zip(dist0, dtilt) if np.isfinite(x) and np.isfinite(y)]
    s["heads_L2_5_spearman_distance_vs_tilt_change"] = metrics.spearman([x for x, _ in ok], [y for _, y in ok]) if len(ok) > 2 else None
    s["heads_L2_5_tilt_mean"] = {k: float(np.nanmean([t for i in ADAPTED for t in tr[f"L{i}.heads"][k]["tilt"]])) for k in tr["L2.heads"]}

    def group(layers_, key, weighted=False):
        res = {}
        for k in op:
            vals = []
            for i in layers_:
                v = op[k][i][key]
                if isinstance(v, list):
                    v = float(np.dot(v, op[k][i]["head_output_share"])) if weighted else float(np.mean(v))
                vals.append(v)
            res[k] = float(np.mean(vals))
        return res
    n = len(next(iter(op.values())))
    groups = {"adapted_L2_5": list(ADAPTED), "downstream_L6_51": list(range(6, n)),
              "downstream_global": [i for i in global_layers if i >= 6]}
    groups = {g: ls for g, ls in groups.items() if ls}
    keys = ("mean_distance", "entropy", "first_token_mass", "peak_mass", "near_mass", "gate_fast", "attn_in_fast",
            "attn_out_fast", "mlp_in_fast", "mlp_out_fast", "attn_out_norm", "mlp_out_norm",
            "input_layernorm_scale_fast", "post_attention_layernorm_scale_fast", "pre_feedforward_layernorm_scale_fast",
            "post_feedforward_layernorm_scale_fast")
    s["operating_point"] = {g: {k: group(ls, k) for k in keys} for g, ls in groups.items()}
    for g, ls in groups.items():
        s["operating_point"][g]["output_weighted_distance"] = group(ls, "mean_distance", weighted=True)
    s["operating_point"]["layer3_global"] = {k: group([3], k) for k in keys}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="criteria/sealed/c6-glimmer-dial-topic-switch-v2.yaml")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--weights", default="/workspace/weights/Muse-Glimmer-30B")
    ap.add_argument("--manifest", default="/workspace/data/books/MANIFEST.json")
    args = ap.parse_args()
    import yaml
    from .run_c6 import NullLedger, Session, write_json
    repo = os.path.abspath(args.repo)
    spec = yaml.safe_load(open(os.path.join(repo, args.spec)))
    tj = json.load(open(os.path.join(repo, os.path.dirname(spec["adapter"]["file"]), "training.json")))
    texts = glimmer.load_texts(spec, args.manifest, os.path.dirname(args.manifest))
    glimmer.configure_determinism()
    t0 = time.time()
    model, decoder, layers, tok, platform = glimmer.load(spec, args.weights)
    check_layout(layers)
    s = Session(spec, texts, model, decoder, layers, tok, platform, repo, NullLedger(), allow_unfilled=True)
    if s.adapter_sha != tj["adapter_sha256"]:
        raise SystemExit("the adapter file differs from the one C5 trained")
    books = sorted(int(g) for g in spec["inputs"]["prompt_sets"]["C"])[:N_BOOKS]
    toks = {g: tok(texts[g], add_special_tokens=False)["input_ids"] for g in books}
    dev = s.device
    print(f"== load {time.time() - t0:.0f} s; adapter {s.adapter_sha[:12]}; books {books}", flush=True)

    t = time.time()
    ids8 = [torch.tensor([toks[g][START:START + TRANSFER_TOKENS]], device=dev) for g in books]
    periods, tr = transfer(decoder, layers, ids8, decoder.config.hidden_size)
    print(f"== part 1 (transfer, SDPA): {time.time() - t:.0f} s", flush=True)

    t = time.time()
    ids4 = [torch.tensor([toks[g][START:START + STATS_TOKENS]], device=dev) for g in books]
    how, op = operating_point(model, decoder, layers, ids4)
    print(f"== part 2 (operating point, eager via {how}): {time.time() - t:.0f} s", flush=True)

    summ = summarize(periods, tr, op, platform["global_layers"])
    run_id = "probe-mech-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(repo, "results", "raw", run_id)
    os.makedirs(out_dir)
    write_json(os.path.join(out_dir, "mechanisms.json"),
               {"exploratory": True, "sealed": False, "what": __doc__.split("\n\n")[2].strip(), "books": books,
                "start": START, "scales": list(SCALES), "seeds": list(SEEDS), "periods": periods,
                "adapter_file": spec["adapter"]["file"], "adapter_sha256": s.adapter_sha, "platform": platform,
                "eager_switch": how, "summary": summ, "transfer": tr,
                "operating_point": {k: {str(i): v for i, v in d.items()} for k, d in op.items()}}, indent=1)

    pct = lambda a, b: f"{100 * (b / a - 1):+.1f}%" if a else "n/a"
    print(f"== C5 curve tilt S0 {summ['c5_curve_tilt']['S0']:.4f} -> S2 {summ['c5_curve_tilt']['S2']:.4f} "
          f"({100 * summ['c5_curve_change_S2']:+.1f}%), min SNR {summ['c5_curve_snr_min']:.1f}")
    print("== tilt by point (S0 -> S1 -> S2), min SNR:")
    for n, v in summ["tilt_by_point"].items():
        print(f"   {n:8s} {v['S0']:+.4f} -> {v['S1']:+.4f} -> {v['S2']:+.4f}  ({pct(v['S0'], v['S2'])}), "
              f"SNR >= {summ['snr_min_by_point'][n]:.1f}")
    print(f"== heads in layers 2 to 5: mean tilt {summ['heads_L2_5_tilt_mean']}; Spearman(S0 distance, tilt change) "
          f"{summ['heads_L2_5_spearman_distance_vs_tilt_change']}")
    print("== operating point, S0 -> S2:")
    for g, d in summ["operating_point"].items():
        print(f"   [{g}]")
        for k, v in d.items():
            print(f"     {k:40s} {v['S0']:.5g} -> {v['S2']:.5g} ({pct(v['S0'], v['S2'])})")
    print(f"== exploratory; written to {os.path.relpath(out_dir, repo)}/mechanisms.json")


if __name__ == "__main__":
    main()
