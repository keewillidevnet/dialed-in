#!/usr/bin/env python3
"""Live dial readout for Dialed In.

Runs only the first few layers of Muse Glimmer-30B, injects the sine probe at
seven periods in one batch, and measures how strongly each attention head in
the read layer carries each period (a lock-in amplifier). Every time the dial
moves, the adapter scale changes and a fresh measurement is streamed to
live/index.html over a WebSocket.

This is a live readout, not a sealed result: one text window, one read layer,
measured on the spot. The sealed numbers stay in results/derived.

    python live/server.py --model <Glimmer path or HF id> --text <book.txt> \
        --adapter module:live.dial_shim:load --amplitude 0.1

    python live/server.py --self-test      # tiny random model, no GPU, no weights
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import math
import time

import torch

PERIODS = [2, 4, 16, 64, 256, 1024, 4096]


class _Stop(Exception):
    """Raised inside the read hook so nothing after the read layer runs."""


# ---------------------------------------------------------------- model plumbing
LAYER_PATHS = ("model.layers", "model.language_model.layers", "model.model.layers",
               "language_model.model.layers", "transformer.h", "model.decoder.layers", "layers")


def get_attr_path(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def find_layers(model, override=None):
    for path in ([override] if override else LAYER_PATHS):
        try:
            layers = get_attr_path(model, path)
            if len(layers):
                return layers
        except AttributeError:
            continue
    raise RuntimeError("Could not find the decoder layers. Pass --layers-attr (for example model.layers).")


def find_out_proj(layer, override=None):
    attn = next((getattr(layer, n) for n in ("self_attn", "attn", "attention", "self_attention") if hasattr(layer, n)), None)
    if attn is None:
        raise RuntimeError("Could not find the attention block in the read layer.")
    for n in ([override] if override else ("o_proj", "out_proj", "dense", "c_proj", "wo")):
        if hasattr(attn, n):
            return getattr(attn, n)
    raise RuntimeError("Could not find the attention output projection. Pass --out-proj-attr.")


def text_config(cfg):
    return getattr(cfg, "text_config", None) or cfg


def load_model(args):
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    tcfg = text_config(cfg)
    keep = max(args.inject_after, args.read_layer) + 1
    total = getattr(tcfg, "num_hidden_layers", None)
    if args.truncate and total and keep < total:
        tcfg.num_hidden_layers = keep          # only the first layers are loaded: a few GB, not 60
        if isinstance(getattr(tcfg, "layer_types", None), list):
            tcfg.layer_types = tcfg.layer_types[:keep]
        print(f"loading the first {keep} of {total} layers")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    kw = dict(config=cfg, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=args.trust_remote_code)
    import transformers
    model = None
    for name in ("AutoModelForMultimodalLM", "AutoModelForCausalLM", "AutoModel"):   # Glimmer loads as multimodal
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(args.model, **kw)
            break
        except (ValueError, KeyError) as err:
            print(f"{name} could not load it ({err.__class__.__name__}); trying the next loader")
    if model is None:
        raise RuntimeError("no transformers loader could load the model")
    return model.to(args.device).eval(), tcfg


def load_tokens(args, vocab):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    with open(args.text, encoding="utf-8") as f:
        ids = tok(f.read(), add_special_tokens=False)["input_ids"]
    if len(ids) < args.tokens:
        print(f"text has {len(ids)} tokens; repeating it to reach {args.tokens}")
        ids = (ids * (args.tokens // max(1, len(ids)) + 1))
    return torch.tensor(ids[args.offset: args.offset + args.tokens])


# ---------------------------------------------------------------- the dial
def attach_dial(model, spec):
    """Return (model, set_scale). spec is 'peft:<dir>', 'module:<pkg.mod>:<fn>' or 'none'.

    module: your function gets the loaded model and returns set_scale(s), a callable that
    sets the trained adapter to strength s (0 = off, 1 = as trained). It may also return
    (model, set_scale) if it wraps the model.
    """
    if spec in (None, "", "none"):
        print("WARNING: no adapter attached; the dial will not change anything")
        return model, lambda s: None
    kind, _, rest = spec.partition(":")
    if kind == "peft":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, rest).eval()
        base = {m: dict(m.scaling) for m in model.modules() if isinstance(getattr(m, "scaling", None), dict)}
        if not base:
            raise RuntimeError("No LoRA layers found in the PEFT adapter.")

        def set_scale(s):
            for m, b in base.items():
                for k, v in b.items():
                    m.scaling[k] = v * s
        return model, set_scale
    if kind == "module":
        mod, _, fn = rest.rpartition(":")
        out = getattr(importlib.import_module(mod), fn)(model)
        return out if isinstance(out, tuple) else (model, out)
    raise ValueError(f"unknown --adapter {spec!r}")


# ---------------------------------------------------------------- measurement
class Engine:
    def __init__(self, model, tcfg, ids, direction, args):
        self.model, self.args = model, args
        layers = find_layers(model, args.layers_attr)
        self.n_heads = getattr(tcfg, "num_attention_heads")
        self.head_dim = getattr(tcfg, "head_dim", None) or tcfg.hidden_size // self.n_heads
        dev = next(model.parameters()).device
        T = ids.numel()
        if T % max(PERIODS):
            raise ValueError(f"--tokens must be a multiple of {max(PERIODS)} so every period fits whole cycles")
        self.B = 1 + len(PERIODS)                         # row 0 = no probe, rows 1.. = one period each
        self.ids = ids.to(dev).unsqueeze(0).repeat(self.B, 1)
        t = torch.arange(T, dtype=torch.float32)
        ph = torch.stack([2 * math.pi * t / N for N in PERIODS])            # (P, T)
        self.sin = torch.sin(ph).to(dev)
        self.cos = torch.cos(ph).to(dev)
        # sums of squares for the lock-in; at N = 2 the sine is zero at every token, so only the cosine counts
        self.css = (self.cos ** 2).sum(1)
        self.sss = (self.sin ** 2).sum(1).clamp_min(1e-6)
        # cosine, not sine: sin(pi * t) is zero at every token, so a sine probe at N = 2 would inject nothing
        wave = torch.cat([torch.zeros(1, T), torch.cos(ph)])                 # (B, T)
        self.wave = wave.to(dev)
        self.dir = (direction / direction.norm()).to(dev)
        self.amp = None                                   # probe size, set from the hidden-state RMS on first pass
        self.cap = None
        self.on = True
        layers[args.inject_after].register_forward_hook(self._inject)
        find_out_proj(layers[args.read_layer], args.out_proj_attr).register_forward_pre_hook(self._read)

    def _inject(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if self.amp is None:
            self.amp = self.args.amplitude or float(h[0].float().pow(2).mean().sqrt()) * self.args.eps
        add = (self.amp * self.wave[:, :, None] * self.dir[None, None, :]).to(h.dtype)
        h = h + add
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def _read(self, module, inputs):
        self.cap = inputs[0].detach().float()            # (B, T, heads * head_dim), heads not yet mixed
        raise _Stop

    @torch.no_grad()
    def measure(self, set_scale, scale):
        set_scale(scale)
        try:
            self.model(input_ids=self.ids, use_cache=False)
        except _Stop:
            pass
        B, T, _ = self.cap.shape
        x = self.cap.view(B, T, self.n_heads, self.head_dim)
        d = x[1:] - x[:1]                                  # what the probe added, per period
        i = torch.einsum("pthd,pt->phd", d, self.cos) / self.css[:, None, None]
        q = torch.einsum("pthd,pt->phd", d, self.sin) / self.sss[:, None, None]
        q = torch.where(self.sss[:, None, None] > 1.0, q, torch.zeros_like(q))
        return ((i.pow(2) + q.pow(2)).sum(-1).sqrt() / self.amp).cpu()     # (P, H): gain per period per head


def lockin(sig, P):
    t = torch.arange(sig.numel(), dtype=torch.float32)
    c, s = torch.cos(2 * math.pi * t / P), torch.sin(2 * math.pi * t / P)
    i = float((sig * c).sum() / (c * c).sum())
    q = float((sig * s).sum() / (s * s).sum()) if float((s * s).sum()) > 1.0 else 0.0
    return math.hypot(i, q)


def lockin_check():
    """The lock-in returns the amplitude of a known wave at every period, N = 2 included, and ignores the others."""
    T = 8192
    t = torch.arange(T, dtype=torch.float32)
    for N in PERIODS:
        other = 16 if N != 16 else 64
        sig = 0.37 * torch.cos(2 * math.pi * t / N + (0.0 if N == 2 else 0.8)) + 0.5 * torch.cos(2 * math.pi * t / other)
        assert abs(lockin(sig, N) - 0.37) < 1e-3, (N, lockin(sig, N))
        assert lockin(sig, 256 if N != 256 else 1024) < 1e-3
    print("lock-in check passed (all seven periods)")


def summarize(gains, base_curve=None, base_tilt=None):
    per = gains.mean(dim=1).clamp_min(1e-9)              # (P,) mean over heads
    curve = (per / per[-1]).tolist()                       # % of the slowest wave
    lx = torch.log(torch.tensor(PERIODS, dtype=torch.float32))
    ly = torch.log(per)
    tilt = float(((lx - lx.mean()) * (ly - ly.mean())).sum() / ((lx - lx.mean()) ** 2).sum())
    out = {"curve": curve, "tilt": tilt, "heads": gains.tolist()}
    if base_tilt:
        out["flatter_pct"] = (1 - tilt / base_tilt) * 100
    return out


# ---------------------------------------------------------------- server
async def serve(engine, set_scale, args, label):
    import websockets

    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    g0 = await loop.run_in_executor(None, engine.measure, set_scale, 0.0)
    base = summarize(g0)
    ms0 = (time.perf_counter() - t0) * 1000
    hello = {"type": "hello", "model": label, "read_layer": args.read_label, "n_heads": engine.n_heads,
             "periods": PERIODS, "tokens": engine.ids.shape[1], "baseline": base,
             "ref": float(g0[-1].max())}             # strongest head at the slowest period, dial at 0
    print(f"ready: {engine.n_heads} heads, baseline tilt {base['tilt']:.4f}, {ms0:.0f} ms per update; ws://{args.host}:{args.port}")
    clients, want, wake = set(), {"s": args.start_scale}, asyncio.Event()

    async def handler(ws, *_):
        clients.add(ws)
        try:
            await ws.send(json.dumps(hello))
            wake.set()
            async for msg in ws:
                d = json.loads(msg)
                if "scale" in d:
                    want["s"] = max(0.0, min(args.max_scale, float(d["scale"])))
                    wake.set()
        finally:
            clients.discard(ws)

    async def worker():
        last = None
        while True:
            await wake.wait()
            wake.clear()
            s = want["s"]
            if s == last and not clients:
                continue
            t1 = time.perf_counter()
            g = await loop.run_in_executor(None, engine.measure, set_scale, s)
            frame = {"type": "frame", "scale": s, "ms": (time.perf_counter() - t1) * 1000,
                     **summarize(g, base["curve"], base["tilt"])}
            last = s
            websockets.broadcast(clients, json.dumps(frame))
            if want["s"] != s:
                wake.set()                                 # the dial moved while we measured: go again

    async with websockets.serve(handler, args.host, args.port, max_size=None):
        task = asyncio.create_task(worker())
        await asyncio.Future()
        task.cancel()


# ---------------------------------------------------------------- self test
def self_test_setup(args):
    """A tiny random Llama with a toy dial, to check the pipeline end to end without weights."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=3, num_attention_heads=16,
                      num_key_value_heads=16, vocab_size=1000, max_position_embeddings=16384)
    model = LlamaForCausalLM(cfg).eval()
    # Toy dial: a path that copies the attention input straight into the head outputs, scaled by s.
    # It carries every period equally, so turning it up flattens the curve, the way the real dial does.
    attn = model.model.layers[args.read_layer].self_attn
    state = {"s": 0.0, "h": None}
    attn.register_forward_pre_hook(lambda m, a, kw: state.__setitem__("h", kw.get("hidden_states", a[0] if a else None)), with_kwargs=True)
    attn.o_proj.register_forward_pre_hook(lambda m, i: (i[0] + 0.04 * state["s"] * state["h"],))
    ids = torch.randint(0, cfg.vocab_size, (args.tokens,))
    return model, cfg, ids, (lambda s: state.__setitem__("s", s))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="Glimmer weights: local path or Hugging Face id")
    p.add_argument("--text", help="plain-text file to run (one of the books)")
    p.add_argument("--adapter", default="none", help="peft:<dir> | module:<pkg.mod>:<fn> | none")
    p.add_argument("--probe", help="probe direction saved with torch.save (a 1-D tensor); default: random with --seed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amplitude", type=float, default=None, help="fixed probe amplitude (the sealed runs use 0.1); overrides --eps")
    p.add_argument("--eps", type=float, default=0.05, help="probe amplitude as a fraction of the hidden-state RMS, if --amplitude is not set")
    p.add_argument("--inject-after", type=int, default=1, help="0-based index of the layer whose output gets the probe")
    p.add_argument("--read-layer", type=int, default=2, help="0-based index of the layer whose heads are read")
    p.add_argument("--read-label", default="layer 2", help="what the page calls the read layer")
    p.add_argument("--tokens", type=int, default=8192, help="window length; a multiple of 4096")
    p.add_argument("--offset", type=int, default=0, help="first token of the window")
    p.add_argument("--layers-attr", help="attribute path to the decoder layers, if auto-detect fails")
    p.add_argument("--out-proj-attr", help="name of the attention output projection, if auto-detect fails")
    p.add_argument("--no-truncate", dest="truncate", action="store_false", help="load every layer")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--start-scale", type=float, default=0.0)
    p.add_argument("--max-scale", type=float, default=2.0)
    p.add_argument("--self-test", action="store_true", help="tiny random model and toy dial, no weights needed")
    args = p.parse_args()

    lockin_check()
    if args.self_test:
        model, cfg, ids, set_scale = self_test_setup(args)
        tcfg, label = cfg, "self-test model (random weights)"
    else:
        if not (args.model and args.text):
            p.error("--model and --text are required (or use --self-test)")
        model, tcfg = load_model(args)
        model, set_scale = attach_dial(model, args.adapter)
        ids, label = load_tokens(args, tcfg.vocab_size), "Muse Glimmer-30B"
    hidden = tcfg.hidden_size
    if args.probe:
        direction = torch.load(args.probe).flatten().float()
        assert direction.numel() == hidden, f"probe has {direction.numel()} dims, model has {hidden}"
    else:
        direction = torch.randn(hidden, generator=torch.Generator().manual_seed(args.seed))
    engine = Engine(model, tcfg, ids, direction, args)
    asyncio.run(serve(engine, set_scale, args, label))


if __name__ == "__main__":
    main()
