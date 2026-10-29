"""Exploratory causal test for the C5 dial. Not sealed; nothing here gates or confirms anything.

    PYTHONPATH=src python3 -m dialedin.probe_localize

Question: the mechanism probe (results/raw/probe-mech-20261029T065204Z) found the tilt change concentrated in the
attention of layers 2 and 3. Is it caused there? Each condition sets the dial to 2 with only a subset of the 32 adapter
modules switched on (the others' B factors are zeroed for that condition, then restored), and measures the C5 curve:
mean gain over every head in layers 2 to 51, one tone per pass at 0.1 x the median residual norm, periods 2 to 2,048,
8,192 tokens, directions 1 and 2, SDPA. Plus perplexity on tokens 2,049 to 8,191.

Conditions: base (dial 0); all; L2 attention only; L2; L3; L2 and L3; L4 and L5; all but L2; attention only; MLP only.
Reported per condition: tilt (from gains averaged over books and directions), change against base, and that change as
a fraction of the full adapter's change. "Sufficient" reads off the subset-only rows, "necessary" off "all but L2".

Text: the first 3 books of set C at tokens 60,000 (as the mechanism probe). Output:
results/raw/probe-loc-<utc>/localize.json, labeled exploratory.
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

from . import glimmer
from . import pilot_c5 as P

TOKENS = 8192
START = 60000
N_BOOKS = 3
SEEDS = (1, 2)
SCALE = 2.0


def layer_of(name: str) -> int:
    return int(name.split(".")[1])


def is_attn(name: str) -> bool:
    return name.split(".", 2)[2].startswith("self_attn.")


CONDITIONS = {
    "all": lambda n: True,
    "L2_attention": lambda n: layer_of(n) == 2 and is_attn(n),
    "L2": lambda n: layer_of(n) == 2,
    "L3": lambda n: layer_of(n) == 3,
    "L2_L3": lambda n: layer_of(n) in (2, 3),
    "L4_L5": lambda n: layer_of(n) in (4, 5),
    "all_but_L2": lambda n: layer_of(n) != 2,
    "attention_only": lambda n: is_attn(n),
    "mlp_only": lambda n: not is_attn(n),
}


class Mask:
    """Switches adapter modules off by zeroing their B factor; restore() puts every B back exactly."""

    def __init__(self, names, params):
        self.names, self.Bs = names, params[1::2]
        self.saved = [b.detach().clone() for b in self.Bs]

    def keep(self, pred):
        on = 0
        with torch.no_grad():
            for n, b, s in zip(self.names, self.Bs, self.saved):
                if pred(n):
                    b.copy_(s); on += 1
                else:
                    b.zero_()
        return on

    def restore(self):
        with torch.no_grad():
            for b, s in zip(self.Bs, self.saved):
                b.copy_(s)


def measure(decoder, rig, ids_list, hidden, periods, logits, scale):
    """Gains averaged over books and directions, tilt from them, mean of per-run tilts, mean perplexity."""
    rows = []
    for ids in ids_list:
        P.LoRALinear.scale = 0.0
        P.one_pass(decoder, rig, ids, None)
        norm = rig.median_norm
        rows += P.evaluate(decoder, rig, ids, SEEDS, (scale,), norm, hidden, periods=periods, logits=logits)
    g = np.mean([r["gains"] for r in rows], axis=0)
    return {"gains": g.tolist(), "tilt": float(P.slope(periods, torch.tensor(g, dtype=torch.float64))),
            "tilt_runs_mean": float(np.mean([r["tilt"] for r in rows])),
            "tilt_runs": [r["tilt"] for r in rows],
            "snr_min": float(np.min([r["snr"] for r in rows])),
            "perplexity": float(np.mean([r["perplexity"] for r in rows])) if logits else None}


def run(decoder, layers, names, params, ids_list, hidden, logits=None, conditions=None, log=print):
    periods = list(P.configure(TOKENS))
    rig = P.Rig(layers, range(2, len(layers)))
    mask = Mask(names, params)
    out = {}
    try:
        with torch.no_grad():
            t = time.time()
            out["base"] = measure(decoder, rig, ids_list, hidden, periods, logits, 0.0)
            out["base"]["modules_on"] = 0
            log(f"   base: tilt {out['base']['tilt']:.4f} ({time.time() - t:.0f} s)")
            for c, pred in (conditions or CONDITIONS).items():
                t = time.time()
                on = mask.keep(pred)
                out[c] = measure(decoder, rig, ids_list, hidden, periods, logits, SCALE)
                out[c]["modules_on"] = on
                mask.restore()
                log(f"   {c}: {on} modules on, tilt {out[c]['tilt']:.4f} ({time.time() - t:.0f} s)")
    finally:
        mask.restore()
        rig.remove()
        P.LoRALinear.scale = 0.0
    base = out["base"]["tilt"]
    full = out["all"]["tilt"] - base if "all" in out else None
    for c, r in out.items():
        r["change"] = r["tilt"] / base - 1
        r["fraction_of_full"] = (r["tilt"] - base) / full if full else None
        if logits and out["base"]["perplexity"]:
            r["perplexity_ratio"] = r["perplexity"] / out["base"]["perplexity"]
    return periods, out


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
    s = Session(spec, texts, model, decoder, layers, tok, platform, repo, NullLedger(), allow_unfilled=True)
    if s.adapter_sha != tj["adapter_sha256"]:
        raise SystemExit("the adapter file differs from the one C5 trained")
    books = sorted(int(g) for g in spec["inputs"]["prompt_sets"]["C"])[:N_BOOKS]
    ids = [torch.tensor([tok(texts[g], add_special_tokens=False)["input_ids"][START:START + TOKENS]], device=s.device)
           for g in books]
    print(f"== load {time.time() - t0:.0f} s; adapter {s.adapter_sha[:12]}, {len(s.adapter_modules)} modules; "
          f"books {books}", flush=True)
    lg = lambda m: print(m, flush=True)
    periods, out = run(decoder, layers, s.adapter_modules, s.params, ids, decoder.config.hidden_size, logits=s.logits, log=lg)
    run_id = "probe-loc-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(repo, "results", "raw", run_id)
    os.makedirs(out_dir)
    write_json(os.path.join(out_dir, "localize.json"),
               {"exploratory": True, "sealed": False, "what": __doc__.split("\n\n")[2].strip(), "books": books,
                "start": START, "tokens": TOKENS, "seeds": list(SEEDS), "scale": SCALE, "periods": periods,
                "adapter_file": spec["adapter"]["file"], "adapter_sha256": s.adapter_sha, "platform": platform,
                "modules": s.adapter_modules, "conditions": out}, indent=1)
    print("== condition              modules   tilt     change   share of full   perplexity   min SNR")
    for c, r in out.items():
        frac = "" if r["fraction_of_full"] is None or c == "base" else f"{100 * r['fraction_of_full']:.0f}%"
        print(f"   {c:20s} {r['modules_on']:5d}   {r['tilt']:.4f}  {100 * r['change']:+6.1f}%  {frac:>12s}"
              f"   x{r.get('perplexity_ratio', float('nan')):.4f}   {r['snr_min']:.1f}")
    print(f"== exploratory; written to {os.path.relpath(out_dir, repo)}/localize.json")


if __name__ == "__main__":
    main()
