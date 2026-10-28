"""Pre-seal dry run of the C4 harness on the pod. Writes nothing to results/.

    PYTHONPATH=src python3 -m dialedin.smoke_c4 --spec criteria/drafts/c4-glimmer-structure-and-text-v1.yaml

Builds the real C4 Session (seal and code checks skipped). Checks that each window size takes effect,
that the RoPE change takes effect and restores exactly, then runs undriven, sham and driven passes
(N = 2 and 4096) under BASE, W4096, R2.0 and CONST, and a closing sham. Prints pass time, peak
memory, sham counts, finiteness and BASE perplexity only. It prints no gains, SNRs, tilts, or the
perplexity of any changed condition, since those decide outcomes.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import time
import types

import numpy as np
import torch
import yaml

from . import glimmer, run_c4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", default="/workspace/weights/Muse-Glimmer-30B")
    ap.add_argument("--manifest", default="/workspace/data/books/MANIFEST.json")
    args = ap.parse_args()
    spec = yaml.safe_load(open(args.spec))
    s = object.__new__(run_c4.Session)
    s.repo, s.seal_hash, s.spec, s.code = os.getcwd(), "smoke", spec, {}
    s.ledger = types.SimpleNamespace(check=lambda *a: None, add=lambda *a: 0.0, last_pass=0.0)
    s.texts = glimmer.load_texts(spec, args.manifest, os.path.dirname(args.manifest))
    glimmer.configure_determinism()
    t = time.time()
    s.model, s.decoder, s.layers, s.tok, s.platform = glimmer.load(spec, args.weights)
    print(f"== load {time.time() - t:.0f} s; preflight passed: {s.platform}")
    run_c4.Session._setup(s)
    print(f"== local layers: {len(s.local_ids)}; RoPE base {s.theta0}; inv_freq size {s.inv_freq0.numel()}; softcap {s.softcap}; output multiplier {s.multiplier}")
    windows = []
    for w in (1024, 2048, 4096):
        s.set_structure(w, 1.0)
        windows.append(s.window_check(w))
        print(f"   window {w}: reaches {windows[-1]['reaches']}  as expected: {windows[-1]['as_expected']}")
    s.set_condition("BASE")
    rope = [s.rope_check(f) for f in (0.5, 2.0)]
    for c in rope:
        print(f"== RoPE x{c['factor']}: layer 0 output changed {c['changed']}; restored bit for bit {c['restored']}")
    lc = s.logits_check()
    print(f"== perplexity path: max |logit difference| {lc['max_abs_diff']:.4g} of max |logit| "
          f"{lc['max_abs_logit']:.4g} (tolerance {lc['tolerance']:.4g}); as expected: {lc['as_expected']}")
    checks_ok = all(c["as_expected"] for c in windows + rope) and lc["as_expected"]
    torch.cuda.reset_peak_memory_stats()

    def timed(label, fn, *a):
        t = time.time()
        out = fn(*a)
        print(f"   {label}: {time.time() - t:.2f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        return out

    shams = []
    for cond in ("BASE", "W4096", "R2.0", "CONST"):
        s.set_condition(cond)
        ids, _ = s.ids(run_c4.PHASE_A_BOOK, run_c4.CONDITIONS[cond][2])
        m, ppl = timed(f"{cond} undriven + perplexity", s.baseline, ids)
        if cond == "BASE":
            print(f"   BASE perplexity on book 1342, tokens 8193 to 32767: {ppl:.3f}")
        shams.append(timed(f"{cond} sham", s.sham_differences, ids))
        for N in (2, 4096):
            out = timed(f"{cond} driven N={N}", s.measure, ids, N, run_c4.A * m, 0.0, "smoke")
            print(f"   outputs finite = {all(bool(np.isfinite(out[k]).all()) for k in ('target', 'floor', 'bin_power'))}")
    shams.append(timed("closing sham", s.sham_differences, ids))
    s.set_condition("BASE")
    print(f"== sham differing values: {shams} (all must be 0)")
    print("== smoke test PASSED" if checks_ok and all(x == 0 for x in shams) else "== smoke test FAILED")


if __name__ == "__main__":
    main()
