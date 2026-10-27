"""Pre-seal dry run of the C1 v5 harness on the pod. Writes nothing to results/.

    PYTHONPATH=src python3 -m dialedin.smoke_c1v5 --spec criteria/drafts/c1-glimmer-frequency-response-v5.yaml

Builds the real v5 Session (seal and code checks skipped, everything else the same), then runs the
undriven pass, a sham, driven passes at N = 2 and 4096 with both phases, and a closing sham. It
prints pass time, peak memory, sham counts and whether outputs are finite. It prints no gains or SNRs.
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

from . import glimmer, run_c1v5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", default="/workspace/weights/Muse-Glimmer-30B")
    ap.add_argument("--manifest", default="/workspace/data/books/MANIFEST.json")
    args = ap.parse_args()
    spec = yaml.safe_load(open(args.spec))
    s = object.__new__(run_c1v5.Session)
    # Session.__init__ without the seal and code-hash checks, which need a sealed file.
    s.repo, s.seal_hash, s.spec, s.code = os.getcwd(), "smoke", spec, {}
    s.ledger = types.SimpleNamespace(check=lambda *a: None, add=lambda *a: 0.0, last_pass=0.0)
    s.texts = glimmer.load_texts(spec, args.manifest, os.path.dirname(args.manifest))
    glimmer.configure_determinism()
    t = time.time()
    s.model, s.decoder, s.layers, s.tok, s.platform = glimmer.load(spec, args.weights)
    print(f"== load {time.time() - t:.0f} s; preflight passed: {s.platform}")
    run_c1v5.Session._setup(s)
    ids, _ = s.ids(run_c1v5.PHASE_A_BOOK)
    torch.cuda.reset_peak_memory_stats()

    def timed(label, fn, *a):
        t = time.time()
        out = fn(*a)
        print(f"   {label}: {time.time() - t:.2f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        return out

    m = timed("undriven", s.baseline, ids)
    sham = timed("sham", s.sham_differences, ids)
    for N in (2, 4096):
        for phase in (run_c1v5.PHASE_MAIN, run_c1v5.PHASE_CONTROL):
            out = timed(f"driven N={N} phase={phase:.3f}", s.measure, ids, N, 0.1 * m, phase, "smoke")
            finite = all(bool(np.isfinite(out[k]).all()) for k in ("target", "floor", "bin_power"))
            print(f"   outputs finite = {finite}; realized drive / nominal = {out['realized']['realized_over_nominal']:.5f}")
    sham_close = timed("closing sham", s.sham_differences, ids)
    print(f"== sham differing values: opening {sham}, closing {sham_close} (both must be 0)")
    print("== smoke test PASSED" if sham == 0 and sham_close == 0 else "== smoke test FAILED")


if __name__ == "__main__":
    main()
