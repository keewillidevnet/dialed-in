"""Pre-seal dry run of the C2 harness on the pod. Writes nothing to results/.

    PYTHONPATH=src python3 -m dialedin.smoke_c2 --spec criteria/drafts/c2-glimmer-knob-tilt-v1.yaml

Builds the real C2 Session (seal and code checks skipped), then under G1.0, G0.5, G1.5 and L0.5 runs the
undriven pass, a sham and driven passes at N = 2 and 4096, and a closing sham. It checks that the
knob leaves the layer 1 amplitude reference unchanged. It prints pass time, peak memory, sham counts
and whether outputs are finite. It prints no gains or SNRs.
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

from . import glimmer, run_c2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", default="/workspace/weights/Muse-Glimmer-30B")
    ap.add_argument("--manifest", default="/workspace/data/books/MANIFEST.json")
    args = ap.parse_args()
    spec = yaml.safe_load(open(args.spec))
    s = object.__new__(run_c2.Session)
    s.repo, s.seal_hash, s.spec, s.code = os.getcwd(), "smoke", spec, {}
    s.ledger = types.SimpleNamespace(check=lambda *a: None, add=lambda *a: 0.0, last_pass=0.0)
    s.texts = glimmer.load_texts(spec, args.manifest, os.path.dirname(args.manifest))
    glimmer.configure_determinism()
    t = time.time()
    s.model, s.decoder, s.layers, s.tok, s.platform = glimmer.load(spec, args.weights)
    print(f"== load {time.time() - t:.0f} s; preflight passed: {s.platform}")
    run_c2.Session._setup(s)
    ids, _ = s.ids(run_c2.PHASE_A_BOOK)
    torch.cuda.reset_peak_memory_stats()

    def timed(label, fn, *a):
        t = time.time()
        out = fn(*a)
        print(f"   {label}: {time.time() - t:.2f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        return out

    shams, norms = [], []
    for cond in (("G", 1.0), ("G", 0.5), ("G", 1.5), ("L", 0.5)):
        name = run_c2.cond_name(cond)
        s.set_condition(cond)
        s.knob_norms.armed = cond == ("G", 1.0)
        norms.append(timed(f"{name} undriven", s.baseline, ids))
        if s.knob_norms.armed:
            k = s.knob_norms.summary()
            print(f"   knob-size probe ran on {len(k['per_layer'])} layers")
        s.knob_norms.armed = False
        shams.append(timed(f"{name} sham", s.sham_differences, ids))
        for N in (2, 4096):
            out = timed(f"{name} driven N={N}", s.measure, ids, N, run_c2.A * norms[0], 0.0, "smoke")
            finite = all(bool(np.isfinite(out[k]).all()) for k in ("target", "floor", "bin_power"))
            print(f"   outputs finite = {finite}")
    shams.append(timed("closing sham", s.sham_differences, ids))
    s.set_condition(("G", 1.0))
    same_norm = len(set(norms)) == 1
    print(f"== sham differing values: {shams} (all must be 0); amplitude reference unchanged by knob: {same_norm}")
    print("== smoke test PASSED" if all(x == 0 for x in shams) and same_norm else "== smoke test FAILED")


if __name__ == "__main__":
    main()
