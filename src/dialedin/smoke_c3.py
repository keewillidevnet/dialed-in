"""Pre-seal dry run of the C3 harness on the pod. Writes nothing to results/.

    PYTHONPATH=src python3 -m dialedin.smoke_c3 --spec criteria/drafts/c3-glimmer-head-dial-v1.yaml

Run from the repo root. Prints the head-scan file's SHA-256, the selected and control heads per layer,
and the control size ratio, so the hash and ratio can be written into the spec before sealing. Then
builds the real C3 Session (seal and code checks skipped) and, under the five Phase A conditions, runs the undriven pass, a sham and driven passes at N = 2 and
4096, and a closing sham. It prints pass time, peak memory, sham counts and whether outputs are
finite. It prints no gains or SNRs.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import time
import types

import numpy as np
import torch
import yaml

from . import glimmer, run_c3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", default="/workspace/weights/Muse-Glimmer-30B")
    ap.add_argument("--manifest", default="/workspace/data/books/MANIFEST.json")
    args = ap.parse_args()
    spec = yaml.safe_load(open(args.spec))
    sel = spec["heads"]["selection"]
    sha = glimmer.sha256_file(sel["file"])
    scan = json.load(open(sel["file"]))
    picked = run_c3.selected_heads(scan)
    control, ratio = run_c3.control_heads(scan, picked)
    comp = lambda hs: ", ".join(f"{l}:{sum(1 for x, _ in hs if x == l)}" for l in run_c3.SELECT_LAYERS)
    print(f"== head scan sha256: {sha}")
    print(f"== selected heads by layer {comp(picked)}; control heads by layer {comp(control)}")
    print(f"== control size ratio (summed size, control over selected): {ratio!r}")
    print(f"== composition for the spec: selected {run_c3.composition(picked)}, control {run_c3.composition(control)}, "
          f"layer gap {run_c3.composition_gap(picked, control)}")
    spec["heads"]["selection"]["sha256"] = sha                 # smoke only: the sealed file will carry these values
    spec["heads"]["control_size_ratio"] = ratio
    spec["heads"]["composition"] = {"selected": run_c3.composition(picked), "control": run_c3.composition(control)}
    s = object.__new__(run_c3.Session)
    s.repo, s.seal_hash, s.spec, s.code = os.getcwd(), "smoke", spec, {}
    s.ledger = types.SimpleNamespace(check=lambda *a: None, add=lambda *a: 0.0, last_pass=0.0)
    s.texts = glimmer.load_texts(spec, args.manifest, os.path.dirname(args.manifest))
    glimmer.configure_determinism()
    t = time.time()
    s.model, s.decoder, s.layers, s.tok, s.platform = glimmer.load(spec, args.weights)
    print(f"== load {time.time() - t:.0f} s; preflight passed: {s.platform}")
    run_c3.Session._setup(s)
    ids, _ = s.ids(run_c3.PHASE_A_BOOK)
    torch.cuda.reset_peak_memory_stats()

    def timed(label, fn, *a):
        t = time.time()
        out = fn(*a)
        print(f"   {label}: {time.time() - t:.2f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        return out

    shams, norms = [], []
    for cond in run_c3.PHASE_A_CONDITIONS:
        name = run_c3.cond_name(cond)
        s.set_condition(cond)
        s.slice_norms.armed = cond == ("S", 1.0)
        norms.append(timed(f"{name} undriven", s.baseline, ids))
        if s.slice_norms.armed:
            print(f"   slice-norm probe ran on {len(s.slice_norms.values)} heads")
        s.slice_norms.armed = False
        shams.append(timed(f"{name} sham", s.sham_differences, ids))
        for N in (2, 4096):
            out = timed(f"{name} driven N={N}", s.measure, ids, N, run_c3.A * norms[0], 0.0, "smoke")
            finite = all(bool(np.isfinite(out[k]).all()) for k in ("target", "floor", "bin_power"))
            print(f"   outputs finite = {finite}")
    shams.append(timed("closing sham", s.sham_differences, ids))
    s.set_condition(("S", 1.0))
    same_norm = len(set(norms)) == 1
    print(f"== sham differing values: {shams} (all must be 0); amplitude reference unchanged by knob: {same_norm}")
    print("== smoke test PASSED" if all(x == 0 for x in shams) and same_norm else "== smoke test FAILED")


if __name__ == "__main__":
    main()
