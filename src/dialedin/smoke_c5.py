"""Pre-seal dry run of the C5 harness on the pod. Writes nothing to results/ (scratch goes to /workspace/env).

    PYTHONPATH=src python3 -m dialedin.smoke_c5 --spec criteria/drafts/c5-glimmer-trained-dial-v1.yaml

1. Phase T mechanics: two training steps with the sealed recipe on the sealed training regions (time, peak
   memory, finiteness). The adapter it writes is a throwaway: two steps decide nothing.
2. Reloads Glimmer as a new process would, builds the real C5 Session on that throwaway adapter (seal and code
   checks skipped), and runs the adapter checks and the perplexity-path check.
3. Undriven, sham and driven passes (N = 2 and 4096) under S0 and S2, and a closing sham.
4. One recall forward and one topic-switch forward (time and finiteness only).
Prints times, peak memory, sham counts, finiteness and S0 perplexity only. No gains, SNRs, tilts, recall scores, or
perplexity under the adapter, since a throwaway adapter says nothing about the sealed run anyway.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import copy
import gc
import shutil
import time
import types

import numpy as np
import torch
import yaml

from . import glimmer, run_c5
from . import pilot_c5 as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", default="/workspace/weights/Muse-Glimmer-30B")
    ap.add_argument("--manifest", default="/workspace/data/books/MANIFEST.json")
    ap.add_argument("--scratch", default="/workspace/env/smoke_c5")
    args = ap.parse_args()
    spec = yaml.safe_load(open(args.spec))
    shutil.rmtree(args.scratch, ignore_errors=True)
    os.makedirs(args.scratch)
    fake_ledger = types.SimpleNamespace(check=lambda *a: None, add=lambda *a: 0.0, last_pass=0.0)

    # 1. two training steps
    texts = glimmer.load_texts(spec, args.manifest, os.path.dirname(args.manifest))
    glimmer.configure_determinism()
    t = time.time()
    model, decoder, layers, tok, platform = glimmer.load(spec, args.weights)
    print(f"== load {time.time() - t:.0f} s; preflight passed: {platform}")
    for reg in spec["training"]["regions"]:
        n = len(tok(texts[int(reg["gutenberg_id"])], add_special_tokens=False)["input_ids"])
        print(f"   training region {reg['gutenberg_id']}@{reg['start']}: book has {n} tokens "
              f"(needs {reg['start'] + spec['training']['tokens']})")
    short = copy.deepcopy(spec)
    short["training"]["steps"] = 2
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    run_c5.train(short, "smoke", {}, fake_ledger, args, args.scratch,
                 loaded=(texts, model, decoder, layers, tok, platform))
    print(f"== 2 training steps (with the base-gain cache): {time.time() - t:.0f} s, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    del model, decoder, layers
    gc.collect(); torch.cuda.empty_cache()
    P.configure(8192)
    P.LoRALinear.scale = 0.0

    # 2. a fresh model, as phases A and B load it (deterministic again, as in a new process)
    glimmer.configure_determinism()
    s = object.__new__(run_c5.Session)
    s.repo, s.seal_hash, s.spec, s.code = os.getcwd(), "smoke", spec, {}
    s.ledger = fake_ledger
    s.texts = texts
    s.model, s.decoder, s.layers, s.tok, s.platform = glimmer.load(spec, args.weights)
    run_c5.Session._setup(s, args.scratch)
    print(f"== adapter on {len(s.adapter_modules)} modules; softcap {s.softcap}; output multiplier {s.multiplier}")
    ad = s.adapter_checks()
    print(f"== adapter checks: {ad}")
    lc = s.logits_check()
    print(f"== perplexity path: max |logit difference| {lc['max_abs_diff']:.4g} of max |logit| "
          f"{lc['max_abs_logit']:.4g} (tolerance {lc['tolerance']:.4g}); as expected: {lc['as_expected']}")
    torch.cuda.reset_peak_memory_stats()

    def timed(label, fn, *a):
        t = time.time()
        out = fn(*a)
        print(f"   {label}: {time.time() - t:.2f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        return out

    # 3. passes under S0 and S2
    shams = []
    ids, _ = s.ids(run_c5.PHASE_A_BOOK)
    for cond in ("S0", "S2"):
        s.set_condition(cond)
        m, ppl = timed(f"{cond} undriven + perplexity", s.baseline, ids)
        if cond == "S0":
            print(f"   S0 perplexity on book 1342, tokens 8193 to 32767: {ppl:.3f}")
        shams.append(timed(f"{cond} sham", s.sham_differences, ids))
        for N in (2, 4096):
            out = timed(f"{cond} driven N={N}", s.measure, ids, N, run_c5.A * m, 0.0, "smoke")
            print(f"   outputs finite = {all(bool(np.isfinite(out[k]).all()) for k in ('target', 'floor', 'bin_power'))}")
    shams.append(timed("closing sham", s.sham_differences, ids))
    s.set_condition("S0")

    # 4. recall and switch timing (hooks off, as phase R sets them)
    s.probe.mode = None
    s.injector.disarm()
    trials = run_c5.recall_trials(s)
    print(f"== recall: {len(trials)} trials of {len(trials[0]['ids'])} tokens; answer {trials[0]['answer_len']} tokens")
    t = time.time()
    with torch.inference_mode():
        s.decoder(input_ids=torch.tensor([trials[0]["ids"]], device=s.device), use_cache=False)
    print(f"   one recall forward: {time.time() - t:.2f} s")
    sw = run_c5.switch_inputs(s)
    print(f"== switch: {len(sw['pairs'])} ordered pairs, {len(sw['clean'])} clean and {len(sw['noise'])} noise inputs "
          f"of {len(next(iter(sw['clean'].values())))} tokens")
    t = time.time()
    first = next(iter(sw["clean"].values()))
    loss = run_c5.token_losses(s, first, run_c5.SWITCH["context"], len(first))
    print(f"   one switch forward with per-token loss: {time.time() - t:.2f} s; losses finite = {bool(np.isfinite(loss).all())}")
    print(f"== sham differing values: {shams} (all must be 0)")
    ok = ad["as_expected"] and lc["as_expected"] and all(x == 0 for x in shams)
    print("== smoke test PASSED" if ok else "== smoke test FAILED")


if __name__ == "__main__":
    main()
