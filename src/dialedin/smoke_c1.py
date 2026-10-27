"""Pre-seal dry run of the C1 harness on the pod. Writes nothing to results/.

    PYTHONPATH=src python3 -m dialedin.smoke_c1 --spec criteria/drafts/c1-glimmer-frequency-response-v4.yaml

Checks that strict determinism does not stop the forward pass, that the sham is bit exact,
and how long a hooked pass takes and how much memory it needs. It prints no gains or SNRs,
so it reveals nothing about the outcome before the criteria are sealed.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import time

import numpy as np
import torch
import yaml

from . import glimmer, metrics
from .hooks import HeadProbe, PulseInjector, ResidualNormProbe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", default="/workspace/weights/Muse-Glimmer-30B")
    ap.add_argument("--manifest", default="/workspace/data/books/MANIFEST.json")
    args = ap.parse_args()
    spec = yaml.safe_load(open(args.spec))
    texts = glimmer.load_texts(spec, args.manifest, os.path.dirname(args.manifest))
    glimmer.configure_determinism()
    t = time.time()
    model, decoder, layers, tok, platform = glimmer.load(spec, args.weights)
    print(f"== load {time.time() - t:.0f} s; preflight passed: {platform}")
    T = spec["inputs"]["length_tokens"]
    ids = tok(texts[1342], add_special_tokens=False)["input_ids"][:T]
    ids = torch.tensor([ids], device="cuda")
    g = torch.Generator().manual_seed(0)
    d = torch.randn(layers[1].self_attn.o_proj.out_features, generator=g, dtype=torch.float32)
    d = (d / d.norm()).cuda()
    inj = PulseInjector(layers[1], d)
    norms = ResidualNormProbe(layers[1])
    probe = HeadProbe(layers, range(2, 52), 32, metrics.WINDOW_START)
    torch.cuda.reset_peak_memory_stats()

    def run(what):
        t = time.time()
        with torch.inference_mode():
            decoder(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
        print(f"   {what}: {time.time() - t:.2f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

    probe.mode = "record"; norms.armed = True
    run("unpulsed")
    norms.armed = False
    m = float(np.median(norms.norms))
    inj.arm(metrics.periodic_positions(16, T), 0.0); probe.mode = "zero_check"
    run("sham")
    inj.disarm()
    sham = int(sum(probe.result.values())); probe.result = {}
    for N in (2, 4096):
        probe.set_frequencies(N, T, "cuda")
        inj.arm(metrics.periodic_positions(N, T), 0.05 * m); probe.mode = "measure"
        run(f"pulsed N={N}")
        inj.disarm()
        out = probe.collect()
        finite = all(bool(np.isfinite(out[k]).all()) for k in ("target", "floor", "bin_power"))
        print(f"   pulsed N={N}: outputs finite = {finite}")
    inj.arm(metrics.periodic_positions(16, T), 0.0); probe.mode = "zero_check"
    run("closing sham")
    inj.disarm()
    sham_close = int(sum(probe.result.values())); probe.result = {}
    print(f"== sham differing values: opening {sham}, closing {sham_close} (both must be 0)")
    print("== smoke test PASSED" if sham == 0 and sham_close == 0 else "== smoke test FAILED")


if __name__ == "__main__":
    main()
