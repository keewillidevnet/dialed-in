"""Exploratory pilot for C1 v5: can a sine drive be measured above BF16 rounding noise? Not a sealed run.

    PYTHONPATH=src python3 -m dialedin.pilot_sine

Why: the sealed C1 v4 run stopped in Phase A (results/raw/c1v4-20261027T064341Z/phaseA.json).
With one pulse every N tokens, the response at N = 128 and 1024 stayed on an amplitude-independent
floor (SNR about 1). A pulse train puts only 1/N of its energy at the target frequency. A sine drive
pushes every token, so the target component builds over the whole window at every N.

What it does: on a text excerpt that no C1 run uses (Moby Dick, book 2701, tokens 98,304 to
131,071; C1 uses only the first 32,768 tokens of each book), it adds a * m * cos(2 pi t / N) along
the fixed unit direction at the layer 1 output, for every token t, at a and 2a. It prints only the
2a/a response ratio (linearity) and the whole-model SNR. It never prints a gain, so it cannot reveal
the shape of the response before v5 is sealed. Output: /workspace/env/pilot_sine.json.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import time

import numpy as np
import torch
import yaml

from . import glimmer, metrics
from .hooks import HeadProbe, ResidualNormProbe

SPEC = "criteria/sealed/c1-glimmer-frequency-response-v4.yaml"     # platform and text hashes only
BOOK, START = 2701, 98304
PILOT_N = (16, 128, 1024, 4096)
AMPS = (0.01, 0.03, 0.1, 0.3)


class WaveInjector:
    """Adds w[t] * direction at every token of one layer's output, in float32, cast once to the model dtype.

    Records the realized drive along the direction, sum_d (after - before) * dir, per token, so the
    input component is computed from what BF16 actually stored, not the nominal wave.
    """

    def __init__(self, layer, direction):
        self.direction = direction
        self.w = None
        self.realized = None
        self.handle = layer.register_forward_hook(self._hook)

    def arm(self, w: np.ndarray):
        self.w = torch.as_tensor(w, dtype=torch.float32, device=self.direction.device)

    def disarm(self):
        self.w = None

    def _hook(self, module, args, output):
        if self.w is None:
            return None
        hidden = output[0] if isinstance(output, tuple) else output
        before = hidden[0].detach().float()
        delta = self.w[:, None] * self.direction[None, :]
        after = torch.where(delta == 0, before, before + delta).to(hidden.dtype)
        self.realized = ((after.float() - before) @ self.direction).double().cpu().numpy()
        new = hidden.clone()
        new[0] = after
        return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/workspace/weights/Muse-Glimmer-30B")
    ap.add_argument("--manifest", default="/workspace/data/books/MANIFEST.json")
    ap.add_argument("--out", default="/workspace/env/pilot_sine.json")
    args = ap.parse_args()
    spec = yaml.safe_load(open(SPEC))
    texts = glimmer.load_texts(spec, args.manifest, os.path.dirname(args.manifest))
    glimmer.configure_determinism()
    model, decoder, layers, tok, platform = glimmer.load(spec, args.weights)
    T, t1 = metrics.WINDOW_END, metrics.WINDOW_START
    ids = tok(texts[BOOK], add_special_tokens=False)["input_ids"][START:START + T]
    assert len(ids) == T, len(ids)
    ids = torch.tensor([ids], device="cuda")
    g = torch.Generator().manual_seed(0)
    d = torch.randn(layers[1].self_attn.o_proj.out_features, generator=g, dtype=torch.float32)
    d = (d / d.norm()).cuda()
    inj = WaveInjector(layers[1], d)
    norms = ResidualNormProbe(layers[1])
    probe = HeadProbe(layers, range(2, 52), 32, t1)

    def run():
        t = time.time()
        with torch.inference_mode():
            decoder(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
        return time.time() - t

    probe.mode = "record"; norms.armed = True
    run()
    norms.armed = False
    m = float(np.median(norms.norms))
    inj.arm(np.zeros(T)); probe.mode = "zero_check"
    run(); inj.disarm()
    sham = int(sum(probe.result.values())); probe.result = {}
    print(f"== book {BOOK} tokens {START}..{START + T - 1}; median norm {m:.2f}; sham differing values {sham}")
    t = np.arange(T, dtype=np.float64)
    rows = []
    for N in PILOT_N:
        cos_b, sin_b = probe.set_frequencies(N, T, "cuda")
        for a in AMPS:
            resp, snr, drive_err = [], None, None
            for scale in (1, 2):
                w = a * scale * m * np.cos(2 * np.pi * t / N)
                inj.arm(w); probe.mode = "measure"
                sec = run(); inj.disarm()
                out = probe.collect()
                realized = inj.realized
                inp = float(metrics.component(realized[t1:], cos_b[:, :1], sin_b[:, :1])[0])
                nominal = float(metrics.component(w[t1:], cos_b[:, :1], sin_b[:, :1])[0])
                resp.append(float(out["target"].mean()))
                if scale == 1:
                    snr = metrics.aggregate_snr(out["target"], out["bin_power"], out["n_below"])
                    drive_err = inp / nominal - 1
            ratio = resp[1] / resp[0]
            rows.append({"N": N, "a": a, "ratio": ratio, "snr": snr, "realized_over_nominal_minus_1": drive_err})
            print(f"   N={N:<5} a={a:<5} ratio={ratio:.3f}  SNR={snr:8.2f}  drive error {drive_err:+.3%}  ({sec:.1f} s/pass)",
                  flush=True)
    json.dump({"book": BOOK, "start": START, "T": T, "median_norm": m, "sham": sham, "platform": platform,
               "rows": rows}, open(args.out, "w"), indent=1)
    print(f"== wrote {args.out}")


if __name__ == "__main__":
    main()
