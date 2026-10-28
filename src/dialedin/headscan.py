"""Exploratory head scan for C3, from the sealed C1 v5 raw data. Set A only. No GPU.

    PYTHONPATH=src python3 -m dialedin.headscan --run-id c1v5-20261027T152040Z

Reads only the five set A book files of the C1 v5 run, so the head choice for C3 never sees sets B
and C, which C3 will use as its held-out test. For each of the 1,600 measured heads it computes the
gain curve (target magnitude over input component, mean over the five books) and two rankings:

  tilt: least-squares slope of ln(gain) against ln(N) over all 23 periods (how low-pass the head is)
  low:  mean gain at N >= 1024 (how large its slow response is)

It also reports how concentrated the slow response is (share of summed low-frequency gain held by the
top k heads) and how much of the whole-model tilt the top k account for arithmetically. That last
number is a caution, not a result: removing heads from the average changes the average by
construction, which is why C3 measures only heads it does not touch.

Output: results/derived/headscan-setA-<run-id>/headscan.json (never overwritten).
"""
import argparse
import glob
import json
import os
import re

import numpy as np


def slope(x, y):
    x = x - x.mean()
    return (x[:, None] * (y - y.mean(axis=0))).sum(axis=0) / (x * x).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="c1v5-20261027T152040Z")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--set", default="A")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    run_dir = os.path.join(args.repo, "results", "raw", args.run_id)
    paths = sorted(p for p in glob.glob(os.path.join(run_dir, f"book_{args.set}_*.json"))
                   if re.fullmatch(rf"book_{args.set}_\d+\.json", os.path.basename(p)))
    if len(paths) != 5:
        raise SystemExit(f"expected 5 set {args.set} books, found {len(paths)}")
    books = [json.load(open(p)) for p in paths]
    grid = sorted(int(n) for n in books[0]["per_N"])
    layers = books[0]["record"]["layers_measured"]
    global_ids = set(books[0]["global_layers"])
    # gain[b, j, layer, head]
    gain = np.stack([np.stack([np.asarray(b["per_N"][str(N)]["target"], dtype=np.float64) / b["per_N"][str(N)]["input_mag"]
                               for N in grid]) for b in books])
    g = gain.mean(axis=0)                                   # [N, layers, heads]
    L, H = g.shape[1], g.shape[2]
    flat = g.reshape(len(grid), L * H)
    lnN = np.log(np.asarray(grid, dtype=np.float64))
    tilt = slope(lnN, np.log(np.maximum(flat, 1e-30)))
    longmask = np.asarray(grid) >= 1024
    low = flat[longmask].mean(axis=0)
    whole = flat.mean(axis=1)
    whole_tilt = float(slope(lnN, np.log(whole)[:, None])[0])

    def label(i):
        layer, head = layers[i // H], i % H
        return {"layer": int(layer), "head": int(head), "kind": "global" if layer in global_ids else "local"}

    def table(score, name):
        order = np.argsort(-score)
        rows = [dict(label(i), rank=r + 1, tilt=float(tilt[i]), low_gain=float(low[i])) for r, i in enumerate(order)]
        top = rows[: args.top]
        print(f"\n== top {args.top} heads by {name}")
        for r in top:
            print(f"   {r['rank']:>3}  layer {r['layer']:>2} head {r['head']:>2} {r['kind']:<6}  tilt {r['tilt']:+.3f}  low-freq gain {r['low_gain']:.4g}")
        return rows

    by_tilt = table(tilt, "tilt (most low-pass)")
    by_low = table(low, "low-frequency gain (largest slow response)")
    conc = {}
    order_low = np.argsort(-low)
    for k in (8, 16, 32, 64, 128):
        keep = np.ones(L * H, bool); keep[order_low[:k]] = False
        rest_tilt = float(slope(lnN, np.log(flat[:, keep].mean(axis=1))[:, None])[0])
        conc[k] = {"share_of_low_gain": float(low[order_low[:k]].sum() / low.sum()),
                   "whole_tilt_without_topk_arithmetic": rest_tilt}
    print(f"\n== whole-model tilt (set {args.set}, all 23 periods): {whole_tilt:.4f}")
    print("== concentration of the slow response (top k by low-frequency gain)")
    for k, v in conc.items():
        print(f"   k={k:<4} share {v['share_of_low_gain']:.1%}   tilt of the remaining heads' average {v['whole_tilt_without_topk_arithmetic']:.4f}")
    kinds = np.array([label(i)["kind"] for i in range(L * H)])
    print("== median head tilt: local %.3f, global %.3f" % (np.median(tilt[kinds == "local"]), np.median(tilt[kinds == "global"])))
    out_dir = os.path.join(args.repo, "results", "derived", f"headscan-set{args.set}-{args.run_id}")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "headscan.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; derived results are not overwritten.")
    json.dump({"run_id": args.run_id, "set": args.set, "books": [os.path.relpath(p, args.repo) for p in paths],
               "grid": grid, "whole_tilt": whole_tilt, "by_tilt": by_tilt, "by_low_gain": by_low,
               "concentration": conc,
               "median_tilt": {"local": float(np.median(tilt[kinds == "local"])),
                               "global": float(np.median(tilt[kinds == "global"]))}},
              open(out, "w"), indent=1)
    print(f"\n== wrote {os.path.relpath(out, args.repo)}")


if __name__ == "__main__":
    main()
