"""Exploratory: does a head's listening range (from sine drives) match where it attends (from attention weights)?

    PYTHONPATH=src python3 -m dialedin.headmap_attention

Joins two independent measurements head by head: the head filter map (results/derived/headmap-c1v5-20261027T152040Z,
C1 v5 sine drives, 15 books, first 32,768 tokens) and the mechanism probe's base-model attention statistics
(results/raw/probe-mech-20261029T065204Z, eager attention at scale 0, set C books 74, 120 and 174, tokens 60,000 to
64,095, queries 2,048 to 4,095). Reports Spearman correlations over heads, and within layers (ranks within each layer
with at least 5 resolved heads, pooled), and attention statistics per listening-range band. No GPU.
Output: results/derived/headmap-attention/compare.json (never overwritten).
"""
import argparse
import collections
import json
import os

import numpy as np

from .metrics import spearman

STATS = ("mean_distance", "near_mass", "entropy", "first_token_mass", "head_output_share")
BANDS = (("corner_16_or_less", 0, 16), ("corner_24_to_96", 24, 96), ("corner_256_or_more", 256, float("inf")))


def join(hm, op):
    rows = []
    for i, l in enumerate(hm["per_head"]["layers"]):
        r = op[str(l)]
        for h in range(len(hm["per_head"]["tilt"][i])):
            rows.append({"layer": l, "head": h, "corner": hm["per_head"]["corner_tokens"][i][h],
                         "tilt": hm["per_head"]["tilt"][i][h], **{k: r[k][h] for k in STATS}})
    return rows


def within_layer_r(rows, x, y):
    by = collections.defaultdict(list)
    for r in rows:
        if r[x] is not None:
            by[r["layer"]].append(r)
    xs, ys = [], []
    for rs in by.values():
        if len(rs) >= 5:
            rank = lambda v: np.argsort(np.argsort(v)) / (len(v) - 1)
            xs += list(rank([r[x] for r in rs])); ys += list(rank([r[y] for r in rs]))
    return float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else None, len(xs)


def compare(rows):
    out = {}
    for x in ("corner", "tilt"):
        sel = [r for r in rows if r[x] is not None]
        for k in STATS:
            w, n_w = within_layer_r(rows, x, k)
            out[f"{x}_vs_{k}"] = {"spearman": spearman([r[x] for r in sel], [r[k] for r in sel]), "heads": len(sel),
                                  "within_layer_r": w, "within_layer_heads": n_w}
    bands = {}
    for name, lo, hi in BANDS:
        sel = [r for r in rows if r["corner"] is not None and lo <= r["corner"] <= hi]
        bands[name] = {"heads": len(sel), **{f"median_{k}": float(np.median([r[k] for r in sel])) for k in STATS}}
    return {"correlations": out, "bands": bands}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--headmap", default="results/derived/headmap-c1v5-20261027T152040Z/headmap.json")
    ap.add_argument("--mech", default="results/raw/probe-mech-20261029T065204Z/mechanisms.json")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    hm = json.load(open(os.path.join(repo, args.headmap)))
    op = json.load(open(os.path.join(repo, args.mech)))["operating_point"]["S0"]
    res = compare(join(hm, op))
    res.update({"exploratory": True, "what": __doc__.split("\n\n")[2].strip(), "cites": [args.headmap, args.mech]})
    out = os.path.join(repo, "results", "derived", "headmap-attention")
    if os.path.exists(os.path.join(out, "compare.json")):
        raise SystemExit(f"{out}/compare.json exists; derived results are not overwritten.")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "compare.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    for k, v in res["correlations"].items():
        print(f"   {k:32s} spearman {v['spearman']:+.3f} (n {v['heads']}), within layer {v['within_layer_r']:+.3f}")
    for k, v in res["bands"].items():
        print(f"   {k:20s} heads {v['heads']:3d}, median distance {v['median_mean_distance']:.1f}, "
              f"last-16 mass {v['median_near_mass']:.3f}, output share {v['median_head_output_share']:.4f}")
    print(f"== written: {os.path.relpath(out, repo)}/compare.json")


if __name__ == "__main__":
    main()
