"""Exploratory head filter map from the sealed C1 v5 raw data. No GPU; nothing here gates anything.

    PYTHONPATH=src python3 -m dialedin.headmap --run-id c1v5-20261027T152040Z

For each of the 1,600 heads in layers 2 to 51, pools the 15 books (sets A, B, C) of C1 v5's main drive (23 periods,
N = 2 to 4,096) into one noise-corrected gain curve: per period, the book mean of target power over input power minus
the book mean of noise power over input power (noise power from the per-head floor, as metrics.gains), square root.
A head-period is resolved when the pooled amplitude SNR, sqrt(mean signal power / mean noise power), is at least 3.

Per head (reported only when resolved at every one of the 23 periods):
  tilt    least-squares slope of ln(gain) on ln(N) (the whole-model tilt, one head at a time)
  corner  the shortest period at which the head's log gain, smoothed over 3 neighboring periods, reaches the midpoint
          between its lowest and highest values (its listening range); only for heads whose gain rises by 1.5x or more
Stability: the same two numbers from each set of 5 books alone (resolved at SNR >= 2, since a set has a third of the
data), and their correlation across sets, over all heads and within layers (each layer's mean removed).
Output: results/derived/headmap-<run-id>/headmap.json and headmap.png (never overwritten).
"""
import argparse
import glob
import json
import os
import re
from itertools import combinations

import numpy as np

from . import metrics

SNR_ALL, SNR_SET = 3.0, 2.0
MIN_RISE = 1.5


def load(run_dir):
    paths = sorted(p for p in glob.glob(os.path.join(run_dir, "book_*_*.json"))
                   if re.fullmatch(r"book_[A-Z]_\d+\.json", os.path.basename(p)))
    mom = metrics.median_over_mean_power(2 * metrics.HEAD_DIM)
    sig, noi, sets, grid, glob_layers, layer_ids = [], [], [], None, None, None
    for p in paths:
        d = json.load(open(p))
        g = sorted(int(k) for k in d["per_N"])
        if grid is None:
            grid, glob_layers = g, d["global_layers"]
        assert g == grid
        sets.append(d["set"])
        cells = [d["per_N"][str(N)] for N in grid]
        sig.append([np.asarray(c["target"], dtype=np.float64) ** 2 / c["input_mag"] ** 2 for c in cells])
        noi.append([np.asarray(c["floor"], dtype=np.float64) ** 2 / mom / c["input_mag"] ** 2 for c in cells])
    return np.array(sig), np.array(noi), sets, grid, glob_layers, [os.path.relpath(p) for p in paths]


def pooled(sig, noi, idx):
    """Noise-corrected gain [N, layers, heads] and pooled amplitude SNR over the books in idx."""
    s, n = sig[idx].mean(0), noi[idx].mean(0)
    return np.sqrt(np.maximum(s - n, 0.0)), np.sqrt(s / np.maximum(n, 1e-300))


def head_stats(gain, snr, grid, thr):
    """tilt and corner per head [layers, heads], NaN where a head is not resolved at every period."""
    x = np.log(np.asarray(grid, dtype=np.float64))
    ok = np.all((snr >= thr) & (gain > 0), axis=0)
    tilt = np.full(ok.shape, np.nan)
    corner = np.full(ok.shape, np.nan)
    for l, h in zip(*np.nonzero(ok)):
        y = np.log(gain[:, l, h])
        tilt[l, h] = np.polyfit(x, y, 1)[0]
        ys = np.convolve(np.pad(y, 1, mode="edge"), np.ones(3) / 3, "valid")
        lo, hi = ys.min(), ys.max()
        if hi - lo >= np.log(MIN_RISE):
            corner[l, h] = grid[int(np.argmax(ys >= (lo + hi) / 2))]
    return tilt, corner


def stability(a, b, log=False):
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 3:
        return {"n": int(m.sum()), "r": None, "r_within_layer": None}
    fa, fb = (np.log(a), np.log(b)) if log else (a, b)
    ra, rb = [], []
    for l in range(a.shape[0]):
        mm = m[l]
        if mm.sum() >= 4:
            ra += list(fa[l, mm] - fa[l, mm].mean()); rb += list(fb[l, mm] - fb[l, mm].mean())
    return {"n": int(m.sum()), "r": float(np.corrcoef(fa[m], fb[m])[0, 1]),
            "r_within_layer": float(np.corrcoef(ra, rb)[0, 1]) if len(ra) > 2 else None}


def summarize(tilt, corner, layer_ids, glob_layers):
    isg = np.array([l in set(glob_layers) for l in layer_ids])
    per_layer = []
    for i, l in enumerate(layer_ids):
        t, c = tilt[i][~np.isnan(tilt[i])], corner[i][~np.isnan(corner[i])]
        per_layer.append({"layer": l, "global": bool(isg[i]), "heads_resolved": int(t.size),
                          "tilt_median": float(np.median(t)) if t.size else None,
                          "corner_tokens": sorted(int(v) for v in c)})
    t_all = tilt[~np.isnan(tilt)]
    c_all = corner[~np.isnan(corner)]
    return {"heads_resolved": int(t_all.size), "heads_total": int(tilt.size),
            "heads_with_corner": int(c_all.size),
            "tilt_percentiles_5_25_50_75_95": np.percentile(t_all, [5, 25, 50, 75, 95]).tolist() if t_all.size else None,
            "heads_with_negative_tilt": int((t_all < 0).sum()),
            "tilt_median_local_layers": float(np.nanmedian(tilt[~isg])) if np.any(~np.isnan(tilt[~isg])) else None,
            "tilt_median_global_layers": float(np.nanmedian(tilt[isg])) if np.any(~np.isnan(tilt[isg])) else None,
            "corner_percentiles_10_25_50_75_90": np.percentile(c_all, [10, 25, 50, 75, 90]).tolist() if c_all.size else None,
            "per_layer": per_layer}


def plot(tilt, corner, layer_ids, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2), sharey=True)
    order = np.argsort(np.nan_to_num(corner, nan=1e9), axis=1)               # heads sorted by corner within a layer
    for ax, data, title, kw in ((axes[0], np.log2(corner), "Listening range per head (tokens)",
                                 {"cmap": "viridis", "vmin": 1, "vmax": 12}),
                                (axes[1], tilt, "Tilt per head (slope of ln gain on ln period)",
                                 {"cmap": "magma", "vmin": 0, "vmax": 0.35})):
        im = ax.imshow(np.take_along_axis(data, order, axis=1), aspect="auto", interpolation="nearest",
                       extent=(-0.5, data.shape[1] - 0.5, layer_ids[-1] + 0.5, layer_ids[0] - 0.5), **kw)
        ax.set_title(title, fontsize=10); ax.set_xlabel("heads in the layer, sorted by listening range")
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        if ax is axes[0]:
            cb.set_ticks([1, 3, 5, 7, 9, 11]); cb.set_ticklabels(["2", "8", "32", "128", "512", "2048"])
    axes[0].set_ylabel("layer (gray: below the noise floor)")
    for ax in axes:
        ax.set_facecolor("#cccccc")
    fig.suptitle("Muse Glimmer-30B: each attention head listens at its own timescale (C1 v5 data, 15 books)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="c1v5-20261027T152040Z")
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    run_dir = os.path.join(repo, "results", "raw", args.run_id)
    cwd = os.getcwd(); os.chdir(repo)
    sig, noi, sets, grid, glob_layers, cites = load(os.path.relpath(run_dir, repo))
    os.chdir(cwd)
    if len(sets) != 15:
        raise SystemExit(f"expected 15 books, found {len(sets)}")
    layer_ids = list(range(2, 2 + sig.shape[2]))
    tilt, corner = head_stats(*pooled(sig, noi, list(range(len(sets)))), grid, SNR_ALL)
    per_set = {s: head_stats(*pooled(sig, noi, [i for i, t in enumerate(sets) if t == s]), grid, SNR_SET)
               for s in sorted(set(sets))}
    stab = {f"{a}_{b}": {"tilt": stability(per_set[a][0], per_set[b][0]),
                         "corner_log": stability(per_set[a][1], per_set[b][1], log=True)}
            for a, b in combinations(sorted(per_set), 2)}
    result = {"exploratory": True, "what": __doc__.split("\n\n")[2].strip(), "run_id": args.run_id, "grid": grid,
              "books": len(sets), "snr_threshold_pooled": SNR_ALL, "snr_threshold_per_set": SNR_SET,
              "summary": summarize(tilt, corner, layer_ids, glob_layers), "stability_across_sets": stab,
              "per_head": {"layers": layer_ids, "tilt": np.where(np.isnan(tilt), None, tilt).tolist(),
                           "corner_tokens": np.where(np.isnan(corner), None, corner).tolist()},
              "cites": cites}
    out = os.path.join(repo, "results", "derived", f"headmap-{args.run_id}")
    if os.path.exists(os.path.join(out, "headmap.json")):
        raise SystemExit(f"{out}/headmap.json exists; derived results are not overwritten.")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "headmap.json"), "w") as fh:
        json.dump(result, fh, indent=1)
    plot(tilt, corner, layer_ids, os.path.join(out, "headmap.png"))
    s = result["summary"]
    print(f"== resolved heads {s['heads_resolved']} of {s['heads_total']}; with a corner {s['heads_with_corner']}")
    print(f"   tilt median local {s['tilt_median_local_layers']:.3f}, global {s['tilt_median_global_layers']:.3f}; "
          f"negative {s['heads_with_negative_tilt']}")
    print(f"   corner percentiles 10/25/50/75/90: {s['corner_percentiles_10_25_50_75_90']}")
    for k, v in stab.items():
        print(f"   stability {k}: tilt r {v['tilt']['r']:.3f} (within layer {v['tilt']['r_within_layer']:.3f}, "
              f"n {v['tilt']['n']}); corner r {v['corner_log']['r']:.3f} (within layer "
              f"{v['corner_log']['r_within_layer']:.3f}, n {v['corner_log']['n']})")
    print(f"   written: {os.path.relpath(out, repo)}/headmap.json and headmap.png")


if __name__ == "__main__":
    main()
