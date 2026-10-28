"""C3 analysis, exactly as sealed: do the 32 strongest slow-response heads work as a dial for the tilt?

    PYTHONPATH=src python3 -m dialedin.analyze_c3 --run-id <id>

Forked from analyze_c2.py. Reads only Phase B book files for curves; Phase A is checked, never curved.
Per-cell gains and SNR come from analyze_c1v5.cell_quantities, tilts from metrics_c2.tilt, outcomes
from metrics_c3. Only layers downstream of every scaled head are measured.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import yaml

from . import glimmer, metrics, metrics_c2, metrics_c3
from .analyze_c1v5 import cell_quantities

SEALED = "criteria/sealed/c3-glimmer-head-dial-v1.yaml"
CONDITIONS = ("S1.0", "S0.0", "S0.5", "S1.5", "S2.0", "R0.0", "R2.0")
BOOT = 1000
VALID_SNR = 3.0


def load_books(run_dir: str, spec: dict, seal_hash: str = None, code: dict = None):
    """Refuses books from another seal, code, a, platform or text, or with any dirty sham. Needs phaseA.json."""
    books = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(run_dir, "book_*_*.json")))
             if re.fullmatch(r"book_[A-Za-z]+_\d+\.json", os.path.basename(p))]
    pa_path = os.path.join(run_dir, "phaseA.json")
    if not os.path.exists(pa_path):
        raise SystemExit(f"Refusing to analyze: no phaseA.json in {run_dir}")
    pa = json.load(open(pa_path))
    texts = {int(k): v for k, v in spec["inputs"].get("text_sha256", {}).items()}
    problems = []
    if pa["sham_differences"] != 0:
        problems.append("Phase A sham was not clean")
    for b in books:
        tag = f"book {b['gutenberg_id']}"
        if any(v != 0 for v in b["sham_differences"].values()) or b["sham_differences_close"] != 0:
            problems.append(f"{tag}: sham not clean")
        if sorted(b["per_condition"]) != sorted(CONDITIONS):
            problems.append(f"{tag}: conditions {sorted(b['per_condition'])}")
        if seal_hash is not None and b["record"]["seal_sha256"] != seal_hash:
            problems.append(f"{tag}: seal hash differs")
        if code is not None and b["record"]["code_sha256"] != code:
            problems.append(f"{tag}: code hashes differ")
        want = texts.get(int(b["gutenberg_id"]))
        if want is not None and b["record"].get("text_sha256") != want:
            problems.append(f"{tag}: text hash differs from the sealed value")
        if b["a"] != pa["chosen_a"]:
            problems.append(f"{tag}: a={b['a']} but Phase A allowed {pa['chosen_a']}")
        if b["record"]["platform"] != pa["record"]["platform"]:
            problems.append(f"{tag}: platform differs from Phase A")
    if seal_hash is not None and pa["record"]["seal_sha256"] != seal_hash:
        problems.append("Phase A seal hash differs")
    if code is not None and pa["record"]["code_sha256"] != code:
        problems.append("Phase A code hashes differ")
    if problems:
        raise SystemExit("Refusing to analyze:\n  " + "\n  ".join(sorted(set(problems))))
    return books


def book_curves(book, grid, cond):
    """Whole, local and global mean gain per N under one condition, and the whole-model SNR per N."""
    layer_ids = book["record"]["layers_measured"]
    out = {g: np.empty(len(grid)) for g in ("whole", "local", "global")}
    snr = np.empty(len(grid))
    for j, N in enumerate(grid):
        gain, _, snr[j] = cell_quantities(book["per_condition"][cond][str(N)])
        gm = metrics.group_means(gain, layer_ids, book["global_layers"])
        for g in out:
            out[g][j] = gm[g]
    return out, snr


def set_tilts(curves_by_cond, grid, group="whole"):
    """curves_by_cond: {cond: [book_curves results]}. A period is valid for a condition if the median over
    books of SNR >= 3; every condition's tilt is fit over the periods valid under all conditions, so a
    lost period cannot shift one condition's slope through curvature. Returns tilts, per-condition masks."""
    valid = {cond: np.median(np.stack([c[1] for c in curves]), axis=0) >= VALID_SNR
             for cond, curves in curves_by_cond.items()}
    common = np.logical_and.reduce(list(valid.values()))
    out = {cond: metrics_c2.tilt(grid, np.mean(np.stack([c[0][group] for c in curves]), axis=0), common)
           for cond, curves in curves_by_cond.items()}
    return out, valid


def analyze(run_dir: str, spec: dict, seal_hash: str = None, code: dict = None, partial: bool = False,
            c1_run_dir: str = None) -> dict:
    grid = spec["drive"]["periods_N"]
    books = load_books(run_dir, spec, seal_hash, code)
    failures = sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "*.FAILED.*.json")))
    by_set = {}
    for b in books:
        by_set.setdefault(b["set"], []).append(b)
    expected = {s: sorted(int(g) for g in v) for s, v in spec["inputs"]["prompt_sets"].items()}
    found = {s: sorted(b["gutenberg_id"] for b in by_set.get(s, [])) for s in expected}
    complete = found == expected and set(by_set) == set(expected)
    if not complete and not partial:
        raise SystemExit(f"Incomplete run: no outcome. Found {found}, expected {expected}.")
    if complete and partial:
        raise SystemExit("The run is complete; analyze it without --partial.")
    sets = sorted(by_set)
    curves = {s: {c: [book_curves(b, grid, c) for b in by_set[s]] for c in CONDITIONS} for s in sets}
    tilts, valid = {}, {}
    for s in sets:
        tilts[s], valid[s] = set_tilts(curves[s], grid)
    ratios = {(b["record"]["knob"]["control_size_ratio"], b["record"]["knob"]["layer_gap"]) for b in books}
    if len(ratios) != 1:
        raise SystemExit(f"books disagree on the control size ratio or layer gap: {ratios}")
    ratio, gap = ratios.pop()
    c2 = metrics_c3.classify_c3(tilts, ratio, gap)
    if not complete:
        c2["outcome"] = None
    result = {"run_dir": run_dir, "complete": complete, "grid": grid, "books": len(books),
              "failure_records": failures, "C3": c2,
              "valid": {s: {c: v.tolist() for c, v in valid[s].items()} for s in sets},
              "curves": {s: {c: np.mean(np.stack([x[0]["whole"] for x in curves[s][c]]), axis=0).tolist()
                             for c in CONDITIONS} for s in sets}}
    if not complete:
        result["warning"] = "PARTIAL: diagnostic look only. An incomplete run has no C3 outcome."

    # Bootstrap within each set: resample books, recompute the tilt changes.
    boot = {}
    for si, s in enumerate(sets):
        rng = np.random.default_rng([0, si])
        n = len(by_set[s])
        dg, dl = [], []
        for _ in range(BOOT):
            pick = rng.integers(0, n, size=n)
            t, _ = set_tilts({c: [curves[s][c][i] for i in pick] for c in CONDITIONS}, grid)
            if all(v is not None for v in t.values()):
                dg.append(t["S2.0"] - t["S0.0"])
                dl.append(t["R2.0"] - t["R0.0"])
        boot[s] = {"delta_selected_95": np.percentile(dg, [2.5, 97.5]).tolist() if dg else None,
                   "delta_random_95": np.percentile(dl, [2.5, 97.5]).tolist() if dl else None,
                   "resamples_defined": len(dg)}
    result["bootstrap"] = boot

    # Descriptive: per-book tilts and signs, and tilts of the local-head and global-head groups.
    per_book, per_book_labeled = [], []
    for s in sets:
        for i, b in enumerate(by_set[s]):
            common = np.logical_and.reduce([curves[s][c][i][1] >= VALID_SNR for c in CONDITIONS])
            t = {c: metrics_c2.tilt(grid, curves[s][c][i][0]["whole"], common) for c in CONDITIONS}
            per_book.append(t)
            per_book_labeled.append({"set": s, "gutenberg_id": b["gutenberg_id"], "tilts": t})
    result["book_tilts"] = per_book_labeled
    sn = [b.get("slice_norms") for b in books if b.get("slice_norms")]
    if sn:
        result["diagnostic_slice_norm_control_over_selected_median"] = float(np.median(
            [x["control_sum"] / x["selected_sum"] for x in sn]))
    result["book_signs"] = metrics_c3.book_signs(per_book)
    result["group_tilts"] = {s: {g: set_tilts(curves[s], grid, g)[0] for g in ("local", "global")} for s in sets}

    # Descriptive: S1.0 should reproduce C1 v5 at the shared periods and layers (same code path, same inputs).
    result["diagnostic_S1_vs_C1v5"] = {"status": "not found", "c1_run": c1_run_dir}
    if c1_run_dir and os.path.isdir(c1_run_dir):
        worst = 0.0
        compared = 0
        for b in books:
            p = os.path.join(c1_run_dir, f"book_{b['set']}_{b['gutenberg_id']}.json")
            if not os.path.exists(p):
                continue
            c1 = json.load(open(p))
            for N in grid:
                if str(N) in c1["per_N"]:
                    x = np.asarray(b["per_condition"]["S1.0"][str(N)]["target"], dtype=np.float64)
                    first = b["record"]["layers_measured"][0] - c1["record"]["layers_measured"][0]
                    y = np.asarray(c1["per_N"][str(N)]["target"], dtype=np.float64)[first:first + len(x)]
                    worst = max(worst, float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1e-30))))
                    compared += 1
        result["diagnostic_S1_vs_C1v5"] = {"status": "compared", "cells_compared": compared,
                                           "max_relative_difference": worst, "c1_run": os.path.basename(c1_run_dir)}
    return result


def plot(result: dict, path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    grid = result["grid"]
    fig, axes = plt.subplots(1, len(result["curves"]), figsize=(13, 4.2), sharey=True)
    for ax, (s, cc) in zip(np.atleast_1d(axes), result["curves"].items()):
        for c in ("S0.0", "S0.5", "S1.0", "S1.5", "S2.0", "R0.0", "R2.0"):
            ax.plot(grid, np.asarray(cc[c]) / cc["S1.0"][-1], marker="o", ls="-" if c[0] == "S" else "--", label=c)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"set {s}"); ax.set_xlabel("Drive period N (tokens)")
    np.atleast_1d(axes)[0].set_ylabel("Gain relative to S1.0 at N = 4096")
    np.atleast_1d(axes)[0].legend()
    fig.suptitle(f"Muse Glimmer-30B: head dial vs tilt (downstream layers), outcome {result['C3']['outcome']}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--c1-run", default="c1v5-20261027T152040Z", help="C1 v5 run for the S1.0 reproduction check")
    ap.add_argument("--partial", action="store_true", help="diagnostic look at an incomplete run; never the reported result")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    seal_hash = glimmer.check_seal(repo, SEALED)
    spec = yaml.safe_load(open(os.path.join(repo, SEALED)))
    code = glimmer.check_code(repo, spec)
    from datetime import datetime, timezone
    from .run_c3 import runs_under_seal, write_json
    raw_root = os.path.join(repo, "results", "raw")
    runs = runs_under_seal(raw_root, seal_hash, spec)
    if args.run_id not in [r["run_id"] for r in runs]:
        raise SystemExit(f"{args.run_id} has no STARTED.json under this seal.")
    complete = [r["run_id"] for r in runs if r["state"] == "complete"]
    if len(complete) > 1:
        raise SystemExit(f"More than one complete run under this seal: {complete}. Refusing to pick one.")
    run_dir = os.path.join(raw_root, args.run_id)
    pa_path = os.path.join(run_dir, "phaseA.json")
    if os.path.exists(pa_path) and json.load(open(pa_path)).get("stopped"):
        pa = json.load(open(pa_path))
        raise SystemExit(f"C3 was stopped by rule in Phase A: {pa['stopped']}. Outcome U (inconclusive). "
                         f"Cite {os.path.relpath(pa_path, repo)}.")
    result = analyze(run_dir, spec, seal_hash, code, partial=args.partial,
                     c1_run_dir=os.path.join(raw_root, args.c1_run))
    result["runs_under_seal"] = runs
    stamp = datetime.now(timezone.utc).strftime("-PARTIAL-%Y%m%dT%H%M%SZ")
    out = os.path.join(repo, "results", "derived", args.run_id + (stamp if args.partial else ""))
    if os.path.exists(os.path.join(out, "c3_summary.json")):
        raise SystemExit(f"{out}/c3_summary.json exists; derived results are not overwritten.")
    os.makedirs(out, exist_ok=True)
    result["cites"] = sorted(os.path.relpath(p, repo) for p in glob.glob(os.path.join(run_dir, "*.json")))
    result["seal_sha256"], result["code_sha256"] = seal_hash, code
    write_json(os.path.join(out, "c3_summary.json"), result, indent=2)
    plot(result, os.path.join(out, "c3_curves.png"))
    print(f"== C3 outcome: {result['C3']['outcome']}  direction {result['C3'].get('direction')}  "
          f"(complete={result['complete']})")
    print(f"   summary: {os.path.relpath(out, repo)}/c3_summary.json")


if __name__ == "__main__":
    main()
