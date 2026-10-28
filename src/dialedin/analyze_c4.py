"""C4 analysis, exactly as sealed: structural levers (window, RoPE base) and text controls, set A.

    PYTHONPATH=src python3 -m dialedin.analyze_c4 --run-id <id>

Per-cell gains and SNR come from analyze_c1v5.cell_quantities, tilts from metrics_c2.tilt, outcomes from
metrics_c4. Reads only Phase B book files for curves; Phase A is checked, never curved.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import yaml

from . import glimmer, metrics, metrics_c2, metrics_c4
from .analyze_c1v5 import cell_quantities

SEALED = "criteria/sealed/c4-glimmer-structure-and-text-v1.yaml"
CONDITIONS = ("BASE", "W1024", "W4096", "R0.5", "R2.0", "SHUF", "CONST")
STRUCT = ("BASE", "W1024", "W4096", "R0.5", "R2.0")
WINDOW = ("BASE", "W1024", "W4096")
ROPE = ("BASE", "R0.5", "R2.0")
TEXT = ("BASE", "SHUF", "CONST")
BOOT = 1000
VALID_SNR = 3.0


def load_books(run_dir: str, spec: dict, seal_hash: str = None, code: dict = None):
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
    for key in ("window_checks", "rope_checks"):
        if not pa.get(key) or not all(c.get("as_expected") for c in pa[key]):
            problems.append(f"Phase A {key} did not all pass")
    if not pa.get("logits_check", {}).get("as_expected"):
        problems.append("Phase A perplexity-path check did not pass")
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


def book_curve(book, grid, cond):
    """Whole-model mean gain per N under one condition, and the whole-model SNR per N."""
    layer_ids = book["record"]["layers_measured"]
    out, snr = np.empty(len(grid)), np.empty(len(grid))
    for j, N in enumerate(grid):
        gain, _, snr[j] = cell_quantities(book["per_condition"][cond][str(N)])
        out[j] = metrics.group_means(gain, layer_ids, book["global_layers"])["whole"]
    return out, snr


def family_tilts(curves: dict, grid, family) -> tuple:
    """Set-level tilts for a family of conditions, each fit over the periods valid (median SNR >= 3 over books)
    under every condition of the family. Returns (tilts, common mask, per-condition masks)."""
    valid = {c: np.median(np.stack([x[1] for x in curves[c]]), axis=0) >= VALID_SNR for c in family}
    common = np.logical_and.reduce([valid[c] for c in family])
    tilts = {c: metrics_c2.tilt(grid, np.mean(np.stack([x[0] for x in curves[c]]), axis=0), common) for c in family}
    return tilts, common, valid


def book_tilts(curves: dict, grid, family, i):
    common = np.logical_and.reduce([curves[c][i][1] >= VALID_SNR for c in family])
    return {c: metrics_c2.tilt(grid, curves[c][i][0], common) for c in family}


def analyze(run_dir: str, spec: dict, seal_hash: str = None, code: dict = None, partial: bool = False,
            c1_run_dir: str = None) -> dict:
    grid = spec["drive"]["periods_N"]
    books = load_books(run_dir, spec, seal_hash, code)
    expected = sorted(int(g) for v in spec["inputs"]["prompt_sets"].values() for g in v)
    found = sorted(b["gutenberg_id"] for b in books)
    complete = found == expected
    if not complete and not partial:
        raise SystemExit(f"Incomplete run: no outcome. Found {found}, expected {expected}.")
    if complete and partial:
        raise SystemExit("The run is complete; analyze it without --partial.")
    curves = {c: [book_curve(b, grid, c) for b in books] for c in CONDITIONS}
    wt, wt_common, valid = family_tilts(curves, grid, WINDOW)
    rt, rt_common, rvalid = family_tilts(curves, grid, ROPE)
    valid.update(rvalid)
    tt, tt_common, tvalid = family_tilts(curves, grid, TEXT)
    valid.update(tvalid)
    ppl = {c: float(np.median([b["perplexity"][c] / b["perplexity"]["BASE"] for b in books])) for c in CONDITIONS}
    per_book = []
    for i, b in enumerate(books):
        per_book.append({"gutenberg_id": b["gutenberg_id"], "window": book_tilts(curves, grid, WINDOW, i),
                         "rope": book_tilts(curves, grid, ROPE, i), "text": book_tilts(curves, grid, TEXT, i)})

    def deltas(fam, lo, hi):
        return [(x[fam][hi] - x[fam][lo]) if None not in (x[fam][hi], x[fam][lo]) else None for x in per_book]

    window = metrics_c4.lever_outcome(wt, deltas("window", "W1024", "W4096"), ppl["W1024"], ppl["W4096"], "window")
    rope = metrics_c4.lever_outcome(rt, deltas("rope", "R0.5", "R2.0"), ppl["R0.5"], ppl["R2.0"], "rope")
    text = metrics_c4.text_outcome(tt["BASE"], tt["SHUF"], tt["CONST"])
    c4 = {"window": window, "rope": rope, "text": text}
    if not complete:
        for v in c4.values():
            v["outcome"] = None
    result = {"run_dir": run_dir, "complete": complete, "grid": grid, "books": len(books),
              "failure_records": sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "*.FAILED.*.json"))),
              "C4": c4, "window_tilts": wt, "rope_tilts": rt, "text_tilts": tt,
              "window_periods": [g for g, v in zip(grid, wt_common) if v],
              "rope_periods": [g for g, v in zip(grid, rt_common) if v],
              "text_periods": [g for g, v in zip(grid, tt_common) if v],
              "valid": {c: v.tolist() for c, v in valid.items()},
              "perplexity_ratio_median": ppl,
              "perplexity_median": {c: float(np.median([b["perplexity"][c] for b in books])) for c in CONDITIONS},
              "curves": {c: np.mean(np.stack([x[0] for x in curves[c]]), axis=0).tolist() for c in CONDITIONS},
              "book_tilts": per_book}
    if not complete:
        result["warning"] = "PARTIAL: diagnostic look only. An incomplete run has no C4 outcome."
    rng = np.random.default_rng([0, 0])
    n = len(books)
    boot = {"window": [], "rope": [], "shuf_over_base": [], "const_over_base": []}
    for _ in range(BOOT):
        pick = rng.integers(0, n, size=n)
        sub = {c: [curves[c][i] for i in pick] for c in CONDITIONS}
        w2, _, _ = family_tilts(sub, grid, WINDOW)
        r2, _, _ = family_tilts(sub, grid, ROPE)
        t2, _, _ = family_tilts(sub, grid, TEXT)
        if None not in (w2["W1024"], w2["W4096"]):
            boot["window"].append(w2["W4096"] - w2["W1024"])
        if None not in (r2["R0.5"], r2["R2.0"]):
            boot["rope"].append(r2["R2.0"] - r2["R0.5"])
        if t2["BASE"] and None not in (t2["SHUF"], t2["CONST"]):
            boot["shuf_over_base"].append(t2["SHUF"] / t2["BASE"])
            boot["const_over_base"].append(t2["CONST"] / t2["BASE"])
    result["bootstrap_95"] = {k: (np.percentile(v, [2.5, 97.5]).tolist() if v else None) for k, v in boot.items()}
    result["diagnostic_BASE_vs_C1v5"] = {"status": "not found", "c1_run": c1_run_dir}
    if c1_run_dir and os.path.isdir(c1_run_dir):
        worst, compared = 0.0, 0
        for b in books:
            p = os.path.join(c1_run_dir, f"book_{b['set']}_{b['gutenberg_id']}.json")
            if not os.path.exists(p):
                continue
            c1 = json.load(open(p))
            for N in grid:
                if str(N) in c1["per_N"]:
                    x = np.asarray(b["per_condition"]["BASE"][str(N)]["target"], dtype=np.float64)
                    y = np.asarray(c1["per_N"][str(N)]["target"], dtype=np.float64)
                    worst = max(worst, float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1e-30))))
                    compared += 1
        result["diagnostic_BASE_vs_C1v5"] = {"status": "compared", "cells_compared": compared,
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    ref = result["curves"]["BASE"][-1]
    for ax, conds, title in ((axes[0], STRUCT, "structure"), (axes[1], TEXT, "text controls")):
        for c in conds:
            ax.plot(grid, np.asarray(result["curves"][c]) / ref, marker="o", label=c)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(title); ax.set_xlabel("Drive period N (tokens)")
        ax.legend()
    axes[0].set_ylabel("Gain relative to BASE at N = 4096")
    c = result["C4"]
    fig.suptitle(f"Muse Glimmer-30B, set A: window {c['window']['outcome']}, RoPE {c['rope']['outcome']}, "
                 f"text {c['text']['outcome']}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--c1-run", default="c1v5-20261027T152040Z", help="C1 v5 run for the BASE reproduction check")
    ap.add_argument("--partial", action="store_true", help="diagnostic look at an incomplete run; never the reported result")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    seal_hash = glimmer.check_seal(repo, SEALED)
    spec = yaml.safe_load(open(os.path.join(repo, SEALED)))
    code = glimmer.check_code(repo, spec)
    from datetime import datetime, timezone
    from .run_c4 import runs_under_seal, write_json
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
        raise SystemExit(f"C4 was stopped by rule in Phase A: {pa['stopped']}. Outcome U (inconclusive). "
                         f"Cite {os.path.relpath(pa_path, repo)}.")
    result = analyze(run_dir, spec, seal_hash, code, partial=args.partial,
                     c1_run_dir=os.path.join(raw_root, args.c1_run))
    result["runs_under_seal"] = runs
    stamp = datetime.now(timezone.utc).strftime("-PARTIAL-%Y%m%dT%H%M%SZ")
    out = os.path.join(repo, "results", "derived", args.run_id + (stamp if args.partial else ""))
    if os.path.exists(os.path.join(out, "c4_summary.json")):
        raise SystemExit(f"{out}/c4_summary.json exists; derived results are not overwritten.")
    os.makedirs(out, exist_ok=True)
    result["cites"] = sorted(os.path.relpath(p, repo) for p in glob.glob(os.path.join(run_dir, "*.json")))
    result["seal_sha256"], result["code_sha256"] = seal_hash, code
    write_json(os.path.join(out, "c4_summary.json"), result, indent=2)
    plot(result, os.path.join(out, "c4_curves.png"))
    c = result["C4"]
    print(f"== C4 outcome: window {c['window']['outcome']}, RoPE {c['rope']['outcome']}, text {c['text']['outcome']}  "
          f"(complete={result['complete']})")
    print(f"   summary: {os.path.relpath(out, repo)}/c4_summary.json")


if __name__ == "__main__":
    main()
