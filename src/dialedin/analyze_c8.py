"""C8 analysis, exactly as sealed: does layer 2's v_proj alone carry a majority of the dial's tilt change? Set C, six
conditions.

    PYTHONPATH=src python3 -m dialedin.analyze_c8 --run-id <id>

Forked from analyze_c7.py. Per-cell gains and SNR come from analyze_c1v5.cell_quantities, tilts from metrics_c2.tilt,
the outcome from metrics_c8. Reads only Phase B book files for curves; Phase A is checked, never curved.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import yaml

from . import glimmer, metrics, metrics_c2, metrics_c8
from .analyze_c1v5 import cell_quantities

SEALED = "criteria/sealed/c8-glimmer-one-matrix-dial-v1.yaml"
CONDITIONS = metrics_c8.CONDITIONS
BOOT = 1000
VALID_SNR = 3.0


def load_books(run_dir: str, spec: dict, seal_hash: str = None, code: dict = None):
    books = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(run_dir, "book_*_*.json")))
             if re.fullmatch(r"book_[A-Za-z]+_\d+\.json", os.path.basename(p))]
    pa_path = os.path.join(run_dir, "phaseA.json")
    if not os.path.exists(pa_path):
        raise SystemExit(f"Refusing to analyze: no phaseA.json in {run_dir}")
    pa = json.load(open(pa_path))
    want_sha = spec["adapter"]["sha256"]
    texts = {int(k): v for k, v in spec["inputs"].get("text_sha256", {}).items()}
    problems = []
    if pa["sham_differences"] != 0:
        problems.append("Phase A sham was not clean")
    if not pa.get("adapter_checks", {}).get("as_expected"):
        problems.append("Phase A adapter and condition checks did not pass")
    if not pa.get("logits_check", {}).get("as_expected"):
        problems.append("Phase A perplexity-path check did not pass")
    if pa["record"]["adapter_sha256"] != want_sha:
        problems.append("Phase A adapter differs from the sealed adapter")
    if pa.get("stopped"):
        problems.append(f"Phase A stopped C8 by rule ({pa['stopped']}): outcome U, no curves")
    for c in CONDITIONS:
        sealed = spec["conditions"][c]["modules"]
        if sealed != "all" and pa["record"]["conditions"][c]["modules_on"] != list(sealed):
            problems.append(f"Phase A {c} modules differ from the sealed list")
    for b in books:
        tag = f"book {b['gutenberg_id']}"
        if any(v != 0 for v in b["sham_differences"].values()) or b["sham_differences_close"] != 0:
            problems.append(f"{tag}: sham not clean")
        if sorted(b["per_condition"]) != sorted(CONDITIONS):
            problems.append(f"{tag}: conditions {sorted(b['per_condition'])}")
        if b["record"]["adapter_sha256"] != want_sha:
            problems.append(f"{tag}: adapter differs from the sealed adapter")
        if b["record"]["conditions"] != pa["record"]["conditions"]:
            problems.append(f"{tag}: condition modules differ from Phase A")
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
    return books, pa


def curve(book, grid, cond, keep_layer=None):
    """Mean gain per N over every measured head (or over the heads of layers keep_layer accepts), and the
    whole-model SNR per N."""
    layer_ids = book["record"]["layers_measured"]
    keep = None if keep_layer is None else np.array([keep_layer(l) for l in layer_ids])
    out, snr = np.empty(len(grid)), np.empty(len(grid))
    for j, N in enumerate(grid):
        gain, _, snr[j] = cell_quantities(book["per_condition"][cond][str(N)])
        out[j] = (metrics.group_means(gain, layer_ids, book["global_layers"])["whole"] if keep is None
                  else float(gain[keep].mean()))
    return out, snr


def set_tilts(curves: dict, grid) -> tuple:
    """Set-level tilt per condition, each fit over the periods valid (median SNR >= 3 over books) under every
    condition."""
    valid = {c: np.median(np.stack([x[1] for x in curves[c]]), axis=0) >= VALID_SNR for c in CONDITIONS}
    common = np.logical_and.reduce([valid[c] for c in CONDITIONS])
    tilts = {c: metrics_c2.tilt(grid, np.mean(np.stack([x[0] for x in curves[c]]), axis=0), common)
             for c in CONDITIONS}
    return tilts, common, valid


def book_tilts(curves: dict, grid, i):
    common = np.logical_and.reduce([curves[c][i][1] >= VALID_SNR for c in CONDITIONS])
    return {c: metrics_c2.tilt(grid, curves[c][i][0], common) for c in CONDITIONS}


def group_diagnostic(books, grid, common) -> dict:
    """Reported, never gated (pre-registered): set-level tilts per condition and shares over (a) layer 2's heads,
    (b) layers 3 to 51, (c) layers 6 to 51 (no adapter), fit over the main periods."""
    groups = {"layer2": lambda l: l == 2, "layers3_51": lambda l: l >= 3, "layers6_51": lambda l: l >= 6}
    out = {}
    for name, fn in groups.items():
        t = {c: metrics_c2.tilt(grid, np.mean(np.stack([curve(b, grid, c, fn)[0] for b in books]), axis=0), common)
             for c in CONDITIONS}
        rel = {c: ((t[c] - t["BASE"]) / t["BASE"] if None not in (t[c], t["BASE"]) and t["BASE"] else None)
               for c in CONDITIONS if c != "BASE"}
        out[name] = {"tilts": t, "relative_change": rel, "shares": metrics_c8.shares(t)}
    return out


def reproduction(books, grid, c1_run_dir) -> dict:
    """BASE runs C1 v5's computation on the same set B tokens. Reported, expected 0, not a gate: the maximum relative
    difference of per-head target magnitudes, of the input component per period, and of the amplitude reference
    (C1 v5's median_norm) from C1 v5's raw files. Status "complete" only if every book and period was compared."""
    rel = lambda x, y: float(np.max(np.abs(np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64))
                                    / np.maximum(np.abs(np.asarray(y, dtype=np.float64)), 1e-30)))
    if not c1_run_dir or not os.path.isdir(c1_run_dir):
        return {"status": "not found", "c1_run": c1_run_dir}
    worst = {"target": 0.0, "input_mag": 0.0, "amplitude_reference": 0.0}
    compared, missing, cells = [], [], 0
    for b in books:
        p = os.path.join(c1_run_dir, f"book_{b['set']}_{b['gutenberg_id']}.json")
        if not os.path.exists(p):
            missing.append(b["gutenberg_id"])
            continue
        c1 = json.load(open(p))
        compared.append(b["gutenberg_id"])
        worst["amplitude_reference"] = max(worst["amplitude_reference"],
                                           rel([b["amplitude_reference"]["BASE"]], [c1["median_norm"]]))
        for N in grid:
            x, y = b["per_condition"]["BASE"][str(N)], c1["per_N"][str(N)]
            worst["target"] = max(worst["target"], rel(x["target"], y["target"]))
            worst["input_mag"] = max(worst["input_mag"], rel([x["input_mag"]], [y["input_mag"]]))
            cells += 1
    full = not missing and cells == len(books) * len(grid)
    return {"status": "complete" if full else "incomplete", "books_compared": compared, "books_missing": missing,
            "cells_compared": cells, "max_relative_difference": worst["target"],
            "max_relative_difference_input_mag": worst["input_mag"],
            "max_relative_difference_amplitude_reference": worst["amplitude_reference"],
            "c1_run": os.path.basename(c1_run_dir)}


def analyze(run_dir: str, spec: dict, seal_hash: str = None, code: dict = None, partial: bool = False,
            c1_run_dir: str = None) -> dict:
    grid = spec["drive"]["periods_N"]
    books, pa = load_books(run_dir, spec, seal_hash, code)
    expected = sorted(int(g) for v in spec["inputs"]["prompt_sets"].values() for g in v)
    found = sorted(b["gutenberg_id"] for b in books)
    complete = found == expected
    if not complete and not partial:
        raise SystemExit(f"Incomplete run: no outcome. Found {found}, expected {expected}.")
    if complete and partial:
        raise SystemExit("The run is complete; analyze it without --partial.")
    curves = {c: [curve(b, grid, c) for b in books] for c in CONDITIONS}
    tilts, common, valid = set_tilts(curves, grid)
    per_book = []
    for i, b in enumerate(books):
        t = book_tilts(curves, grid, i)
        per_book.append({"gutenberg_id": b["gutenberg_id"], "tilts": t, "shares": metrics_c8.shares(t),
                         "agrees": metrics_c8.book_agrees(t)})
    c8 = metrics_c8.one_matrix_outcome(tilts, [x["tilts"] for x in per_book])
    if not complete:                       # a partial look has no outcome; its numbers are kept apart
        c8 = {"outcome": None, "partial_diagnostic": c8}
    ppl = {c: float(np.median([b["perplexity"][c] / b["perplexity"]["BASE"] for b in books])) for c in CONDITIONS}
    result = {"run_dir": run_dir, "complete": complete, "grid": grid, "books": len(books),
              "failure_records": sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "*.FAILED.*.json"))),
              "C8": c8, "tilts": tilts, "periods_used": [g for g, v in zip(grid, common) if v],
              "valid": {c: v.tolist() for c, v in valid.items()},
              "curves": {c: np.mean(np.stack([x[0] for x in curves[c]]), axis=0).tolist() for c in CONDITIONS},
              "book_tilts": per_book, "perplexity_ratio_median": ppl,
              "modules_on": {c: len(v["modules_on"]) for c, v in pa["record"]["conditions"].items()},
              "diagnostic_groups": group_diagnostic(books, grid, common),
              "reproduction_BASE_vs_C1v5": reproduction(books, grid, c1_run_dir)}
    vb = [x["shares"]["V2"] for x in per_book if x["shares"]["V2"] is not None]
    result["secondary_mean_book_share_V2"] = {"value": float(np.mean(vb)) if vb else None, "books": len(vb),
                                              "note": "Muse Wonder's proposed metric; reported, not gated"}
    if not complete:
        result["warning"] = "PARTIAL: diagnostic look only. An incomplete run has no C8 outcome."
    rng = np.random.default_rng([0, 0])
    n = len(books)
    boot = {"share_V2": [], "share_V4": [], "share_QKV2": [], "share_QK2": [], "full_change": []}
    for _ in range(BOOT):
        pick = rng.integers(0, n, size=n)
        t2, _, _ = set_tilts({c: [curves[c][i] for i in pick] for c in CONDITIONS}, grid)
        if None in t2.values() or t2["BASE"] <= 0:
            continue
        fc = (t2["FULL"] - t2["BASE"]) / t2["BASE"]
        boot["full_change"].append(fc)
        if fc <= -metrics_c8.MIN_FULL_CHANGE:      # shares only where FULL flattens enough to split, as the outcome
            sh = metrics_c8.shares(t2)
            for c in ("V2", "V4", "QKV2", "QK2"):
                boot[f"share_{c}"].append(sh[c])
    result["bootstrap_95"] = {k: (np.percentile(v, [2.5, 97.5]).tolist() if v else None) for k, v in boot.items()}
    result["bootstrap_resamples_used"] = {k: len(v) for k, v in boot.items()}
    result["bootstrap_resamples_drawn"] = BOOT
    return result


def plot(result: dict, path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    grid = result["grid"]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ref = result["curves"]["BASE"][-1]
    labels = {"BASE": "dial 0", "FULL": "dial 2, all 32 matrices", "QKV2": "dial 2, layer 2 q, k, v",
              "V2": "dial 2, layer 2 v only", "QK2": "dial 2, layer 2 q, k", "V4": "dial 2, layer 4 v (control)"}
    for c in CONDITIONS:
        ax.plot(grid, np.asarray(result["curves"][c]) / ref, marker="o", label=labels[c])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Drive period N (tokens)")
    ax.set_ylabel("Gain relative to dial 0 at N = 4096"); ax.legend()
    ax.set_title(f"Muse Glimmer-30B, set C: one-matrix dial, outcome {result['C8']['outcome']}")
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
    from .run_c8 import runs_under_seal, write_json
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
        raise SystemExit(f"C8 was stopped by rule in Phase A: {json.load(open(pa_path))['stopped']}. Outcome U. "
                         f"Cite {os.path.relpath(pa_path, repo)}.")
    result = analyze(run_dir, spec, seal_hash, code, partial=args.partial,
                     c1_run_dir=os.path.join(raw_root, args.c1_run))
    result["runs_under_seal"] = runs
    stamp = datetime.now(timezone.utc).strftime("-PARTIAL-%Y%m%dT%H%M%SZ")
    out = os.path.join(repo, "results", "derived", args.run_id + (stamp if args.partial else ""))
    if os.path.exists(os.path.join(out, "c8_summary.json")):
        raise SystemExit(f"{out}/c8_summary.json exists; derived results are not overwritten.")
    os.makedirs(out, exist_ok=True)
    result["cites"] = sorted(os.path.relpath(p, repo) for p in glob.glob(os.path.join(run_dir, "*.json")))
    result["seal_sha256"], result["code_sha256"] = seal_hash, code
    write_json(os.path.join(out, "c8_summary.json"), result, indent=2)
    plot(result, os.path.join(out, "c8_curves.png"))
    c, b = result["C8"], result["bootstrap_95"]
    pct = lambda v: "n/a" if v is None else f"{100 * v:.0f}%"
    ci = lambda k: f"95% {pct(b[k][0])} to {pct(b[k][1])}" if b.get(k) else "95% n/a"
    print(f"== C8 outcome: {c['outcome']} (complete={result['complete']})" + (f"; {c['reason']}" if c.get("reason") else ""))
    if "share_V2" in c:
        print(f"   FULL tilt change {100 * c['full_change']:+.1f}%; share V2 {pct(c['share_V2'])} ({ci('share_V2')}); "
              f"control V4 {pct(c['share_V4'])} ({ci('share_V4')}); books agreeing {c['books_agree_V2']}/5")
        print(f"   QKV2 {pct(c['share_QKV2'])} ({ci('share_QKV2')}); QK2 {pct(c['share_QK2'])}; V2 as a share of QKV2 "
              f"{pct(c['share_V2_of_QKV2'])}; (V2 + QK2) / QKV2 {pct(c['additivity_V2_plus_QK2_over_QKV2'])}")
    rp = result["reproduction_BASE_vs_C1v5"]
    print(f"   BASE vs C1 v5 ({rp['status']}): max relative difference {rp.get('max_relative_difference')}; "
          f"perplexity ratios {', '.join(f'{k} x{v:.4f}' for k, v in result['perplexity_ratio_median'].items())}")
    print(f"   summary: {os.path.relpath(out, repo)}/c8_summary.json")


if __name__ == "__main__":
    main()
