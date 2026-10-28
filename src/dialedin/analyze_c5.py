"""C5 analysis, exactly as sealed: does the trained adapter act as a dial on the tilt? Set A, five scales.

    PYTHONPATH=src python3 -m dialedin.analyze_c5 --run-id <id>

Forked from analyze_c4.py. Per-cell gains and SNR come from analyze_c1v5.cell_quantities, tilts from metrics_c2.tilt,
the outcome from metrics_c5. Reads only Phase B book files for curves; Phase A is checked, never curved. The
recall phase, if it ran, is summarized as exploratory and never enters the outcome.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import yaml

from . import glimmer, metrics, metrics_c2, metrics_c5
from .analyze_c1v5 import cell_quantities

SEALED = "criteria/sealed/c5-glimmer-trained-dial-v1.yaml"
CONDITIONS = metrics_c5.CONDITIONS
BOOT = 1000
VALID_SNR = 3.0


def load_books(run_dir: str, spec: dict, seal_hash: str = None, code: dict = None):
    books = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(run_dir, "book_*_*.json")))
             if re.fullmatch(r"book_[A-Za-z]+_\d+\.json", os.path.basename(p))]
    pa_path = os.path.join(run_dir, "phaseA.json")
    if not os.path.exists(pa_path):
        raise SystemExit(f"Refusing to analyze: no phaseA.json in {run_dir}")
    pa = json.load(open(pa_path))
    tj = json.load(open(os.path.join(run_dir, "training.json")))
    texts = {int(k): v for k, v in spec["inputs"].get("text_sha256", {}).items()}
    problems = []
    if pa["sham_differences"] != 0:
        problems.append("Phase A sham was not clean")
    if not pa.get("adapter_checks", {}).get("as_expected"):
        problems.append("Phase A adapter checks did not pass")
    if not pa.get("logits_check", {}).get("as_expected"):
        problems.append("Phase A perplexity-path check did not pass")
    if pa["record"]["adapter_sha256"] != tj["adapter_sha256"]:
        problems.append("Phase A adapter differs from the trained adapter")
    for b in books:
        tag = f"book {b['gutenberg_id']}"
        if any(v != 0 for v in b["sham_differences"].values()) or b["sham_differences_close"] != 0:
            problems.append(f"{tag}: sham not clean")
        if sorted(b["per_condition"]) != sorted(CONDITIONS):
            problems.append(f"{tag}: conditions {sorted(b['per_condition'])}")
        if b["record"]["adapter_sha256"] != tj["adapter_sha256"]:
            problems.append(f"{tag}: adapter differs from the trained adapter")
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
    for name, rec in (("Phase A", pa["record"]), ("training", tj["record"])):
        if seal_hash is not None and rec["seal_sha256"] != seal_hash:
            problems.append(f"{name} seal hash differs")
        if code is not None and rec["code_sha256"] != code:
            problems.append(f"{name} code hashes differ")
    if problems:
        raise SystemExit("Refusing to analyze:\n  " + "\n  ".join(sorted(set(problems))))
    return books, tj


def book_curve(book, grid, cond):
    """Whole-model mean gain per N under one condition, and the whole-model SNR per N."""
    layer_ids = book["record"]["layers_measured"]
    out, snr = np.empty(len(grid)), np.empty(len(grid))
    for j, N in enumerate(grid):
        gain, _, snr[j] = cell_quantities(book["per_condition"][cond][str(N)])
        out[j] = metrics.group_means(gain, layer_ids, book["global_layers"])["whole"]
    return out, snr


def set_tilts(curves: dict, grid) -> tuple:
    """Set-level tilt per scale, each fit over the periods valid (median SNR >= 3 over books) under every scale."""
    valid = {c: np.median(np.stack([x[1] for x in curves[c]]), axis=0) >= VALID_SNR for c in CONDITIONS}
    common = np.logical_and.reduce([valid[c] for c in CONDITIONS])
    tilts = {c: metrics_c2.tilt(grid, np.mean(np.stack([x[0] for x in curves[c]]), axis=0), common)
             for c in CONDITIONS}
    return tilts, common, valid


def book_tilts(curves: dict, grid, i):
    common = np.logical_and.reduce([curves[c][i][1] >= VALID_SNR for c in CONDITIONS])
    return {c: metrics_c2.tilt(grid, curves[c][i][0], common) for c in CONDITIONS}


def group_curve(book, grid, cond, mask_fn):
    """Mean gain per N over the heads of the measured layers that mask_fn keeps."""
    layer_ids = book["record"]["layers_measured"]
    keep = np.array([mask_fn(l) for l in layer_ids])
    out = np.empty(len(grid))
    for j, N in enumerate(grid):
        gain, _, _ = cell_quantities(book["per_condition"][cond][str(N)])
        out[j] = float(gain[keep].mean())
    return out


def group_diagnostic(books, grid, common, adapted_layers) -> dict:
    """Reported, never gated (pre-registered): the set-level tilt per scale over (a) the measured layers the adapter
    does not touch, (b) local layers, (c) global layers, fit over the same periods as the main tilts."""
    adapted = set(adapted_layers)
    groups = {"unadapted_layers": lambda l: l not in adapted,
              "local_layers": lambda l: l not in set(books[0]["global_layers"]),
              "global_layers": lambda l: l in set(books[0]["global_layers"])}
    out = {}
    for name, fn in groups.items():
        t = {c: metrics_c2.tilt(grid, np.mean(np.stack([group_curve(b, grid, c, fn) for b in books]), axis=0), common)
             for c in CONDITIONS}
        rel = (t["S2"] - t["S0"]) / t["S0"] if None not in (t["S0"], t["S2"]) and t["S0"] else None
        out[name] = {"tilts": t, "relative_change_S2": rel}
    return out


def recall_summary(run_dir: str):
    path = os.path.join(run_dir, "recall.json")
    if not os.path.exists(path):
        return {"status": "not run"}
    rows = json.load(open(path))["rows"]
    out = {"status": "exploratory, not gated"}
    for c in CONDITIONS:
        r = [x for x in rows if x["condition"] == c]
        out[c] = {"trials": len(r), "mean_answer_logprob": float(np.mean([x["answer_logprob"] for x in r])),
                  "all_top1": int(sum(x["all_top1"] for x in r))}
    return out


def analyze(run_dir: str, spec: dict, seal_hash: str = None, code: dict = None, partial: bool = False,
            c4_run_dir: str = None) -> dict:
    grid = spec["drive"]["periods_N"]
    books, tj = load_books(run_dir, spec, seal_hash, code)
    expected = sorted(int(g) for v in spec["inputs"]["prompt_sets"].values() for g in v)
    found = sorted(b["gutenberg_id"] for b in books)
    complete = found == expected
    if not complete and not partial:
        raise SystemExit(f"Incomplete run: no outcome. Found {found}, expected {expected}.")
    if complete and partial:
        raise SystemExit("The run is complete; analyze it without --partial.")
    curves = {c: [book_curve(b, grid, c) for b in books] for c in CONDITIONS}
    tilts, common, valid = set_tilts(curves, grid)
    ppl = {c: float(np.median([b["perplexity"][c] / b["perplexity"]["S0"] for b in books])) for c in CONDITIONS}
    per_book = [{"gutenberg_id": b["gutenberg_id"], "tilts": book_tilts(curves, grid, i)} for i, b in enumerate(books)]
    deltas = [(x["tilts"]["S2"] - x["tilts"]["S0"]) if None not in (x["tilts"]["S2"], x["tilts"]["S0"]) else None
              for x in per_book]
    c5 = metrics_c5.dial_outcome(tilts, deltas, ppl)
    if not complete:
        c5["outcome"] = None
    result = {"run_dir": run_dir, "complete": complete, "grid": grid, "books": len(books),
              "failure_records": sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "*.FAILED.*.json"))),
              "C5": c5, "tilts": tilts, "periods_used": [g for g, v in zip(grid, common) if v],
              "valid": {c: v.tolist() for c, v in valid.items()},
              "perplexity_ratio_median": ppl,
              "perplexity_median": {c: float(np.median([b["perplexity"][c] for b in books])) for c in CONDITIONS},
              "curves": {c: np.mean(np.stack([x[0] for x in curves[c]]), axis=0).tolist() for c in CONDITIONS},
              "book_tilts": per_book,
              "training": {"adapter_sha256": tj["adapter_sha256"], "modules": len(tj["modules"]),
                           "params": tj["params"], "steps": len(tj["steps"]), "rollbacks": tj["rollbacks"],
                           "kl_last20_mean": float(np.mean([s["kl"] for s in tj["steps"][-20:]]))},
              "recall": recall_summary(run_dir),
              "diagnostic_groups": group_diagnostic(books, grid, common, spec["training"]["layers"])}
    if not complete:
        result["warning"] = "PARTIAL: diagnostic look only. An incomplete run has no C5 outcome."
    rng = np.random.default_rng([0, 0])
    n = len(books)
    boot = {"relative_change_S2": [], "relative_change_S1": [], "spearman": []}
    for _ in range(BOOT):
        pick = rng.integers(0, n, size=n)
        sub = {c: [curves[c][i] for i in pick] for c in CONDITIONS}
        t2, _, _ = set_tilts(sub, grid)
        if None not in t2.values() and t2["S0"]:
            boot["relative_change_S2"].append((t2["S2"] - t2["S0"]) / t2["S0"])
            boot["relative_change_S1"].append((t2["S1"] - t2["S0"]) / t2["S0"])
            boot["spearman"].append(metrics.spearman(list(metrics_c5.SCALES), [t2[c] for c in CONDITIONS]))
    result["bootstrap_95"] = {k: (np.percentile(v, [2.5, 97.5]).tolist() if v else None) for k, v in boot.items()}
    result["diagnostic_S0_vs_C4_BASE"] = {"status": "not found", "c4_run": c4_run_dir}
    if c4_run_dir and os.path.isdir(c4_run_dir):
        worst, compared = 0.0, 0
        for b in books:
            p = os.path.join(c4_run_dir, f"book_{b['set']}_{b['gutenberg_id']}.json")
            if not os.path.exists(p):
                continue
            c4 = json.load(open(p))
            for N in grid:
                x = np.asarray(b["per_condition"]["S0"][str(N)]["target"], dtype=np.float64)
                y = np.asarray(c4["per_condition"]["BASE"][str(N)]["target"], dtype=np.float64)
                worst = max(worst, float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1e-30))))
                compared += 1
        result["diagnostic_S0_vs_C4_BASE"] = {"status": "compared", "cells_compared": compared,
                                              "max_relative_difference": worst, "c4_run": os.path.basename(c4_run_dir)}
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
    ref = result["curves"]["S0"][-1]
    for c in CONDITIONS:
        ax.plot(grid, np.asarray(result["curves"][c]) / ref, marker="o", label=f"scale {c[1:]}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Drive period N (tokens)")
    ax.set_ylabel("Gain relative to scale 0 at N = 4096"); ax.legend()
    ax.set_title(f"Muse Glimmer-30B, set A, trained dial: outcome {result['C5']['outcome']}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--c4-run", default="c4v1-20261028T065820Z", help="C4 run for the S0 reproduction check")
    ap.add_argument("--partial", action="store_true", help="diagnostic look at an incomplete run; never the reported result")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    seal_hash = glimmer.check_seal(repo, SEALED)
    spec = yaml.safe_load(open(os.path.join(repo, SEALED)))
    code = glimmer.check_code(repo, spec)
    from datetime import datetime, timezone
    from .run_c5 import runs_under_seal, write_json
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
        raise SystemExit(f"C5 was stopped by rule in Phase A: {pa['stopped']}. Outcome U (inconclusive). "
                         f"Cite {os.path.relpath(pa_path, repo)}.")
    result = analyze(run_dir, spec, seal_hash, code, partial=args.partial,
                     c4_run_dir=os.path.join(raw_root, args.c4_run))
    result["runs_under_seal"] = runs
    stamp = datetime.now(timezone.utc).strftime("-PARTIAL-%Y%m%dT%H%M%SZ")
    out = os.path.join(repo, "results", "derived", args.run_id + (stamp if args.partial else ""))
    if os.path.exists(os.path.join(out, "c5_summary.json")):
        raise SystemExit(f"{out}/c5_summary.json exists; derived results are not overwritten.")
    os.makedirs(out, exist_ok=True)
    result["cites"] = sorted(os.path.relpath(p, repo) for p in glob.glob(os.path.join(run_dir, "*.json")))
    result["seal_sha256"], result["code_sha256"] = seal_hash, code
    write_json(os.path.join(out, "c5_summary.json"), result, indent=2)
    plot(result, os.path.join(out, "c5_curves.png"))
    c = result["C5"]
    rel = c.get("relative_change")
    print(f"== C5 outcome: {c['outcome']}  (complete={result['complete']}); tilt S0 -> S2 change "
          f"{100 * rel:+.1f}%" if rel is not None else f"== C5 outcome: {c['outcome']}")
    print(f"   summary: {os.path.relpath(out, repo)}/c5_summary.json")


if __name__ == "__main__":
    main()
