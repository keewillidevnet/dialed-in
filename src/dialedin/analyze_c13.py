"""C13 analysis, exactly as sealed: does C11's lean-path result hold in Llama-3.1-8B-Instruct? Set A, tokens 0 to
32,767 (C11's segment), three conditions.

    PYTHONPATH=src python3 -m dialedin.analyze_c13 --run-id <id>

Forked from analyze_c11.py. Per-cell gains and SNR come from analyze_c1v5.cell_quantities, tilts from metrics_c2.tilt,
the outcome from metrics_c11 (C11's bands, unchanged). Reads only Phase B book files for curves; Phase A is checked, never curved.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import yaml

from . import glimmer, metrics, metrics_c2, metrics_c11
from .analyze_c1v5 import cell_quantities

SEALED = "criteria/sealed/c13-llama-lean-path-v1.yaml"
CONDITIONS = metrics_c11.CONDITIONS
GATED = metrics_c11.GATED
FROZEN = {"BASE": [], "VONLY": ["q_proj", "k_proj"], "PONLY": ["v_proj"]}
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
    if not isinstance(pa["sham_differences"], dict) or any(v != 0 for v in pa["sham_differences"].values()):
        problems.append("Phase A shams were not clean")
    for key, what in (("identity_check", "attention input identity"), ("logits_check", "perplexity path"),
                      ("effect_check", "freeze effect")):
        if not pa.get(key, {}).get("as_expected"):
            problems.append(f"Phase A {what} check did not pass")
    if pa.get("stopped"):
        problems.append(f"Phase A stopped C13 by rule ({pa['stopped']}): outcome U, no curves")
    if {c: pa["record"]["conditions"][c]["frozen"] for c in CONDITIONS} != FROZEN:
        problems.append("Phase A frozen modules differ from the sealed lists")
    sealed = {c: list(spec["conditions"][c]["frozen"]) for c in CONDITIONS}
    if sealed != FROZEN:
        problems.append("the sealed frozen modules differ from this analysis")
    for b in books:
        tag = f"book {b['gutenberg_id']}"
        if any(v != 0 for v in b["sham_differences"].values()) or b["sham_differences_close"] != 0:
            problems.append(f"{tag}: sham not clean")
        if sorted(b["sham_differences"]) != sorted(CONDITIONS) or sorted(b["per_condition"]) != sorted(CONDITIONS):
            problems.append(f"{tag}: conditions {sorted(b['per_condition'])}")
        if b["record"]["conditions"] != pa["record"]["conditions"]:
            problems.append(f"{tag}: frozen modules differ from Phase A")
        if b["record"]["freeze"] != pa["record"]["freeze"]:
            problems.append(f"{tag}: freeze setup differs from Phase A")
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


GROUPS = {"layer2": lambda l, g: l == 2, "layers3_31": lambda l, g: l >= 3,
          "layers2_16": lambda l, g: l <= 16, "layers17_31": lambda l, g: l >= 17}


def curve(book, grid, cond, group=None):
    """Mean gain per N over every measured head (group None: the whole-model curve, layers 2 to 31) or over the
    heads of one group's layers, and the whole-model SNR per N."""
    layer_ids = book["record"]["layers_measured"]
    keep = None if group is None else np.array([GROUPS[group](l, book["global_layers"]) for l in layer_ids])
    out, snr = np.empty(len(grid)), np.empty(len(grid))
    for j, N in enumerate(grid):
        gain, _, snr[j] = cell_quantities(book["per_condition"][cond][str(N)])
        out[j] = float(gain.mean()) if keep is None else float(gain[keep].mean())   # as group_means' "whole"; Llama has no local layers
    return out, snr


def validity(curves: dict, idx=None) -> tuple:
    """Per condition, the periods whose median whole-model SNR over the books is at least 3; common = valid under
    BASE and VONLY. PONLY's tilt uses the common periods that are also valid under PONLY."""
    idx = range(len(curves[CONDITIONS[0]])) if idx is None else idx
    valid = {c: np.median(np.stack([curves[c][i][1] for i in idx]), axis=0) >= VALID_SNR for c in CONDITIONS}
    common = np.logical_and.reduce([valid[c] for c in GATED])
    return valid, common


def set_tilts(curves: dict, grid, idx=None, groups: dict = None) -> tuple:
    idx = list(range(len(curves[CONDITIONS[0]]))) if idx is None else list(idx)
    valid, common = validity(curves, idx)
    src = curves if groups is None else groups
    use = {c: common if c in GATED else common & valid[c] for c in CONDITIONS}
    tilts = {c: metrics_c2.tilt(grid, np.mean(np.stack([src[c][i][0] for i in idx]), axis=0), use[c]) for c in CONDITIONS}
    return tilts, common, valid, use


def book_tilts(curves: dict, grid, i):
    common = np.logical_and.reduce([curves[c][i][1] >= VALID_SNR for c in GATED])
    use = {c: common if c in GATED else common & (curves[c][i][1] >= VALID_SNR) for c in CONDITIONS}
    return {c: metrics_c2.tilt(grid, curves[c][i][0], use[c]) for c in CONDITIONS}


def analyze(run_dir: str, spec: dict, seal_hash: str = None, code: dict = None, partial: bool = False,
            repo: str = ".") -> dict:
    grid = spec["drive"]["periods_N"]
    books, pa = load_books(run_dir, spec, seal_hash, code)
    expected = sorted(int(g) for v in spec["inputs"]["prompt_sets"].values() for g in v)
    found = sorted(b["gutenberg_id"] for b in books)
    complete = found == expected
    if not complete and not partial:
        raise SystemExit(f"Incomplete run: no outcome. Found {found}, expected {expected}.")
    if complete and partial:
        raise SystemExit("The run is complete; analyze it without --partial.")
    if not books:
        raise SystemExit("No book files in the run: there is nothing to look at, partial or not.")
    curves = {c: [curve(b, grid, c) for b in books] for c in CONDITIONS}
    tilts, common, valid, use = set_tilts(curves, grid)
    per_book = []
    for i, b in enumerate(books):
        t = book_tilts(curves, grid, i)
        per_book.append({"gutenberg_id": b["gutenberg_id"], "tilts": t, "T_V": metrics_c11.ratio(t, "VONLY"),
                         "T_P": metrics_c11.ratio(t, "PONLY"), "agrees": metrics_c11.book_agrees(t)})
    c11 = metrics_c11.path_outcome(tilts, [x["tilts"] for x in per_book])
    if not complete:                       # a partial look has no outcome; its numbers are kept apart
        c11 = {"outcome": None, "partial_diagnostic": c11}
    mean = {c: np.mean(np.stack([x[0] for x in curves[c]]), axis=0) for c in CONDITIONS}
    groups = {}
    for name in GROUPS:
        gc = {c: [curve(b, grid, c, name) for b in books] for c in CONDITIONS}
        gt = set_tilts(curves, grid, groups=gc)[0]
        groups[name] = {"tilts": gt, "T_V": metrics_c11.ratio(gt, "VONLY"), "T_P": metrics_c11.ratio(gt, "PONLY"),
                        "curves": {c: np.mean(np.stack([x[0] for x in gc[c]]), axis=0).tolist() for c in CONDITIONS}}
    result = {"run_dir": run_dir, "complete": complete, "grid": grid, "books": len(books),
              "failure_records": sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "*.FAILED.*.json"))),
              "C13": c11, "tilts": tilts, "periods_used": {c: [g for g, v in zip(grid, use[c]) if v] for c in CONDITIONS},
              "valid": {c: v.tolist() for c, v in valid.items()},
              "curves": {c: mean[c].tolist() for c in CONDITIONS},
              "gain_ratio_per_N": {c: (mean[c] / mean["BASE"]).tolist() for c in CONDITIONS[1:]},
              "snr_median_per_N": {c: np.median(np.stack([x[1] for x in curves[c]]), axis=0).tolist() for c in CONDITIONS},
              "book_tilts": per_book, "diagnostic_groups": groups,
              "perplexity_median": float(np.median([b["perplexity"] for b in books])),
              "phase_a_rows": pa["rows"],
              "c11_reference": {"T_V": 1.865, "T_V_95": [1.813, 1.927], "tilt_BASE": 0.137, "run": "c11v1-20261110T040043Z"}}
    if not complete:
        result["warning"] = "PARTIAL: diagnostic look only. An incomplete run has no C13 outcome."
    rng = np.random.default_rng([0, 0])
    n = len(books)
    boot = {"T_V": [], "T_P": [], "tilt_BASE": [], "tilt_VONLY": []}
    for _ in range(BOOT):
        pick = rng.integers(0, n, size=n)
        t2 = set_tilts(curves, grid, idx=pick)[0]
        if t2["BASE"] is None or t2["BASE"] <= 0:
            continue
        boot["tilt_BASE"].append(t2["BASE"])
        if t2["VONLY"] is not None:
            boot["T_V"].append(t2["VONLY"] / t2["BASE"])
            boot["tilt_VONLY"].append(t2["VONLY"])
        if t2["PONLY"] is not None:
            boot["T_P"].append(t2["PONLY"] / t2["BASE"])
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
    labels = {"BASE": "the model as it is", "VONLY": "probe reaches the values only (patterns held)",
              "PONLY": "probe reaches queries and keys only (values held)"}
    for c in CONDITIONS:
        ax.plot(grid, np.asarray(result["curves"][c]) / ref, marker="o", label=labels[c])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Drive period N (tokens)")
    ax.set_ylabel("Gain relative to BASE at N = 4096"); ax.legend()
    oc = result["C13"]["outcome"]
    ax.set_title(f"Llama-3.1-8B-Instruct, set A tokens 0 to 32,767: lean path, outcome {oc}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--partial", action="store_true", help="diagnostic look at an incomplete run; never the reported result")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    seal_hash = glimmer.check_seal(repo, SEALED)
    spec = yaml.safe_load(open(os.path.join(repo, SEALED)))
    code = glimmer.check_code(repo, spec)
    from datetime import datetime, timezone
    from .run_c13 import runs_under_seal, write_json
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
        raise SystemExit(f"C13 was stopped by rule in Phase A: {json.load(open(pa_path))['stopped']}. Outcome U. "
                         f"Cite {os.path.relpath(pa_path, repo)}.")
    result = analyze(run_dir, spec, seal_hash, code, partial=args.partial, repo=repo)
    result["runs_under_seal"] = runs
    stamp = datetime.now(timezone.utc).strftime("-PARTIAL-%Y%m%dT%H%M%SZ")
    out = os.path.join(repo, "results", "derived", args.run_id + (stamp if args.partial else ""))
    if os.path.exists(os.path.join(out, "c13_summary.json")):
        raise SystemExit(f"{out}/c13_summary.json exists; derived results are not overwritten.")
    os.makedirs(out, exist_ok=True)
    result["cites"] = sorted(os.path.relpath(p, repo) for p in glob.glob(os.path.join(run_dir, "*.json")))
    result["seal_sha256"], result["code_sha256"] = seal_hash, code
    write_json(os.path.join(out, "c13_summary.json"), result, indent=2)
    plot(result, os.path.join(out, "c13_curves.png"))
    c = result["C13"] if result["complete"] else result["C13"]["partial_diagnostic"]
    b = result["bootstrap_95"]
    f = lambda v: "n/a" if v is None else f"{v:.3f}"
    ci = lambda k: f"95% {b[k][0]:.3f} to {b[k][1]:.3f}" if b.get(k) else "95% n/a"
    print(f"== C13 outcome: {result['C13']['outcome']} (complete={result['complete']}); C11 in Glimmer was VX, T_V 1.865"
          + (f"; {c['reason']}" if c.get("reason") else ""))
    print(f"   tilts: BASE {f(result['tilts']['BASE'])}, VONLY {f(result['tilts']['VONLY'])}, "
          f"PONLY {f(result['tilts']['PONLY'])}")
    print(f"   T_V = {f(c.get('T_V'))} ({ci('T_V')}); books agreeing {c.get('books_agreeing')}/5; "
          f"T_P = {f(c.get('T_P'))} ({ci('T_P')})")
    for name, g in result["diagnostic_groups"].items():
        print(f"   {name}: T_V {f(g['T_V'])}, T_P {f(g['T_P'])}")
    print(f"   summary: {os.path.relpath(out, repo)}/c13_summary.json")


if __name__ == "__main__":
    main()
