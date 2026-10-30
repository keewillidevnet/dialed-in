"""C9 analysis, exactly as sealed: is layer 2's v_proj necessary inside the full dial, and specifically so? Set A,
tokens 32,768 to 65,535, six conditions.

    PYTHONPATH=src python3 -m dialedin.analyze_c9 --run-id <id>

Forked from analyze_c8.py. Per-cell gains and SNR come from analyze_c1v5.cell_quantities, tilts from metrics_c2.tilt,
the outcome from metrics_c9. Reads only Phase B book files for curves; Phase A is checked, never curved.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import yaml

from . import glimmer, metrics, metrics_c2, metrics_c9
from .analyze_c1v5 import cell_quantities

SEALED = "criteria/sealed/c9-glimmer-necessity-v1.yaml"
CONDITIONS = metrics_c9.CONDITIONS
DOWN = lambda l: l >= 5                  # the downstream window: heads in layers 5 to 51
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
        problems.append(f"Phase A stopped C9 by rule ({pa['stopped']}): outcome U, no curves")
    names = pa["record"]["adapter_modules"]
    for c in CONDITIONS:
        sealed = spec["conditions"][c]["modules"]
        want = (list(names) if sealed == "all" else
                [n for n in names if n not in sealed["all_except"]] if isinstance(sealed, dict) else list(sealed))
        if pa["record"]["conditions"][c]["modules_on"] != want:
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
    (b) layers 3 to 51, (c) layers 5 to 51 (downstream of layer 2's and layer 4's modules), (d) layers 6 to 51 (no adapter), fit over
    the main periods. A group with no measured layer is None."""
    groups = {"layer2": lambda l: l == 2, "layers3_51": lambda l: l >= 3, "layers5_51": lambda l: l >= 5,
              "layers6_51": lambda l: l >= 6}
    out = {}
    measured = books[0]["record"]["layers_measured"]
    for name, fn in groups.items():
        if not any(fn(l) for l in measured):
            out[name] = None
            continue
        t = {c: metrics_c2.tilt(grid, np.mean(np.stack([curve(b, grid, c, fn)[0] for b in books]), axis=0), common)
             for c in CONDITIONS}
        rel = {c: ((t[c] - t["BASE"]) / t["BASE"] if None not in (t[c], t["BASE"]) and t["BASE"] else None)
               for c in CONDITIONS if c != "BASE"}
        out[name] = {"tilts": t, "relative_change": rel, "retained": metrics_c9.retained(t)}
    return out


def window_tilts(curves5: dict, grid, common, idx=None) -> dict:
    """Set-level downstream tilt per condition from the books in idx (all if None), fit over the main periods."""
    idx = range(len(curves5[CONDITIONS[0]])) if idx is None else idx
    return {c: metrics_c2.tilt(grid, np.mean(np.stack([curves5[c][i][0] for i in idx]), axis=0), common)
            for c in CONDITIONS}


def analyze(run_dir: str, spec: dict, seal_hash: str = None, code: dict = None, partial: bool = False) -> dict:
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
    curves5 = {c: [curve(b, grid, c, DOWN) for b in books] for c in CONDITIONS}
    tilts, common, valid = set_tilts(curves, grid)
    down = window_tilts(curves5, grid, common) if common.sum() else {c: None for c in CONDITIONS}
    per_book = []
    for i, b in enumerate(books):
        t = book_tilts(curves, grid, i)
        ok = np.logical_and.reduce([curves[c][i][1] >= VALID_SNR for c in CONDITIONS])
        t5 = {c: metrics_c2.tilt(grid, curves5[c][i][0], ok) for c in CONDITIONS}
        per_book.append({"gutenberg_id": b["gutenberg_id"], "tilts": t, "retained": metrics_c9.retained(t),
                         "ordered": metrics_c9.book_orders(t), "tilts_downstream": t5,
                         "retained_downstream": metrics_c9.retained(t5)})
    c9 = metrics_c9.necessity_outcome(tilts, down, [x["tilts"] for x in per_book])
    if not complete:                       # a partial look has no outcome; its numbers are kept apart
        c9 = {"outcome": None, "partial_diagnostic": c9}
    ppl = {c: float(np.median([b["perplexity"][c] / b["perplexity"]["BASE"] for b in books])) for c in CONDITIONS}
    result = {"run_dir": run_dir, "complete": complete, "grid": grid, "books": len(books),
              "failure_records": sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "*.FAILED.*.json"))),
              "C9": c9, "tilts": tilts, "tilts_downstream": down, "periods_used": [g for g, v in zip(grid, common) if v],
              "valid": {c: v.tolist() for c, v in valid.items()},
              "curves": {c: np.mean(np.stack([x[0] for x in curves[c]]), axis=0).tolist() for c in CONDITIONS},
              "book_tilts": per_book, "perplexity_ratio_median": ppl,
              "modules_on": {c: len(v["modules_on"]) for c, v in pa["record"]["conditions"].items()},
              "diagnostic_groups": group_diagnostic(books, grid, common),
              "reproduction": "not applicable: no earlier run measured set A at tokens 32,768 to 65,535"}
    mean_book = {}
    for key in ("retained", "retained_downstream"):
        for c in ("NOV2", "NOV4", "NOG2"):
            v = [x[key][c] for x in per_book if x[key][c] is not None]
            mean_book[f"{key}_{c}"] = float(np.mean(v)) if v else None
    result["secondary_mean_of_book_shares"] = dict(mean_book, books=len(per_book),
                                                   note="Muse Wonder's proposed per-book mean; reported, not gated")
    if not complete:
        result["warning"] = "PARTIAL: diagnostic look only. An incomplete run has no C9 outcome."
    rng = np.random.default_rng([0, 0])
    n = len(books)
    boot = {"retained_V2": [], "retained_NOV2": [], "retained_NOV4": [], "retained_NOG2": [], "full_change": [],
            "retained5_V2": [], "retained5_NOV2": [], "retained5_NOV4": [], "full_change5": []}
    for _ in range(BOOT):
        pick = rng.integers(0, n, size=n)
        t2, common2, _ = set_tilts({c: [curves[c][i] for i in pick] for c in CONDITIONS}, grid)
        if None not in t2.values() and t2["BASE"] > 0:      # each window is resampled and judged on its own
            fc = (t2["FULL"] - t2["BASE"]) / t2["BASE"]
            boot["full_change"].append(fc)
            if fc <= -metrics_c9.MIN_FULL_CHANGE:  # shares only where FULL flattens enough to split, as the outcome
                r = metrics_c9.retained(t2)
                for c in ("V2", "NOV2", "NOV4", "NOG2"):
                    boot[f"retained_{c}"].append(r[c])
        if common2.sum() == 0:
            continue
        d2 = window_tilts(curves5, grid, common2, pick)
        if None in d2.values() or d2["BASE"] <= 0:
            continue
        fc5 = (d2["FULL"] - d2["BASE"]) / d2["BASE"]
        boot["full_change5"].append(fc5)
        if fc5 <= -metrics_c9.MIN_DOWN_CHANGE:
            r5 = metrics_c9.retained(d2)
            for c in ("V2", "NOV2", "NOV4"):
                boot[f"retained5_{c}"].append(r5[c])
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
    labels = {"BASE": "dial 0", "FULL": "dial 2, all 32 matrices", "V2": "dial 2, layer 2 v only",
              "NOV2": "dial 2, all but layer 2 v", "NOV4": "dial 2, all but layer 4 v (control)",
              "NOG2": "dial 2, all but layer 2 gate (control)"}
    for c in CONDITIONS:
        ax.plot(grid, np.asarray(result["curves"][c]) / ref, marker="o", label=labels[c])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Drive period N (tokens)")
    ax.set_ylabel("Gain relative to dial 0 at N = 4096"); ax.legend()
    ax.set_title(f"Muse Glimmer-30B, set A tokens 32,768 to 65,535: necessity, outcome {result['C9']['outcome']}")
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
    from .run_c9 import runs_under_seal, write_json
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
        raise SystemExit(f"C9 was stopped by rule in Phase A: {json.load(open(pa_path))['stopped']}. Outcome U. "
                         f"Cite {os.path.relpath(pa_path, repo)}.")
    result = analyze(run_dir, spec, seal_hash, code, partial=args.partial)
    result["runs_under_seal"] = runs
    stamp = datetime.now(timezone.utc).strftime("-PARTIAL-%Y%m%dT%H%M%SZ")
    out = os.path.join(repo, "results", "derived", args.run_id + (stamp if args.partial else ""))
    if os.path.exists(os.path.join(out, "c9_summary.json")):
        raise SystemExit(f"{out}/c9_summary.json exists; derived results are not overwritten.")
    os.makedirs(out, exist_ok=True)
    result["cites"] = sorted(os.path.relpath(p, repo) for p in glob.glob(os.path.join(run_dir, "*.json")))
    result["seal_sha256"], result["code_sha256"] = seal_hash, code
    write_json(os.path.join(out, "c9_summary.json"), result, indent=2)
    plot(result, os.path.join(out, "c9_curves.png"))
    c, b = result["C9"], result["bootstrap_95"]
    pct = lambda v: "n/a" if v is None else f"{100 * v:.0f}%"
    ci = lambda k: f"95% {pct(b[k][0])} to {pct(b[k][1])}" if b.get(k) else "95% n/a"
    print(f"== C9 outcome: {c['outcome']} (complete={result['complete']})" + (f"; {c['reason']}" if c.get("reason") else ""))
    if "retained_NOV2" in c:
        print(f"   FULL main tilt change {100 * c['full_change']:+.1f}%; V2 alone keeps {pct(c['retained_V2'])} "
              f"({ci('retained_V2')})")
        print(f"   main (layers 2 to 51): without layer 2 v keeps {pct(c['retained_NOV2'])} ({ci('retained_NOV2')}); "
              f"without layer 2 gate {pct(c['retained_NOG2'])} ({ci('retained_NOG2')}); without layer 4 v "
              f"{pct(c['retained_NOV4'])}; books ordered {c['books_ordered']}/5")
        d = c["downstream"]
        if "retained_NOV2" in d:
            print(f"   downstream (layers 5 to 51): FULL {100 * d['full_change']:+.1f}%; without layer 2 v keeps "
                  f"{pct(d['retained_NOV2'])} ({ci('retained5_NOV2')}); without layer 4 v {pct(d['retained_NOV4'])} "
                  f"({ci('retained5_NOV4')}); judged {d['judged']}" + (f" ({d['why']})" if d.get("why") else ""))
        print(f"   V2 alone minus what removing it loses: {c['gap_sufficiency_minus_necessity']:+.3f}")
    print(f"   perplexity ratios {', '.join(f'{k} x{v:.4f}' for k, v in result['perplexity_ratio_median'].items())}")
    print(f"   summary: {os.path.relpath(out, repo)}/c9_summary.json")


if __name__ == "__main__":
    main()
