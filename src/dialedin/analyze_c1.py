"""C1 analysis, exactly as sealed. Reads only Phase B book files for curves; Phase A is checked, never curved.

    PYTHONPATH=src python3 -m dialedin.analyze_c1 --run-id <id>
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import yaml

from . import glimmer, metrics

SEALED = "criteria/sealed/c1-glimmer-frequency-response-v4.yaml"
BOOT = 1000
VALID_SNR = 3.0


def load_books(run_dir: str, spec: dict, seal_hash: str = None, code: dict = None):
    """Loads book files and refuses any that ran under another seal, other code, another a or platform,
    another text, or a dirty sham. phaseA.json is required."""
    books = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(run_dir, "book_*_*.json")))
             if re.fullmatch(r"book_[A-Za-z]+_\d+\.json", os.path.basename(p))]      # never ATTEMPT or FAILED records
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
        if b["sham_differences_open"] != 0 or b["sham_differences_close"] != 0:
            problems.append(f"{tag}: sham not clean")
        if seal_hash is not None and b["record"]["seal_sha256"] != seal_hash:
            problems.append(f"{tag}: seal hash differs")
        if code is not None and b["record"]["code_sha256"] != code:
            problems.append(f"{tag}: code hashes differ")
        want_text = texts.get(int(b["gutenberg_id"]))
        if want_text is not None and b["record"].get("text_sha256") != want_text:
            problems.append(f"{tag}: text hash differs from the sealed value")
        if b["a"] != pa["chosen_a"]:
            problems.append(f"{tag}: a={b['a']} but Phase A chose {pa['chosen_a']}")
        if b["record"]["platform"] != pa["record"]["platform"]:
            problems.append(f"{tag}: platform differs from Phase A")
    if seal_hash is not None and pa["record"]["seal_sha256"] != seal_hash:
        problems.append("Phase A seal hash differs")
    if code is not None and pa["record"]["code_sha256"] != code:
        problems.append("Phase A code hashes differ")
    if problems:
        raise SystemExit("Refusing to analyze:\n  " + "\n  ".join(sorted(set(problems))))
    return books


def cell_quantities(cell: dict):
    """Gain, noise-corrected gain [layers, heads] and whole-model SNR for one periodic cell."""
    target = np.asarray(cell["target"], dtype=np.float64)
    floor = np.asarray(cell["floor"], dtype=np.float64)
    gain, corrected = metrics.gains(target, floor, cell["input_mag"])
    snr = metrics.aggregate_snr(target, np.asarray(cell["bin_power"], dtype=np.float64), cell["n_below"])
    return gain, corrected, snr


def book_curves(book, grid, key="gain"):
    """Per book: whole, local and global mean gain per N, and the whole-model SNR per N."""
    layer_ids = book["record"]["layers_measured"]
    out = {g: np.empty(len(grid)) for g in ("whole", "local", "global")}
    snr = np.empty(len(grid))
    for j, N in enumerate(grid):
        gain, corrected, snr[j] = cell_quantities(book["per_N"][str(N)])
        gm = metrics.group_means(gain if key == "gain" else corrected, layer_ids, book["global_layers"])
        for g in out:
            out[g][j] = gm[g]
    return out, snr


def set_summary(curves):
    """curves: list of book_curves results. Validity: median over books of whole-model SNR >= 3."""
    valid = np.median(np.stack([c[1] for c in curves]), axis=0) >= VALID_SNR
    mean = {g: np.mean(np.stack([c[0][g] for c in curves]), axis=0) for g in ("whole", "local", "global")}
    return mean, valid


def analyze(run_dir: str, spec: dict, seal_hash: str = None, code: dict = None, partial: bool = False) -> dict:
    grid = spec["pulse"]["interval_grid_N"]
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

    result = {"run_dir": run_dir, "complete": complete, "grid": grid, "books": len(books),
              "failure_records": failures}
    if not complete:
        result["warning"] = "PARTIAL: diagnostic look only. An incomplete run has no C1 outcome."
    sets = sorted(by_set)
    per_book = {s: [book_curves(b, grid) for b in by_set[s]] for s in sets}
    whole_norm, valid, local_norm, global_norm = {}, {}, {}, {}
    for s in sets:
        mean, v = set_summary(per_book[s])
        valid[s] = v
        whole_norm[s] = metrics.normalize(mean["whole"], v)
        local_norm[s] = metrics.normalize(mean["local"], v)
        global_norm[s] = metrics.normalize(mean["global"], v)
    c1 = metrics.classify_c1(grid, whole_norm, valid)
    if not complete:
        c1["outcome"] = None
    result["C1"] = c1
    result["curves"] = {s: {"whole": whole_norm[s].tolist(), "local": local_norm[s].tolist(),
                            "global": global_norm[s].tolist(), "valid": valid[s].tolist(),
                            "snr_median_over_books": np.median(np.stack([c[1] for c in per_book[s]]), axis=0).tolist()}
                        for s in sets}

    # C1b: descriptive only, never a gate.
    loc = np.nanmean(np.stack([local_norm[s] for s in sets]), axis=0)
    glo = np.nanmean(np.stack([global_norm[s] for s in sets]), axis=0)
    result["C1b_descriptive"] = {"local_mean_over_sets": loc.tolist(), "global_mean_over_sets": glo.tolist(),
                                 "notch_index_local_2048": metrics.notch_index(grid, loc),
                                 "notch_index_global_2048": metrics.notch_index(grid, glo),
                                 "box_filter_2048_reference": metrics.box_filter_null(grid, 2048),
                                 "note": "Reported, not a pass/fail check. The structural test is C2."}

    # Bootstrap within each set: resample books, recompute everything downstream.
    boot = {}
    for si, s in enumerate(sets):
        rng = np.random.default_rng([0, si])
        bc = per_book[s]
        cut, peak, dirs = {"HP": [], "LP": []}, [], []
        for _ in range(BOOT):
            pick = rng.integers(0, len(bc), size=len(bc))
            mean, v = set_summary([bc[i] for i in pick])
            curve = metrics.normalize(mean["whole"], v)
            if v.sum() >= 1:
                co = metrics.cutoff(grid, curve, v)
                dirs.append(co["direction"])
                if co["cutoff_N"] is not None:
                    cut[co["direction"]].append(co["cutoff_N"])
                vi = np.where(v)[0]
                peak.append(grid[int(vi[np.argmax(curve[vi])])])
        peaks = np.array(peak)
        boot[s] = {"cutoff_N_95_by_direction": {d: (np.percentile(v, [2.5, 97.5]).tolist() if v else None)
                                                for d, v in cut.items()},
                   "cutoff_found_fraction": float((len(cut["HP"]) + len(cut["LP"])) / BOOT),
                   "direction_fractions": {d: dirs.count(d) / BOOT for d in ("HP", "LP", None)},
                   "peak_N_95": np.percentile(peaks, [2.5, 97.5]).tolist() if len(peaks) else None,
                   "distinct_resamples_possible": 126 if len(bc) == 5 else None}
    result["bootstrap"] = {str(k): {kk: ({str(d): f for d, f in vv.items()} if isinstance(vv, dict) else vv)
                                    for kk, vv in v.items()} for k, v in boot.items()}

    # Diagnostics, labeled as such.
    diag_norm, diag_valid = {}, {}
    for s in sets:
        mean, v = set_summary([book_curves(b, grid, key="corrected") for b in by_set[s]])
        diag_norm[s], diag_valid[s] = metrics.normalize(mean["whole"], v), v
    result["diagnostic_noise_corrected_outcome"] = metrics.classify_c1(grid, diag_norm, diag_valid)["outcome"]
    ratio = {}
    for N in grid:
        r = []
        for b in books:
            cell = b["per_N"][str(N)]
            gp = np.mean(np.asarray(cell["target"]) / cell["input_mag"])
            ap = cell["aperiodic"]
            ga = np.mean(np.asarray(ap["target"]) / ap["input_mag"]) if ap["input_mag"] > 0 else np.nan
            r.append(ga / gp if gp > 0 else np.nan)
        ratio[N] = float(np.nanmedian(r))
    result["diagnostic_aperiodic_over_periodic_median"] = ratio
    result["realized_pulse_off_direction_median"] = float(np.median(
        [b["per_N"][str(N)]["realized_pulse"]["off_over_amplitude_median"] for b in books for N in grid]))
    return result


def plot(result: dict, path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    grid = result["grid"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s, c in result["curves"].items():
        ax.plot(grid, c["whole"], marker="o", label=f"set {s}")
    ax.set_xscale("log")
    ax.set_xlabel("Pulse interval N (tokens); frequency is 1/N")
    ax.set_ylabel("Normalized gain")
    ax.set_title(f"Muse Glimmer-30B frequency response: outcome {result['C1']['outcome']}")
    ax.legend()
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
    from .run_c1 import runs_under_seal, write_json
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
        raise SystemExit(f"C1 was stopped by rule in Phase A: {pa['stopped']}. Outcome U (inconclusive), not a kill. "
                         f"Cite {os.path.relpath(pa_path, repo)}.")
    result = analyze(run_dir, spec, seal_hash, code, partial=args.partial)
    result["runs_under_seal"] = runs
    stamp = datetime.now(timezone.utc).strftime("-PARTIAL-%Y%m%dT%H%M%SZ")
    out = os.path.join(repo, "results", "derived", args.run_id + (stamp if args.partial else ""))
    if os.path.exists(os.path.join(out, "c1_summary.json")):
        raise SystemExit(f"{out}/c1_summary.json exists; derived results are not overwritten.")
    os.makedirs(out, exist_ok=True)
    result["cites"] = sorted(os.path.relpath(p, repo) for p in glob.glob(os.path.join(run_dir, "*.json")))
    result["seal_sha256"], result["code_sha256"] = seal_hash, code
    write_json(os.path.join(out, "c1_summary.json"), result, indent=2)
    plot(result, os.path.join(out, "c1_curves.png"))
    print(f"== C1 outcome: {result['C1']['outcome']}  (complete={result['complete']})")
    print(f"   summary: {os.path.relpath(out, repo)}/c1_summary.json")


if __name__ == "__main__":
    main()
