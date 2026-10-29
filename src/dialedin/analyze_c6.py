"""C6 analysis, exactly as sealed: does the C5 adapter's scale raise the excess loss after a topic switch?

    PYTHONPATH=src python3 -m dialedin.analyze_c6 --run-id <id>

Reads switch.json; the outcome comes from metrics_c6. Secondary values (reported, never gated): the split between target
books the adapter trained on (other regions) and books it never saw, early and late bins, noise excess, clean loss.
"""
import argparse
import glob
import json
import os

import numpy as np
import yaml

from . import glimmer, metrics, metrics_c6

SEALED = "criteria/sealed/c6-glimmer-dial-topic-switch-v2.yaml"
CONDITIONS = metrics_c6.CONDITIONS
BOOT = 1000
BINS = ((1, 16), (17, 64), (65, 256), (257, 1024), (1025, 2048))


def per_target(cond_data: dict, books: list) -> dict:
    """Mean switch excess over the pairs that switch INTO each target book."""
    out = {}
    for y in books:
        vals = [v for k, v in cond_data["switch_pair_total"].items() if int(k.split("->")[1]) == y]
        out[y] = float(np.mean(vals))
    return out


def per_source(cond_data: dict, books: list) -> dict:
    """Mean switch excess over the pairs that switch OUT OF each source book."""
    return {x: float(np.mean([v for k, v in cond_data["switch_pair_total"].items() if int(k.split("->")[0]) == x]))
            for x in books}


def split_totals(cond_data: dict, after: int) -> dict:
    """Mean over pairs of the summed switch loss and of the summed clean loss on the same tokens (excess = their
    difference): shows whether a change in excess comes from the switch input or from the clean input."""
    sw, cl = [], []
    for k, e in cond_data["switch_pair_total"].items():
        y = k.split("->")[1]
        clean_sum = after * cond_data["clean_mean_loss"][y]
        cl.append(clean_sum); sw.append(e + clean_sum)
    return {"switch_loss_sum": float(np.mean(sw)), "clean_loss_sum": float(np.mean(cl))}


def reproduction(d: dict, c5_behavior: dict) -> dict:
    """C6's set A pairs (trained adapter) against C5's behavior.json per-pair totals, S0 and S2."""
    out, worst = {}, 0.0
    c5 = {r["condition"]: r["per_pair_total"] for r in c5_behavior["switch"]}
    for cond, data in d["reproduction_of_c5"].items():
        for k, v in data["switch_pair_total"].items():
            ref = c5[cond][k]
            rel = abs(v - ref) / max(abs(ref), 1e-30)
            worst = max(worst, rel)
            out[f"{cond} {k}"] = {"c6": v, "c5": ref}
    return {"pairs": out, "max_relative_difference": worst}


def analyze(d: dict, spec: dict, c5_behavior: dict = None) -> dict:
    books = [int(b) for b in d["books"]]
    conds = d["conditions"]
    excess = {c: float(np.mean(list(conds[c]["switch_pair_total"].values()))) for c in CONDITIONS}
    tgt = {c: per_target(conds[c], books) for c in CONDITIONS}
    book_rise = [tgt["S2"][y] / tgt["S0"][y] - 1 if tgt["S0"][y] > 0 else None for y in books]
    c6 = metrics_c6.switch_outcome(excess, book_rise)
    rng = np.random.default_rng([0, 0])
    boot = []
    for _ in range(BOOT):
        pick = rng.integers(0, len(books), size=len(books))
        s0 = np.mean([tgt["S0"][books[i]] for i in pick])
        s2 = np.mean([tgt["S2"][books[i]] for i in pick])
        boot.append(s2 / s0 - 1)
    trained = set(int(r["gutenberg_id"]) for r in spec["adapter"]["trained_on_regions"])

    def group_rise(group):
        ys = [y for y in books if (y in trained) == group]
        if not ys:
            return None
        return float(np.mean([tgt["S2"][y] for y in ys]) / np.mean([tgt["S0"][y] for y in ys]) - 1)

    def binned(curve):                       # summed excess (nats) over the positions after the seam in each bin
        c = np.asarray(curve)
        return {f"{lo}-{hi}": float(c[lo - 1:hi].sum()) for lo, hi in BINS}

    after = spec["switch"]["after"]
    src = {c: per_source(conds[c], books) for c in ("S0", "S2")}
    loso = []
    for x in books:
        keep = lambda c: [v for k, v in conds[c]["switch_pair_total"].items() if int(k.split("->")[0]) != x]
        loso.append(float(np.mean(keep("S2")) / np.mean(keep("S0")) - 1))
    control = None
    if "control_random_adapter" in d:
        ctl = float(np.mean(list(d["control_random_adapter"]["S2"]["switch_pair_total"].values())))
        control = {"excess_S2": ctl, "relative_rise": ctl / excess["S0"] - 1,
                   "trained_relative_rise": c6.get("relative_rise"),
                   "noise_excess_S2": float(np.mean(list(d["control_random_adapter"]["S2"]["noise_book_total"].values()))),
                   "clean_mean_loss_S2": float(np.mean(list(d["control_random_adapter"]["S2"]["clean_mean_loss"].values()))),
                   "norms": d["control_random_adapter"]["norms"]}

    noise = {c: float(np.mean(list(conds[c]["noise_book_total"].values()))) for c in CONDITIONS}
    clean = {c: float(np.mean(list(conds[c]["clean_mean_loss"].values()))) for c in CONDITIONS}
    return {"C6": c6, "excess": excess, "per_target_book": {c: {str(k): v for k, v in tgt[c].items()} for c in CONDITIONS},
            "book_rise_S2": {str(y): r for y, r in zip(books, book_rise)},
            "bootstrap_95_relative_rise": np.percentile(boot, [2.5, 97.5]).tolist(),
            "control_random_adapter": control,
            "reproduction_of_c5": reproduction(d, c5_behavior) if (c5_behavior and "reproduction_of_c5" in d) else None,
            "secondary": {"switch_and_clean_loss_sums": {c: split_totals(conds[c], after) for c in CONDITIONS},
                          "source_book_rise_S2": {str(x): src["S2"][x] / src["S0"][x] - 1 for x in books},
                          "source_books_rising": int(sum(src["S2"][x] > src["S0"][x] for x in books)),
                          "leave_one_source_out_rise": {"min": min(loso), "max": max(loso)},
                          "rise_targets_trained_books": group_rise(True),
                          "rise_targets_untrained_books": group_rise(False),
                          "switch_excess_by_bin": {c: binned(conds[c]["switch_curve_mean"]) for c in CONDITIONS},
                          "noise_excess": noise,
                          "noise_trend_spearman": metrics.spearman(list(metrics_c6.SCALES), [noise[c] for c in CONDITIONS]),
                          "clean_mean_loss": clean,
                          "clean_perplexity_ratio": {c: float(np.exp(clean[c] - clean["S0"])) for c in CONDITIONS}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    seal_hash = glimmer.check_seal(repo, SEALED)
    spec = yaml.safe_load(open(os.path.join(repo, SEALED)))
    code = glimmer.check_code(repo, spec)
    from .run_c6 import runs_under_seal, write_json
    runs = runs_under_seal(os.path.join(repo, "results", "raw"), seal_hash)
    done = [r["run_id"] for r in runs if r["complete"]]
    if args.run_id not in [r["run_id"] for r in runs]:
        raise SystemExit(f"{args.run_id} has no STARTED.json under this seal.")
    if len(done) > 1:
        raise SystemExit(f"More than one complete run under this seal: {done}.")
    run_dir = os.path.join(repo, "results", "raw", args.run_id)
    path = os.path.join(run_dir, "switch.json")
    if not os.path.exists(path):
        raise SystemExit("No switch.json: the run is not complete, so there is no outcome.")
    d = json.load(open(path))
    rec = d["record"]
    problems = [m for m, bad in (("seal hash differs", rec["seal_sha256"] != seal_hash),
                                 ("code hashes differ", rec["code_sha256"] != code),
                                 ("adapter differs", rec["adapter_sha256"] != spec["adapter"]["sha256"]),
                                 ("setup checks failed", not (d["checks"]["adapter"]["as_expected"]
                                                              and d["checks"]["logits"]["as_expected"]))) if bad]
    if problems:
        raise SystemExit("Refusing to analyze: " + "; ".join(problems))
    c5b = json.load(open(os.path.join(repo, spec["reproduction_of_c5"]["c5_behavior"])))
    result = analyze(d, spec, c5b)
    result.update({"runs_under_seal": runs, "seal_sha256": seal_hash, "code_sha256": code,
                   "cites": sorted(os.path.relpath(p, repo) for p in glob.glob(os.path.join(run_dir, "*.json")))})
    out = os.path.join(repo, "results", "derived", args.run_id)
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "c6_summary.json"), result, indent=2)
    c = result["C6"]
    print(f"== C6 outcome: {c['outcome']}; switch excess S0 -> S2 {100 * c.get('relative_rise', float('nan')):+.1f}% "
          f"(95% interval {100 * result['bootstrap_95_relative_rise'][0]:+.1f}% to "
          f"{100 * result['bootstrap_95_relative_rise'][1]:+.1f}%); Spearman {c.get('spearman')}; "
          f"books rising {c.get('books_agree')}/10")
    ctl = result["control_random_adapter"]
    if ctl:
        print(f"   control (random adapter, matched norm) at S2: {100 * ctl['relative_rise']:+.1f}%")
    rp = result["reproduction_of_c5"]
    if rp:
        print(f"   reproduction of C5's set A pairs: max relative difference {rp['max_relative_difference']:.3g}")
    print(f"   summary: {os.path.relpath(out, repo)}/c6_summary.json")


if __name__ == "__main__":
    main()
