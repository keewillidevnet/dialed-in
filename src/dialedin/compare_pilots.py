"""Compares C5 pilot runs (one adapter family each) under a ranking rule fixed before any family run.

    PYTHONPATH=src python3 -m dialedin.compare_pilots results/pilots/*.json

Exploratory: these are pilots on one book's held-out region with 4 held-out directions and one training seed each.
The winner is a candidate for a sealed experiment, never a reported result.

Rule (written 2026-10-28, before the family runs):
  eligible  Spearman(scale, mean tilt) <= -0.9 (the change grows with the dial, toward a flatter tilt),
            all 4 held-out directions change the same way at scale 1 and at scale 2,
            and the mean perplexity ratio at scale 1 is at most 1.10.
  winner    the eligible family with the largest mean relative change at scale 1.
  tie       eligible families whose scale-1 changes are within 2 percentage points of the winner's.
"""
import json
import sys

MIN_SPEARMAN = -0.9
MAX_PPL_SCALE1 = 1.10
TIE_POINTS = 0.02


def row_at(dose, scale):
    return next(r for r in dose["rows"] if r["scale"] == scale)


def summarize(path: str) -> dict:
    rep = json.load(open(path))
    dose = rep["dose_response"]
    r1, r2 = row_at(dose, 1.0), row_at(dose, 2.0)
    same_way = all((r["relative_change_min"] < 0) == (r["relative_change_max"] < 0) and r["relative_change_max"] != 0
                   for r in (r1, r2))
    kl = [s["kl"] for s in rep.get("train", [])][-20:]
    out = {"file": path, "family": rep["lora"].get("family", "all"), "params": rep["lora"]["params"],
           "steps": len(rep.get("train", [])), "spearman": dose["spearman_scale_vs_tilt"],
           "change_1": r1["relative_change_mean"], "change_2": r2["relative_change_mean"],
           "spread_1": [r1["relative_change_min"], r1["relative_change_max"]],
           "ppl_1": r1["perplexity_ratio"], "ppl_2": r2["perplexity_ratio"],
           "same_way": same_way, "train_kl_last20": sum(kl) / len(kl) if kl else None}
    out["eligible"] = bool(out["spearman"] <= MIN_SPEARMAN and same_way and out["ppl_1"] <= MAX_PPL_SCALE1)
    return out


def rank(rows: list) -> dict:
    ok = [r for r in rows if r["eligible"]]
    if not ok:
        return {"winner": None, "ties": [], "note": "no family met the rule"}
    best = min(ok, key=lambda r: r["change_1"])                  # most negative = largest flattening
    ties = [r["family"] for r in ok if r is not best and abs(r["change_1"] - best["change_1"]) <= TIE_POINTS]
    return {"winner": best["family"], "ties": ties}


def main(paths):
    rows = [summarize(p) for p in paths]
    print(f"{'family':<6} {'params':>11} {'steps':>5} {'chg@1':>7} {'chg@2':>7} {'dirs@1':>16} {'rho':>6} "
          f"{'ppl@1':>6} {'ppl@2':>6} {'KL':>8}  eligible")
    for r in sorted(rows, key=lambda r: r["change_1"]):
        kl = f"{r['train_kl_last20']:.3f}" if r["train_kl_last20"] is not None else "-"
        print(f"{r['family']:<6} {r['params']:>11,} {r['steps']:>5} {100 * r['change_1']:>+6.1f}% "
              f"{100 * r['change_2']:>+6.1f}% {100 * r['spread_1'][0]:>+6.1f}..{100 * r['spread_1'][1]:>+5.1f}% "
              f"{r['spearman']:>6.2f} {r['ppl_1']:>6.3f} {r['ppl_2']:>6.3f} {kl:>8}  {r['eligible']}")
    res = rank(rows)
    print(f"== winner: {res['winner']}" + (f"; tied within 2 points: {res['ties']}" if res.get("ties") else "")
          + (f" ({res['note']})" if res.get("note") else ""))
    return rows, res


if __name__ == "__main__":
    main(sys.argv[1:])
