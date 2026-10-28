"""C5 metrics: does a trained low-rank adapter act as a dial on the tilt? Implements
criteria/sealed/c5-glimmer-trained-dial-v1.yaml. Tilts come from metrics_c2.tilt. Pure Python.
"""
from .metrics import spearman

SCALES = (0.0, 0.5, 1.0, 1.5, 2.0)
CONDITIONS = ("S0", "S0.5", "S1", "S1.5", "S2")
MIN_ABS_SPEARMAN = 0.9          # the tilt must fall with the scale: Spearman(scale, tilt) <= -0.9 over the 5 scales
EFFECT_FRACTION = 0.25          # |tilt(S2) - tilt(S0)| must reach this fraction of |tilt(S0)|
MIN_BOOKS_AGREE = 4             # of the 5 books, at least 4 must change S0 -> S2 in the set's direction
MAX_PPL_RATIO = 1.10            # median perplexity at every nonzero scale at most 10 percent above S0


def dial_outcome(tilts: dict, book_deltas: list, ppl_ratios: dict) -> dict:
    """tilts: set-level tilt per condition (S0 .. S2); book_deltas: per-book tilt(S2) - tilt(S0), None where a book's
    tilt is undefined (such a book counts as disagreeing); ppl_ratios: per condition, median over books of
    perplexity / S0's.

    The adapter was trained to flatten, so a dial must flatten: tilt(S2) < tilt(S0) and Spearman <= -0.9.
    L1 dial: that, the effect, consistency, and no perplexity ratio above 1.10 at S0.5, S1, S1.5 or S2.
    L2 dial with damage: the same, with a perplexity ratio above 1.10 at some nonzero scale.
    L0 no dial: every tilt defined and L1 and L2 do not hold (a steepening is L0, reported with its direction).
    U: a tilt is undefined, or tilt(S0) is not positive (flattening is defined only for a low-pass tilt).
    """
    vals = [tilts[c] for c in CONDITIONS]
    if None in vals:
        return {"outcome": "U", "reason": "a tilt is undefined", "tilts": dict(tilts)}
    if tilts["S0"] <= 0:                     # C1 to C4 found a positive (low-pass) tilt; flattening is defined from it
        return {"outcome": "U", "reason": "tilt(S0) is not positive", "tilts": dict(tilts)}
    rho = spearman(list(SCALES), vals)
    delta = tilts["S2"] - tilts["S0"]
    monotone = rho <= -MIN_ABS_SPEARMAN
    flattens = delta < 0
    effect = abs(delta) >= EFFECT_FRACTION * abs(tilts["S0"])
    agree = sum(1 for d in book_deltas if d is not None and d * delta > 0)
    consistent = agree >= MIN_BOOKS_AGREE
    worst = max(ppl_ratios[c] for c in CONDITIONS[1:])
    damaged = worst > MAX_PPL_RATIO
    dial = bool(flattens and monotone and effect and consistent)
    outcome = ("L2" if damaged else "L1") if dial else "L0"
    return {"outcome": outcome, "spearman": rho, "monotone": bool(monotone), "delta": delta,
            "relative_change": delta / tilts["S0"] if tilts["S0"] else None,
            "needed": EFFECT_FRACTION * abs(tilts["S0"]), "effect": bool(effect), "books_agree": agree,
            "consistent": bool(consistent), "flattens": bool(flattens),
            "ppl_ratios": {c: ppl_ratios[c] for c in CONDITIONS[1:]}, "worst_ppl_ratio": worst,
            "damaged": bool(damaged), "direction": "flatter" if delta < 0 else "steeper", "tilts": dict(tilts)}
