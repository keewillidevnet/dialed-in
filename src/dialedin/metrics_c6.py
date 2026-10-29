"""C6 metrics: does the C5 adapter's scale raise the excess loss after a topic switch? Implements
criteria/sealed/c6-glimmer-dial-topic-switch-v1.yaml. Pure Python plus numpy.
"""
import numpy as np

from .metrics import spearman

SCALES = (0.0, 0.5, 1.0, 1.5, 2.0)
CONDITIONS = ("S0", "S0.5", "S1", "S1.5", "S2")
MIN_SPEARMAN = 0.9              # excess must rise with the scale: Spearman(scale, excess) >= 0.9 over the 5 scales
MIN_RELATIVE_RISE = 0.05        # excess(S2) / excess(S0) - 1 >= 5 percent (C5's exploratory value was 11.4 percent)
MIN_BOOKS_AGREE = 8             # of the 10 target books, at least 8 must rise S0 -> S2


def switch_outcome(excess: dict, book_rise: list) -> dict:
    """excess: per condition, the mean over ordered pairs of the summed excess loss (nats) on the 2,048 tokens after
    the seam; book_rise: per target book, excess(S2) / excess(S0) - 1 over the pairs that switch into it (None if
    undefined, which counts as not agreeing).

    B1 confirmed: Spearman >= 0.9, the relative rise >= 5 percent, and at least 8 of 10 target books rising.
    B0 not confirmed: every value defined and B1 does not hold (a fall is B0, reported with its direction).
    U: a value undefined or not finite, or excess(S0) not positive.
    """
    vals = [excess.get(c) for c in CONDITIONS]
    if any(v is None or not np.isfinite(v) for v in vals) or vals[0] <= 0:
        return {"outcome": "U", "reason": "an excess is undefined, or excess(S0) is not positive", "excess": dict(excess)}
    rho = spearman(list(SCALES), vals)
    rise = excess["S2"] / excess["S0"] - 1
    agree = sum(1 for r in book_rise if r is not None and np.isfinite(r) and r > 0)
    ok = rho >= MIN_SPEARMAN and rise >= MIN_RELATIVE_RISE and agree >= MIN_BOOKS_AGREE
    return {"outcome": "B1" if ok else "B0", "spearman": rho, "monotone": bool(rho >= MIN_SPEARMAN),
            "relative_rise": rise, "effect": bool(rise >= MIN_RELATIVE_RISE), "books_agree": agree,
            "consistent": bool(agree >= MIN_BOOKS_AGREE), "direction": "rises" if rise > 0 else "falls",
            "excess": dict(excess)}
