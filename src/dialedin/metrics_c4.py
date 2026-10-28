"""C4 metrics: structural levers (window, RoPE base) and text controls. Implements
criteria/sealed/c4-glimmer-structure-and-text-v1.yaml. Tilts come from metrics_c2.tilt. Pure Python.
"""
EFFECT_FRACTION = 0.25          # |tilt(high) - tilt(low)| must reach this fraction of |tilt(BASE)|
MIN_BOOKS_AGREE = 4             # of the 5 books, at least 4 must change in the set's direction
MAX_PPL_RATIO = 1.10            # a lever is "without damage" if median perplexity at both ends is at most 10 percent above BASE
ARCH_FRACTION = 0.5             # text controls keep the tilt if each is at least half of BASE's (and positive)
TEXT_FRACTION = 0.25            # text shapes the tilt if either control is at most a quarter of BASE's

LEVERS = {"window": ("W1024", "BASE", "W4096"), "rope": ("R0.5", "BASE", "R2.0")}


def lever_outcome(tilts: dict, book_deltas: list, ppl_ratio_low: float, ppl_ratio_high: float, lever: str) -> dict:
    """tilts: set-level tilts for the lever's three conditions; book_deltas: per-book tilt(high) - tilt(low),
    None where a book's tilt is undefined (such a book does not count as agreeing).

    D1 dial: a monotone order low < BASE < high or the reverse, |delta| >= EFFECT_FRACTION * |tilt(BASE)|, at least
       MIN_BOOKS_AGREE books changing the same way as the set, and median perplexity ratio <= MAX_PPL_RATIO at both ends.
    D2 dial with damage: the same but a perplexity ratio above MAX_PPL_RATIO.
    D0 no dial: every tilt defined and neither holds.  U: a tilt is undefined.
    """
    low, mid, high = (tilts[c] for c in LEVERS[lever])
    if None in (low, mid, high):
        return {"outcome": "U", "reason": "a tilt is undefined"}
    delta = high - low
    order = "+" if low < mid < high else "-" if low > mid > high else None
    effect = abs(delta) >= EFFECT_FRACTION * abs(mid)
    agree = sum(1 for d in book_deltas if d is not None and d * delta > 0)
    consistent = agree >= MIN_BOOKS_AGREE
    damaged = max(ppl_ratio_low, ppl_ratio_high) > MAX_PPL_RATIO
    dial = bool(order and effect and consistent)
    outcome = ("D2" if damaged else "D1") if dial else "D0"
    return {"outcome": outcome, "order": order, "delta": delta, "needed": EFFECT_FRACTION * abs(mid),
            "effect": bool(effect), "books_agree": agree, "consistent": bool(consistent),
            "ppl_ratio_low": ppl_ratio_low, "ppl_ratio_high": ppl_ratio_high, "damaged": bool(damaged),
            "tilts": {"low": low, "base": mid, "high": high}}


def text_outcome(base: float, shuffled: float, constant: float) -> dict:
    """A architectural: both controls positive and at least ARCH_FRACTION of BASE's tilt.
    X text-shaped: either control at most TEXT_FRACTION of BASE's tilt (including zero or negative).
    M mixed: otherwise.  U: a tilt undefined or BASE's tilt not positive."""
    if None in (base, shuffled, constant) or base <= 0:
        return {"outcome": "U", "tilts": {"base": base, "shuffled": shuffled, "constant": constant}}
    rs, rc = shuffled / base, constant / base
    if rs >= ARCH_FRACTION and rc >= ARCH_FRACTION:
        outcome = "A"
    elif rs <= TEXT_FRACTION or rc <= TEXT_FRACTION:
        outcome = "X"
    else:
        outcome = "M"
    return {"outcome": outcome, "shuffled_over_base": rs, "constant_over_base": rc,
            "tilts": {"base": base, "shuffled": shuffled, "constant": constant}}
