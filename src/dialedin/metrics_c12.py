"""C12 metrics: which part pushes Glimmer's lean back, the attention patterns or the output gate?
Implements criteria/sealed/c12-glimmer-pushback-split-v1.yaml. Tilts come from metrics_c2.tilt. Pure Python.

C11 found tilt(VONLY) far above tilt(BASE): holding queries, keys and the gate at their undriven state nearly doubles
the lean. The excess, tilt(VONLY) - tilt(BASE), is what moving them takes away (the pushback). A held set X recovers
    share(X) = (tilt(X) - tilt(BASE)) / (tilt(VONLY) - tilt(BASE))
of the excess: 1 if holding X alone gives the whole of VONLY's lean (X carries the pushback), 0 if holding X changes
nothing (X carries none of it). HOLDQK holds queries and keys (the patterns); HOLDG holds the gate.
"""
CONDITIONS = ("BASE", "VONLY", "HOLDQK", "HOLDG")
GATED = ("BASE", "VONLY")
REPLICATION = (1.40, 2.40)      # T_V = tilt(VONLY) / tilt(BASE) on set B must lie here (C11, set A: 1.865, 95% 1.81 to 1.93)
HI, LO = 0.70, 0.30             # a share at or above HI carries most of the pushback; below LO, little of it


def shares(t: dict) -> dict:
    """share(HOLDQK) and share(HOLDG), or None where a tilt is undefined or the excess is not positive."""
    if any(t.get(c) is None for c in ("BASE", "VONLY")) or t["VONLY"] - t["BASE"] <= 0:
        return {"HOLDQK": None, "HOLDG": None}
    ex = t["VONLY"] - t["BASE"]
    return {c: (None if t.get(c) is None else (t[c] - t["BASE"]) / ex) for c in ("HOLDQK", "HOLDG")}


def classify(s: dict):
    """P: patterns carry the pushback (share QK >= 0.70, share G < 0.30). G: the gate does (the mirror).
    B: both carry a part (both shares >= 0.30). N: neither held alone recovers much (both below 0.30), so either path
    alone pushes the lean back. W: one share between 0.30 and 0.70 and the other below 0.30 (one path, partly)."""
    qk, g = s["HOLDQK"], s["HOLDG"]
    if qk is None or g is None:
        return None
    if qk >= HI and g < LO:
        return "P"
    if g >= HI and qk < LO:
        return "G"
    if qk >= LO and g >= LO:
        return "B"
    if qk < LO and g < LO:
        return "N"
    return "W"


def split_outcome(main: dict, book_main: list) -> dict:
    """main: set-level tilts over layers 2 to 51; book_main: per book, its tilts.

    U:  tilt(BASE) or tilt(VONLY) undefined, or tilt(BASE) not positive.
    R0: T_V outside 1.40 to 2.40: C11's effect does not replicate on set B; the split is not judged.
    U also if a held condition's tilt is undefined.
    Otherwise the set-level class (P, G, B, N or W, see classify), and the number of books whose own class matches.
    """
    common = {"tilts": dict(main)}
    if main.get("BASE") is None or main.get("VONLY") is None:
        return {"outcome": "U", "reason": "tilt(BASE) or tilt(VONLY) is undefined", **common}
    if main["BASE"] <= 0:
        return {"outcome": "U", "reason": "tilt(BASE) is not positive", **common}
    tv = main["VONLY"] / main["BASE"]
    common["T_V"] = tv
    lo, hi = REPLICATION
    if not lo <= tv <= hi:
        return {"outcome": "R0", "reason": f"T_V {tv:.3f} outside {lo} to {hi}: C11 does not replicate on set B", **common}
    s = shares(main)
    cls = classify(s)
    book = [classify(shares(t)) for t in book_main]
    common.update(share_HOLDQK=s["HOLDQK"], share_HOLDG=s["HOLDG"], share_sum=None if cls is None else s["HOLDQK"] + s["HOLDG"],
                  book_classes=book, books_matching=sum(1 for b in book if b == cls))
    if cls is None:
        return {"outcome": "U", "reason": "a held condition's tilt is undefined", **common}
    return {"outcome": cls, "reason": None, **common}
