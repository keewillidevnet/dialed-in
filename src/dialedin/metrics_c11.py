"""C11 metrics: where does Glimmer's lean come from? Implements criteria/sealed/c11-glimmer-lean-path-v1.yaml.
Tilts come from metrics_c2.tilt. Pure Python.

Three conditions, all with the probe injected at the layer 1 output as in C1 v5:
  BASE   the model as it is.
  VONLY  the probe reaches the values only: in layers 2 to 51, q_proj, k_proj and the attention output gate
         (self_attn.gate_proj) receive the attention input of the undriven pass, so every attention pattern and
         every gate is the undriven one, and only v_proj (and every MLP) sees the probe.
  PONLY  the mirror: v_proj receives the undriven input; q_proj, k_proj and the gate see the probe.

The primary number is the value-path tilt ratio
    T_V = tilt(VONLY) / tilt(BASE)
over the main window (heads in layers 2 to 51). If fixed-pattern averaging of the values is where the lean comes
from, T_V is near 1.
"""
CONDITIONS = ("BASE", "VONLY", "PONLY")
GATED = ("BASE", "VONLY")           # Phase A gates and period validity use these two; PONLY is reported
V1_BAND = (0.85, 1.15)              # set-level T_V for V1: averaging through the undriven patterns carries the lean
BOOK_BAND = (0.70, 1.30)            # a book agrees with V1 if its own T_V lies here
MIN_BOOKS = 4                       # of 5
PARTIAL_MIN = 0.50                  # VH: 0.50 <= T_V < 0.85; V0: T_V < 0.50


def ratio(t: dict, cond: str):
    """tilt(cond) / tilt(BASE), or None if either is undefined or tilt(BASE) is not positive."""
    if t.get(cond) is None or t.get("BASE") is None or t["BASE"] <= 0:
        return None
    return t[cond] / t["BASE"]


def book_agrees(t: dict) -> bool:
    r = ratio(t, "VONLY")
    return r is not None and BOOK_BAND[0] <= r <= BOOK_BAND[1]


def path_outcome(main: dict, book_main: list) -> dict:
    """main: set-level tilts per condition over layers 2 to 51; book_main: per book, its main tilts.

    U:   tilt(BASE) or tilt(VONLY) undefined, or tilt(BASE) not positive.
    V1:  0.85 <= T_V <= 1.15 and at least 4 of 5 books have their own T_V within 0.70 to 1.30. The lean is
         carried by the values averaged through the undriven attention patterns and gates.
    V1S: T_V within 0.85 to 1.15 at set level, but fewer than 4 books agree (set level only; reported as such).
    VX:  T_V above 1.15: averaging alone leans more than the model; the pattern and gate path flattens it.
    VH:  0.50 <= T_V < 0.85: averaging carries part of the lean; the pattern and gate path adds the rest.
    V0:  T_V below 0.50: most of the lean needs the probe to reach the patterns or gates.
    T_P = tilt(PONLY) / tilt(BASE) is reported with every outcome, never gated.
    """
    common = {"tilts": dict(main), "T_P": ratio(main, "PONLY")}
    if main.get("BASE") is None or main.get("VONLY") is None:
        return {"outcome": "U", "reason": "tilt(BASE) or tilt(VONLY) is undefined", "T_V": None, **common}
    if main["BASE"] <= 0:
        return {"outcome": "U", "reason": "tilt(BASE) is not positive", "T_V": None, **common}
    tv = ratio(main, "VONLY")
    books = sum(1 for t in book_main if book_agrees(t))
    common.update(T_V=tv, books_agreeing=books, book_T_V=[ratio(t, "VONLY") for t in book_main],
                  book_T_P=[ratio(t, "PONLY") for t in book_main])
    lo, hi = V1_BAND
    if lo <= tv <= hi:
        if books >= MIN_BOOKS:
            return {"outcome": "V1", "reason": None, **common}
        return {"outcome": "V1S", "reason": f"{books} of 5 books have T_V within {BOOK_BAND[0]} to {BOOK_BAND[1]}, "
                                           f"fewer than {MIN_BOOKS}", **common}
    if tv > hi:
        return {"outcome": "VX", "reason": f"T_V {tv:.3f} above {hi}", **common}
    if tv >= PARTIAL_MIN:
        return {"outcome": "VH", "reason": f"T_V {tv:.3f} within {PARTIAL_MIN} to {lo}", **common}
    return {"outcome": "V0", "reason": f"T_V {tv:.3f} below {PARTIAL_MIN}", **common}
