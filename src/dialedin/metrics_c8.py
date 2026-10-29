"""C8 metrics: does one matrix, layer 2's v_proj, carry a majority of the C5 dial's tilt change, specifically?
Implements criteria/sealed/c8-glimmer-one-matrix-dial-v1.yaml. Tilts come from metrics_c2.tilt. Pure Python.
"""
CONDITIONS = ("BASE", "FULL", "QKV2", "V2", "QK2", "V4")
MIN_FULL_CHANGE = 0.15          # FULL must flatten the tilt by at least 15 percent (Muse Wonder's floor)
MIN_SHARE_V2 = 0.50             # V2's change must be at least half of FULL's: a majority from 1 of 32 matrices ...
MAX_SHARE_V2 = 1.40             # ... without overshooting FULL by more than 40 percent
MAX_ABS_SHARE_V4 = 0.30         # the matched control, layer 4's v, must stay within 30 percent of FULL in either direction
MIN_BOOKS = 4                   # of the 5 books, at least 4 must agree (book rule below)
BOOK_MIN_SHARE = 0.30           # a book agrees if FULL and V2 both flatten it and V2's change is >= 30 percent of FULL's


def shares(tilts: dict) -> dict:
    """Each masked condition's tilt change from BASE as a fraction of FULL's change. None if undefined."""
    if any(tilts.get(c) is None for c in CONDITIONS) or tilts["FULL"] == tilts["BASE"]:
        return {c: None for c in CONDITIONS[2:]}
    full = tilts["FULL"] - tilts["BASE"]
    return {c: (tilts[c] - tilts["BASE"]) / full for c in CONDITIONS[2:]}


def book_agrees(t: dict) -> bool:
    if any(t.get(c) is None for c in ("BASE", "FULL", "V2")):
        return False
    dv, df = t["V2"] - t["BASE"], t["FULL"] - t["BASE"]
    return bool(dv < 0 and df < 0 and dv / df >= BOOK_MIN_SHARE)


def one_matrix_outcome(tilts: dict, book_tilts: list) -> dict:
    """tilts: set-level tilt per condition; book_tilts: per book, its tilt per condition (None where undefined).

    U: a tilt undefined, tilt(BASE) not positive, or FULL does not flatten by at least 15 percent.
    sufficient: 0.50 <= share(V2) <= 1.40 and at least 4 of 5 books agree.
    specific: |share(V4)| <= 0.30.
    V1 one-matrix dial: sufficient and specific. V2S sufficient, not specific. V0 not sufficient.
    """
    if any(tilts.get(c) is None for c in CONDITIONS):
        return {"outcome": "U", "reason": "a tilt is undefined", "tilts": dict(tilts)}
    if tilts["BASE"] <= 0:
        return {"outcome": "U", "reason": "tilt(BASE) is not positive", "tilts": dict(tilts)}
    full_change = (tilts["FULL"] - tilts["BASE"]) / tilts["BASE"]
    if full_change > -MIN_FULL_CHANGE:
        return {"outcome": "U", "reason": "FULL does not flatten the tilt by at least 15 percent",
                "full_change": full_change, "tilts": dict(tilts)}
    sh = shares(tilts)
    books = sum(1 for t in book_tilts if book_agrees(t))
    in_range = MIN_SHARE_V2 <= sh["V2"] <= MAX_SHARE_V2
    sufficient = in_range and books >= MIN_BOOKS
    specific = abs(sh["V4"]) <= MAX_ABS_SHARE_V4
    outcome = ("V1" if specific else "V2S") if sufficient else "V0"
    reason = None
    if not sufficient:
        reason = "; ".join(r for r, bad in (
            (f"share(V2) {sh['V2']:.3f} outside [{MIN_SHARE_V2}, {MAX_SHARE_V2}]", not in_range),
            (f"{books} of 5 books agree, fewer than {MIN_BOOKS}", books < MIN_BOOKS)) if bad)
    elif not specific:
        reason = f"|share(V4)| {abs(sh['V4']):.3f} above {MAX_ABS_SHARE_V4}"
    return {"outcome": outcome, "reason": reason, "full_change": full_change,
            **{f"share_{c}": sh[c] for c in CONDITIONS[2:]}, "share_V2_in_range": bool(in_range),
            "books_agree_V2": books, "sufficient": bool(sufficient), "specific": bool(specific),
            "share_V2_of_QKV2": sh["V2"] / sh["QKV2"] if sh["QKV2"] else None,
            "additivity_V2_plus_QK2_over_QKV2": (sh["V2"] + sh["QK2"]) / sh["QKV2"] if sh["QKV2"] else None,
            "tilts": dict(tilts)}
