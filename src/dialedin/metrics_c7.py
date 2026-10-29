"""C7 metrics: is the C5 dial's tilt change carried by layer 2's q, k and v? Implements
criteria/sealed/c7-glimmer-dial-localization-v1.yaml. Tilts come from metrics_c2.tilt. Pure Python.
"""
CONDITIONS = ("BASE", "FULL", "QKV2", "REST")
MIN_FULL_CHANGE = 0.10          # FULL must flatten the tilt by at least 10 percent, or there is no effect to split
MIN_SHARE_QKV2 = 0.60           # QKV2's change must reach 60 percent of FULL's ...
MAX_SHARE_QKV2 = 1.40           # ... and not overshoot it by more than 40 percent (then the parts interact strongly)
MAX_ABS_SHARE_REST = 0.30       # for M1, REST's change must stay within 30 percent of FULL's, in either direction
MIN_BOOKS_QKV2 = 4              # of the 5 books, at least 4 must agree (book rule below)
BOOK_MIN_SHARE = 0.30           # a book agrees if FULL and QKV2 both flatten it and QKV2's change is >= 30 percent of FULL's


def shares(tilts: dict) -> dict:
    """Each condition's tilt change from BASE as a fraction of FULL's change. None if undefined."""
    if any(tilts.get(c) is None for c in CONDITIONS) or tilts["FULL"] == tilts["BASE"]:
        return {c: None for c in ("QKV2", "REST")}
    full = tilts["FULL"] - tilts["BASE"]
    return {c: (tilts[c] - tilts["BASE"]) / full for c in ("QKV2", "REST")}


def book_agrees(t: dict) -> bool:
    """A book agrees when its tilts are defined, FULL and QKV2 both flatten it, and QKV2's change is at least 30
    percent of FULL's in that book."""
    if any(t.get(c) is None for c in ("BASE", "FULL", "QKV2")):
        return False
    dq, df = t["QKV2"] - t["BASE"], t["FULL"] - t["BASE"]
    return bool(dq < 0 and df < 0 and dq / df >= BOOK_MIN_SHARE)


def localization_outcome(tilts: dict, book_tilts: list) -> dict:
    """tilts: set-level tilt per condition; book_tilts: per book, its tilt per condition (None where undefined).

    U: a tilt undefined, tilt(BASE) not positive, or FULL does not flatten by at least 10 percent (nothing to split).
    sufficient: 0.6 <= share(QKV2) <= 1.4 and at least 4 of 5 books agree (book_agrees).
    exclusive: |share(REST)| <= 0.3.
    M1 localized: sufficient and exclusive. M2 sufficient, not exclusive. M0 not localized: every tilt defined, FULL
    flattens enough, not sufficient.
    """
    if any(tilts.get(c) is None for c in CONDITIONS):
        return {"outcome": "U", "reason": "a tilt is undefined", "tilts": dict(tilts)}
    if tilts["BASE"] <= 0:
        return {"outcome": "U", "reason": "tilt(BASE) is not positive", "tilts": dict(tilts)}
    full_change = (tilts["FULL"] - tilts["BASE"]) / tilts["BASE"]
    if full_change > -MIN_FULL_CHANGE:
        return {"outcome": "U", "reason": "FULL does not flatten the tilt by at least 10 percent",
                "full_change": full_change, "tilts": dict(tilts)}
    sh = shares(tilts)
    books = sum(1 for t in book_tilts if book_agrees(t))
    in_range = MIN_SHARE_QKV2 <= sh["QKV2"] <= MAX_SHARE_QKV2
    sufficient = in_range and books >= MIN_BOOKS_QKV2
    exclusive = abs(sh["REST"]) <= MAX_ABS_SHARE_REST
    outcome = ("M1" if exclusive else "M2") if sufficient else "M0"
    reason = None
    if not sufficient:
        reason = "; ".join(r for r, bad in (
            (f"share(QKV2) {sh['QKV2']:.3f} outside [{MIN_SHARE_QKV2}, {MAX_SHARE_QKV2}]", not in_range),
            (f"{books} of 5 books agree, fewer than {MIN_BOOKS_QKV2}", books < MIN_BOOKS_QKV2)) if bad)
    elif not exclusive:
        reason = f"|share(REST)| {abs(sh['REST']):.3f} above {MAX_ABS_SHARE_REST}"
    return {"outcome": outcome, "reason": reason, "full_change": full_change, "share_QKV2": sh["QKV2"], "share_REST": sh["REST"],
            "share_QKV2_in_range": bool(in_range), "books_agree_QKV2": books, "sufficient": bool(sufficient),
            "exclusive": bool(exclusive), "additivity": sh["QKV2"] + sh["REST"], "tilts": dict(tilts)}
