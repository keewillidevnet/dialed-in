"""C9 metrics: is layer 2's v_proj necessary inside the full C5 dial, and is that specific to it?
Implements criteria/sealed/c9-glimmer-necessity-v1.yaml. Tilts come from metrics_c2.tilt. Pure Python.

A condition's retained share is the fraction of the full dial's tilt change it keeps:
    retained(X) = (tilt(X) - tilt(BASE)) / (tilt(FULL) - tilt(BASE))
computed over two read windows: the main tilt (heads in layers 2 to 51) and the downstream tilt (heads in layers 5 to
51, which both layer 2's and layer 4's modules can reach).
"""
CONDITIONS = ("BASE", "FULL", "V2", "NOV2", "NOV4", "NOG2")
MIN_FULL_CHANGE = 0.15          # FULL must flatten the main tilt by at least 15 percent (as C8; Muse Wonder's floor)
V2_GATE_MAIN = (0.40, 1.40)     # replication gate: layer 2 v alone, main window (Muse Wonder's revised proposal)
V2_GATE_DOWN = (0.20, 1.40)     # the same, downstream window
MAX_RETAIN_NOV2 = 0.60          # main: without layer 2 v the dial keeps at most 60 percent (loses at least 40)
MIN_RETAIN_NOG2 = 0.80          # main: without layer 2's output gate (same layer, reaches layer 2's heads) it keeps at least 80 percent
MIN_BOOKS = 4                   # of 5 books, at least 4 keep less without layer 2 v than without layer 2 gate (main)
MIN_DOWN_CHANGE = 0.025         # downstream is judged only if FULL flattens layers 5 to 51 by at least 2.5 percent
MAX_RETAIN5_NOV2 = 0.70         # downstream: without layer 2 v the dial keeps at most 70 percent
MIN_RETAIN5_NOV4 = 0.80         # downstream: without layer 4 v (same type, control) it keeps at least 80 percent


def retained(tilts: dict) -> dict:
    """Each non-BASE, non-FULL condition's tilt change from BASE as a fraction of FULL's change. None if undefined."""
    if any(tilts.get(c) is None for c in CONDITIONS) or tilts["FULL"] == tilts["BASE"]:
        return {c: None for c in CONDITIONS[2:]}
    full = tilts["FULL"] - tilts["BASE"]
    return {c: (tilts[c] - tilts["BASE"]) / full for c in CONDITIONS[2:]}


def book_orders(t: dict) -> bool:
    """A book counts if FULL flattens it and it keeps less of the change without layer 2 v than without layer 2's gate."""
    if any(t.get(c) is None for c in ("BASE", "FULL", "NOV2", "NOG2")):
        return False
    df = t["FULL"] - t["BASE"]
    if not df < 0:
        return False
    return bool((t["NOV2"] - t["BASE"]) / df < (t["NOG2"] - t["BASE"]) / df)


def downstream_part(down: dict) -> dict:
    """Judged only if every downstream tilt is defined, tilt(BASE) is positive, FULL flattens layers 5 to 51 by at
    least MIN_DOWN_CHANGE, and V2 alone passes its downstream replication gate."""
    if down is None or any(down.get(c) is None for c in CONDITIONS) or down["BASE"] <= 0:
        return {"judged": False, "passes": False, "why": "a downstream tilt is undefined or tilt(BASE) is not positive"}
    change = (down["FULL"] - down["BASE"]) / down["BASE"]
    r = retained(down)
    out = {"full_change": change, **{f"retained_{c}": r[c] for c in CONDITIONS[2:]}}
    if change > -MIN_DOWN_CHANGE:
        return {**out, "judged": False, "passes": False,
                "why": f"FULL flattens layers 5 to 51 by {-change:.3f}, less than {MIN_DOWN_CHANGE}"}
    lo, hi = V2_GATE_DOWN
    if not lo <= r["V2"] <= hi:
        return {**out, "judged": False, "passes": False,
                "why": f"V2 alone keeps {r['V2']:.3f} downstream, outside {lo} to {hi}"}
    fails = [w for w, bad in (
        (f"without layer 2 v it keeps {r['NOV2']:.3f}, above {MAX_RETAIN5_NOV2}", r["NOV2"] > MAX_RETAIN5_NOV2),
        (f"without layer 4 v it keeps {r['NOV4']:.3f}, below {MIN_RETAIN5_NOV4}", r["NOV4"] < MIN_RETAIN5_NOV4)) if bad]
    return {**out, "judged": True, "passes": not fails, "why": "; ".join(fails) or None}


def necessity_outcome(main: dict, down: dict, book_main: list) -> dict:
    """main, down: set-level tilts per condition over layers 2 to 51 and 5 to 51; book_main: per book, its main tilts.

    U: a main tilt undefined, tilt(BASE) not positive, or FULL flattens the main tilt by less than 15 percent.
    R0: V2 alone keeps a main share outside 0.40 to 1.40 (C8 does not replicate here); necessity is not judged.
    Main window passes: retained(NOV2) <= 0.60, retained(NOG2) >= 0.80, and at least 4 of 5 books order NOV2 below NOG2.
    Downstream passes (when judged, see downstream_part): retained5(NOV2) <= 0.70 and retained5(NOV4) >= 0.80.
    N1: both pass. N1L: the main window passes and the downstream does not (or is not judged). N0: the main window fails.
    """
    if any(main.get(c) is None for c in CONDITIONS):
        return {"outcome": "U", "reason": "a main tilt is undefined", "tilts": dict(main)}
    if main["BASE"] <= 0:
        return {"outcome": "U", "reason": "tilt(BASE) is not positive", "tilts": dict(main)}
    full_change = (main["FULL"] - main["BASE"]) / main["BASE"]
    if full_change > -MIN_FULL_CHANGE:
        return {"outcome": "U", "reason": "FULL does not flatten the main tilt by at least 15 percent",
                "full_change": full_change, "tilts": dict(main)}
    r = retained(main)
    books = sum(1 for t in book_main if book_orders(t))
    ds = downstream_part(down)
    common = {"full_change": full_change, **{f"retained_{c}": r[c] for c in CONDITIONS[2:]},
              "books_ordered": books, "downstream": ds,
              "gap_sufficiency_minus_necessity": r["V2"] - (1 - r["NOV2"]),   # V2 alone minus what removing it loses
              "tilts": dict(main), "tilts_downstream": dict(down) if down else None}
    lo, hi = V2_GATE_MAIN
    if not lo <= r["V2"] <= hi:
        return {"outcome": "R0", "reason": f"V2 alone keeps {r['V2']:.3f} of the main change, outside {lo} to {hi}: "
                                          f"C8 does not replicate here, so necessity is not judged",
                "main_passes": False, **common}
    fails = [w for w, bad in (
        (f"without layer 2 v the dial keeps {r['NOV2']:.3f}, above {MAX_RETAIN_NOV2}", r["NOV2"] > MAX_RETAIN_NOV2),
        (f"without layer 2 gate it keeps {r['NOG2']:.3f}, below {MIN_RETAIN_NOG2}", r["NOG2"] < MIN_RETAIN_NOG2),
        (f"{books} of 5 books order NOV2 below NOG2, fewer than {MIN_BOOKS}", books < MIN_BOOKS)) if bad]
    main_passes = not fails
    if not main_passes:
        outcome, reason = "N0", "; ".join(fails)
    elif ds["passes"]:
        outcome, reason = "N1", None
    else:
        outcome, reason = "N1L", f"downstream: {ds['why']}"
    return {"outcome": outcome, "reason": reason, "main_passes": main_passes, **common}
