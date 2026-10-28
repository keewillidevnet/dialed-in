"""C3 metrics: the head-dial outcome. Implements criteria/sealed/c3-glimmer-head-dial-v1.yaml.

The tilt itself is metrics_c2.tilt (sealed with C2), used by analyze_c3. Pure Python.
"""

EFFECT_FRACTION = 0.25              # |tilt(S2.0) - tilt(S0.0)| must reach this fraction of |tilt(S1.0)|
SPECIFICITY = 2.0                   # the selected-head change must be at least this many times the control-head change
MIN_SIZE_RATIO = 0.5                # specificity is never claimed if the control is under half the selected heads' size
MAX_LAYER_GAP = 8                   # nor if the selected and control heads' per-layer counts differ by more than 8 in total
S_LEVELS = ("S0.0", "S0.5", "S1.0", "S1.5", "S2.0")


def classify_c3(tilts: dict, size_ratio: float, layer_gap: int) -> dict:
    """tilts: {set: {condition: tilt or None}} for S0.0, S0.5, S1.0, S1.5, S2.0, R0.0, R2.0 (R = size-matched control).
    size_ratio: summed broadband size of the control heads over that of the selected heads (sealed in the spec).
    layer_gap: sum over layers 2 to 5 of the difference in selected and control head counts (sealed in the spec).

    U  inconclusive: any tilt is None.
    Direction: "+" if on every set the tilt rises strictly through S0.0, S0.5, S1.0, S1.5, S2.0; "-" if it falls strictly.
    Effect: on every set |tilt(S2.0) - tilt(S0.0)| >= EFFECT_FRACTION * |tilt(S1.0)|.
    Specificity: on every set |tilt(S2.0) - tilt(S0.0)| >= SPECIFICITY * |tilt(R2.0) - tilt(R0.0)| / min(size_ratio, 1),
    so a smaller control must be beaten by a proportionally larger margin; and size_ratio >= MIN_SIZE_RATIO and
    layer_gap <= MAX_LAYER_GAP, otherwise a difference could come from size or layer rather than head identity.
    K1: a direction, the effect and specificity. K2: a direction and the effect only. K0: otherwise.
    The pre-registered prediction is direction "+".
    """
    sets = sorted(tilts)
    per = {}
    for s in sets:
        t = tilts[s]
        info = {"tilts": dict(t)}
        if any(v is None for v in t.values()):
            info["defined"] = False
        else:
            seq = [t[c] for c in S_LEVELS]
            ds = t["S2.0"] - t["S0.0"]
            dr = t["R2.0"] - t["R0.0"]
            info.update(defined=True, delta_selected=ds, delta_random=dr,
                        order="+" if all(a < b for a, b in zip(seq, seq[1:])) else
                              "-" if all(a > b for a, b in zip(seq, seq[1:])) else None,
                        effect=bool(abs(ds) >= EFFECT_FRACTION * abs(t["S1.0"])),
                        specific=bool(abs(ds) >= SPECIFICITY * abs(dr) / min(size_ratio, 1.0)))
        per[s] = info
    if not all(per[s]["defined"] for s in sets):
        return {"outcome": "U", "direction": None, "prediction_held": None, "per_set": per,
                "reason": "a tilt is undefined (too few valid periods) in some set and condition"}
    orders = {per[s]["order"] for s in sets}
    direction = orders.pop() if len(orders) == 1 and None not in orders else None
    effect = all(per[s]["effect"] for s in sets)
    size_matched = bool(size_ratio >= MIN_SIZE_RATIO and layer_gap <= MAX_LAYER_GAP)
    specific = all(per[s]["specific"] for s in sets) and size_matched
    outcome = "K1" if direction and effect and specific else "K2" if direction and effect else "K0"
    return {"outcome": outcome, "direction": direction,
            "prediction_held": (direction == "+") if outcome in ("K1", "K2") else None,
            "effect_all_sets": effect, "specific_all_sets": specific, "size_ratio": size_ratio, "layer_gap": layer_gap,
            "size_matched": size_matched, "per_set": per}


def book_signs(book_tilts: list) -> dict:
    """Descriptive: of the books with defined tilts, how many have tilt(S2.0) > tilt(S0.0), and the same for R."""
    ok = [b for b in book_tilts if all(v is not None for v in b.values())]
    return {"books": len(ok),
            "selected_up_steeper": int(sum(b["S2.0"] > b["S0.0"] for b in ok)),
            "random_up_steeper": int(sum(b["R2.0"] > b["R0.0"] for b in ok))}
