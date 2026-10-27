"""C2 metrics: the tilt of the frequency response and the knob outcome. Implements
criteria/sealed/c2-glimmer-knob-tilt-v1.yaml.

metrics.py is locked by the C1 seals, so C2's metrics live here and import the shared ones from it.
Pure numpy, no model code.
"""
import numpy as np

MIN_VALID = 5                       # a tilt needs at least 5 of the 7 periods valid
EFFECT_FRACTION = 0.25              # |tilt(G1.5) - tilt(G0.5)| must reach this fraction of |tilt(G1.0)|
SPECIFICITY = 2.0                   # the global change must be at least this many times the local change


def tilt(grid, curve, valid) -> float:
    """Least-squares slope of ln(gain) against ln(N) over valid points: the exponent of a power-law fit.
    None if fewer than MIN_VALID points are valid or any valid gain is not positive."""
    g = np.asarray(grid, dtype=np.float64)
    c = np.asarray(curve, dtype=np.float64)
    v = np.asarray(valid, dtype=bool)
    if v.sum() < MIN_VALID or np.any(c[v] <= 0):
        return None
    x, y = np.log(g[v]), np.log(c[v])
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def classify_c2(tilts: dict) -> dict:
    """tilts: {set: {condition name: tilt or None}}, conditions G1.0, G0.5, G1.5, L0.5, L1.5.

    U  inconclusive: any tilt is None.
    Direction: on every set tilt(G0.5) < tilt(G1.0) < tilt(G1.5) is "+"; the reverse order on every set is "-".
    Effect: on every set |tilt(G1.5) - tilt(G0.5)| >= EFFECT_FRACTION * |tilt(G1.0)|.
    Specificity: on every set |tilt(G1.5) - tilt(G0.5)| >= SPECIFICITY * |tilt(L1.5) - tilt(L0.5)|.
    K1 tunable and specific: a direction, the effect and specificity.
    K2 tunable, not specific: a direction and the effect, not specificity.
    K0 not tunable: anything else with every tilt defined.
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
            dg = t["G1.5"] - t["G0.5"]
            dl = t["L1.5"] - t["L0.5"]
            info.update(defined=True, delta_global=dg, delta_local=dl,
                        order="+" if t["G0.5"] < t["G1.0"] < t["G1.5"] else
                              "-" if t["G0.5"] > t["G1.0"] > t["G1.5"] else None,
                        effect=bool(abs(dg) >= EFFECT_FRACTION * abs(t["G1.0"])),
                        specific=bool(abs(dg) >= SPECIFICITY * abs(dl)))
        per[s] = info
    if not all(per[s]["defined"] for s in sets):
        return {"outcome": "U", "direction": None, "prediction_held": None, "per_set": per,
                "reason": "a tilt is undefined (too few valid periods) in some set and condition"}
    orders = {per[s]["order"] for s in sets}
    direction = orders.pop() if len(orders) == 1 and None not in orders else None
    effect = all(per[s]["effect"] for s in sets)
    specific = all(per[s]["specific"] for s in sets)
    if direction and effect and specific:
        outcome = "K1"
    elif direction and effect:
        outcome = "K2"
    else:
        outcome = "K0"
    return {"outcome": outcome, "direction": direction,
            "prediction_held": (direction == "+") if outcome in ("K1", "K2") else None,
            "effect_all_sets": effect, "specific_all_sets": specific, "per_set": per}


def book_signs(book_tilts: list) -> dict:
    """Descriptive: of the books with defined tilts, how many have tilt(G1.5) > tilt(G0.5), and the same for L."""
    ok = [b for b in book_tilts if all(v is not None for v in b.values())]
    return {"books": len(ok),
            "global_up_steeper": int(sum(b["G1.5"] > b["G0.5"] for b in ok)),
            "local_up_steeper": int(sum(b["L1.5"] > b["L0.5"] for b in ok))}
