"""The fact sheet: every number Muse may use, read from results/derived/*/c*_summary.json, each with its source.

Muse sees only this sheet, never raw files, so every number it writes can be checked against a cited fact.
"""
import glob
import json
import math
import os

from .. import metrics_c2


def _latest(repo, pattern):
    paths = sorted(glob.glob(os.path.join(repo, "results", "derived", pattern)))
    paths = [p for p in paths if "PARTIAL" not in p]
    return paths[-1] if paths else None


class Sheet:
    def __init__(self):
        self.facts = []

    def add(self, label, value, source):
        fid = f"F{len(self.facts) + 1}"
        self.facts.append({"id": fid, "label": label, "value": value, "source": source})
        return fid

    def text(self) -> str:
        out = []
        for f in self.facts:
            v = f["value"]
            v = f"{v:.4g}" if isinstance(v, float) else v
            out.append(f"[{f['id']}] {f['label']}: {v}")
        return "\n".join(out)

    def numbers(self) -> list:
        return [f["value"] for f in self.facts if isinstance(f["value"], (int, float)) and not isinstance(f["value"], bool)]


def build(repo: str = ".") -> Sheet:
    sh = Sheet()
    p = _latest(repo, "c1v5-*/c1_summary.json")
    if p:
        r = json.load(open(p)); rel = os.path.relpath(p, repo); c = r["C1"]
        sh.add("C1 outcome (sealed rules: R resonance, HP/LP cutoff, T tilt, F flat, S structured, U inconclusive)",
               c["outcome"], f"{rel}#C1.outcome")
        sh.add("C1 tilt direction", c.get("tilt_direction"), f"{rel}#C1.tilt_direction")
        sh.add("C1 books", r["books"], f"{rel}#books")
        grid = r["grid"]
        sh.add("C1 drive periods N, shortest", min(grid), f"{rel}#grid")
        sh.add("C1 drive periods N, longest", max(grid), f"{rel}#grid")
        for s, v in sorted(c["per_set"].items()):
            sh.add(f"C1 set {s}: valid periods of {len(grid)}", v["valid_points"], f"{rel}#C1.per_set.{s}.valid_points")
            sh.add(f"C1 set {s}: Spearman rho of gain vs N", v["rho_gain_vs_N"], f"{rel}#C1.per_set.{s}.rho_gain_vs_N")
            sh.add(f"C1 set {s}: gain at the shortest period, relative to the maximum", v["min_normalized"],
                   f"{rel}#C1.per_set.{s}.min_normalized")
            curve = r["curves"][s]["whole"]
            ok = [x is not None and math.isfinite(x) for x in curve]
            t = metrics_c2.tilt(grid, [x if o else 1.0 for x, o in zip(curve, ok)], ok)
            if t is not None:
                sh.add(f"C1 set {s}: tilt (slope of ln gain vs ln N, computed from the curve)", t, f"{rel}#curves.{s}.whole")
        for k, x in sorted(c["shape_replication"].items()):
            sh.add(f"C1 shape agreement between sets {k} (Spearman rho)", x["rho"], f"{rel}#C1.shape_replication.{k}.rho")
    for tag, pattern, key, d_keys, labels in (
            ("C2", "c2v1-*/c2_summary.json", "C2", ("delta_global", "delta_local"),
             ("global-layer knob, change in tilt from 0.5 to 1.5", "local-layer knob, change in tilt from 0.5 to 1.5")),
            ("C3", "c3v1-*/c3_summary.json", "C3", ("delta_selected", "delta_random"),
             ("selected heads, change in tilt from 0 to 2", "size-matched control heads, change in tilt from 0 to 2"))):
        p = _latest(repo, pattern)
        if not p:
            continue
        r = json.load(open(p)); rel = os.path.relpath(p, repo); c = r[key]
        sh.add(f"{tag} outcome (sealed rules: K1 tunable and specific, K2 tunable not specific, K0 not tunable, U inconclusive)",
               c["outcome"], f"{rel}#{key}.outcome")
        sh.add(f"{tag} direction", c.get("direction"), f"{rel}#{key}.direction")
        for s, v in sorted(c["per_set"].items()):
            for cond, t in sorted(v["tilts"].items()):
                sh.add(f"{tag} set {s}: tilt under {cond}", t, f"{rel}#{key}.per_set.{s}.tilts.{cond}")
            for dk, lab in zip(d_keys, labels):
                if dk in v:
                    sh.add(f"{tag} set {s}: {lab}", v[dk], f"{rel}#{key}.per_set.{s}.{dk}")
        for bk, bv in r.get("book_signs", {}).items():
            sh.add(f"{tag} books: {bk}", bv, f"{rel}#book_signs.{bk}")
        for diag in ("diagnostic_G1_vs_C1v5", "diagnostic_S1_vs_C1v5"):
            if diag in r and r[diag].get("status") == "compared":
                sh.add(f"{tag} reproduction of C1: cells compared", r[diag]["cells_compared"], f"{rel}#{diag}.cells_compared")
                sh.add(f"{tag} reproduction of C1: largest relative difference", r[diag]["max_relative_difference"],
                       f"{rel}#{diag}.max_relative_difference")
    return sh
