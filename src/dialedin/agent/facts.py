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
    _c4_to_c7(sh, repo)
    _exploratory(sh, repo)
    return sh


def _c4_to_c7(sh, repo):
    p = _latest(repo, "c4v1-*/c4_summary.json")
    if p:
        r = json.load(open(p)); rel = os.path.relpath(p, repo); c = r["C4"]
        for lever, name in (("window", "local sliding-window size"), ("rope", "local RoPE base")):
            v = c[lever]
            sh.add(f"C4 {name} lever outcome (sealed rules: D1 a dial, D0 no dial)", v["outcome"], f"{rel}#C4.{lever}.outcome")
            sh.add(f"C4 {name}: tilt at the low, base and high setting", ", ".join(f"{v['tilts'][k]:.4f}" for k in ("low", "base", "high")),
                   f"{rel}#C4.{lever}.tilts")
            sh.add(f"C4 {name}: change in tilt from low to high", v["delta"], f"{rel}#C4.{lever}.delta")
            sh.add(f"C4 {name}: change needed for a dial", v["needed"], f"{rel}#C4.{lever}.needed")
            sh.add(f"C4 {name}: worst perplexity ratio", max(v["ppl_ratio_low"], v["ppl_ratio_high"]), f"{rel}#C4.{lever}")
        t = c["text"]
        sh.add("C4 text control outcome (A: the tilt belongs mostly to the model)", t["outcome"], f"{rel}#C4.text.outcome")
        sh.add("C4 tilt with shuffled tokens, as a fraction of the book's tilt", t["shuffled_over_base"], f"{rel}#C4.text.shuffled_over_base")
        sh.add("C4 tilt with one repeated token, as a fraction of the book's tilt", t["constant_over_base"], f"{rel}#C4.text.constant_over_base")
    p = _latest(repo, "c5v1-*/c5_summary.json")
    if p:
        r = json.load(open(p)); rel = os.path.relpath(p, repo); c = r["C5"]
        sh.add("C5 trained-dial outcome (sealed rules: L1 a dial, L2 a dial with damage, L0 no dial; needs a 25 percent flattening at scale 2)",
               c["outcome"], f"{rel}#C5.outcome")
        sh.add("C5 adapter: 32 low-rank modules in layers 2 to 5, rank", 8, "criteria/sealed/c5-glimmer-trained-dial-v1.yaml#training.rank")
        for k in ("S0", "S1", "S2"):
            sh.add(f"C5 tilt on set A at dial scale {k[1:]}", c["tilts"][k], f"{rel}#C5.tilts.{k}")
        sh.add("C5 relative tilt change at scale 2", c["relative_change"], f"{rel}#C5.relative_change")
        lo, hi = r["bootstrap_95"]["relative_change_S2"]
        sh.add("C5 relative tilt change at scale 2, 95 percent interval low", lo, f"{rel}#bootstrap_95.relative_change_S2")
        sh.add("C5 relative tilt change at scale 2, 95 percent interval high", hi, f"{rel}#bootstrap_95.relative_change_S2")
        sh.add("C5 Spearman of tilt against scale", c["spearman"], f"{rel}#C5.spearman")
        sh.add("C5 books agreeing (of 5)", c["books_agree"], f"{rel}#C5.books_agree")
        sh.add("C5 worst perplexity ratio over scales", c["worst_ppl_ratio"], f"{rel}#C5.worst_ppl_ratio")
        g = r["diagnostic_groups"]["unadapted_layers"]["relative_change_S2"]
        sh.add("C5 relative tilt change at scale 2 over layers 6 to 51 only (no adapter there)", g, f"{rel}#diagnostic_groups.unadapted_layers")
        sh.add("C5 scale 0 against C4 base: largest relative difference", r["diagnostic_S0_vs_C4_BASE"]["max_relative_difference"],
               f"{rel}#diagnostic_S0_vs_C4_BASE")
        b = r["behavior"]
        sh.add("C5 exploratory recall: passkey trials correct at scale 2 (of 20)", b["recall"]["S2"]["all_top1"], f"{rel}#behavior.recall.S2")
        sh.add("C5 exploratory topic-switch excess loss at scale 0 (nats)", b["switch"]["S0"]["excess_total"], f"{rel}#behavior.switch.S0")
        sh.add("C5 exploratory topic-switch excess loss at scale 2 (nats)", b["switch"]["S2"]["excess_total"], f"{rel}#behavior.switch.S2")
    p = _latest(repo, "c6v*-*/c6_summary.json")
    if p:
        r = json.load(open(p)); rel = os.path.relpath(p, repo); c = r["C6"]
        sh.add("C6 topic-switch outcome (sealed rules: B1 confirmed needs a 5 percent rise; B0 not confirmed)", c["outcome"], f"{rel}#C6.outcome")
        sh.add("C6 relative rise in topic-switch excess loss at scale 2 (10 fresh books, 90 pairs)", c["relative_rise"], f"{rel}#C6.relative_rise")
        lo, hi = r["bootstrap_95_relative_rise"]
        sh.add("C6 relative rise, 95 percent interval low", lo, f"{rel}#bootstrap_95_relative_rise")
        sh.add("C6 relative rise, 95 percent interval high", hi, f"{rel}#bootstrap_95_relative_rise")
        sh.add("C6 target books rising (of 10)", c["books_agree"], f"{rel}#C6.books_agree")
        sh.add("C6 random adapter of matched size: relative rise at scale 2", r["control_random_adapter"]["relative_rise"],
               f"{rel}#control_random_adapter.relative_rise")
        sh.add("C6 reproduction of C5's pairs: largest relative difference", r["reproduction_of_c5"]["max_relative_difference"],
               f"{rel}#reproduction_of_c5")
        sh.add("C6 perplexity ratio at scale 2", r["secondary"]["clean_perplexity_ratio"]["S2"], f"{rel}#secondary.clean_perplexity_ratio.S2")
    p = _latest(repo, "c7v1-*/c7_summary.json")
    if p:
        r = json.load(open(p)); rel = os.path.relpath(p, repo); c = r["C7"]
        sh.add("C7 localization outcome (sealed rules: M1 localized needs layer 2 q, k, v alone at 60 to 140 percent of the full "
               "dial's tilt change, the other 29 modules within 30 percent, 4 of 5 books)", c["outcome"], f"{rel}#C7.outcome")
        sh.add("C7 full dial: relative tilt change on set B at scale 2", c["full_change"], f"{rel}#C7.full_change")
        sh.add("C7 share of the change from layer 2 q, k, v alone (3 of 32 modules)", c["share_QKV2"], f"{rel}#C7.share_QKV2")
        sh.add("C7 share of the change from the other 29 modules", c["share_REST"], f"{rel}#C7.share_REST")
        for k in ("share_QKV2", "share_REST"):
            lo, hi = r["bootstrap_95"][k]
            sh.add(f"C7 {k}, 95 percent interval low", lo, f"{rel}#bootstrap_95.{k}")
            sh.add(f"C7 {k}, 95 percent interval high", hi, f"{rel}#bootstrap_95.{k}")
        sh.add("C7 books agreeing (of 5)", c["books_agree_QKV2"], f"{rel}#C7.books_agree_QKV2")
        sh.add("C7 perplexity ratio with only layer 2 q, k, v on", r["perplexity_ratio_median"]["QKV2"], f"{rel}#perplexity_ratio_median.QKV2")
        sh.add("C7 relative tilt change over layer 2's own heads, layer 2 q, k, v only", r["diagnostic_groups"]["layer2"]["relative_change"]["QKV2"],
               f"{rel}#diagnostic_groups.layer2")
        sh.add("C7 base condition against C1: largest relative difference", r["reproduction_BASE_vs_C1v5"]["max_relative_difference"],
               f"{rel}#reproduction_BASE_vs_C1v5")


def _exploratory(sh, repo):
    """Unsealed probes. Every label starts with EXPLORATORY so a narration cannot present them as sealed results."""
    raw = lambda pat: (sorted(glob.glob(os.path.join(repo, "results", "raw", pat))) or [None])[-1]
    p = raw("probe-loc-2*/localize.json")
    if p:
        r = json.load(open(p))["conditions"]; rel = os.path.relpath(p, repo)
        for k, lab in (("L2_attention", "layer 2 attention only"), ("all_but_L2", "everything except layer 2"),
                       ("mlp_only", "MLP modules only")):
            sh.add(f"EXPLORATORY localization, set C at 8K tokens: share of the full tilt change from {lab}", r[k]["fraction_of_full"],
                   f"{rel}#conditions.{k}")
    p = raw("probe-loc-deep-*/localize.json")
    if p:
        r = json.load(open(p))["conditions"]; rel = os.path.relpath(p, repo)
        b, a = r["base"]["tilt"], r["L2_attention"]["tilt"]
        for k, lab in (("L2_qk_only", "q and k only"), ("L2_v_only", "v only"), ("L2_gate_only", "the output gate only"),
                       ("L2_o_only", "the output projection only")):
            sh.add(f"EXPLORATORY localization within layer 2 attention: share from {lab}", (r[k]["tilt"] - b) / (a - b),
                   f"{rel}#conditions.{k}")
    p = os.path.join(repo, "results", "derived", "headmap-c1v5-20261027T152040Z", "headmap.json")
    if os.path.exists(p):
        r = json.load(open(p)); rel = os.path.relpath(p, repo); s = r["summary"]
        sh.add("EXPLORATORY head map: heads resolved individually (of 1,600)", s["heads_resolved"], f"{rel}#summary.heads_resolved")
        c = s["corner_percentiles_10_25_50_75_90"]
        sh.add("EXPLORATORY head map: median listening range of a head (tokens)", c[2], f"{rel}#summary.corner_percentiles")
        sh.add("EXPLORATORY head map: 90th percentile listening range (tokens)", c[4], f"{rel}#summary.corner_percentiles")
        sh.add("EXPLORATORY head map: median head tilt, local layers", s["tilt_median_local_layers"], f"{rel}#summary")
        sh.add("EXPLORATORY head map: median head tilt, global layers", s["tilt_median_global_layers"], f"{rel}#summary")
        st = r["stability_across_sets"]
        rs = [v["corner_log"]["r"] for v in st.values()]
        sh.add("EXPLORATORY head map: lowest correlation of listening range between book sets", min(rs), f"{rel}#stability_across_sets")
        sh.add("EXPLORATORY head map: highest correlation of listening range between book sets", max(rs), f"{rel}#stability_across_sets")
    p = os.path.join(repo, "results", "derived", "headmap-attention", "compare.json")
    if os.path.exists(p):
        r = json.load(open(p))["correlations"]; rel = os.path.relpath(p, repo)
        sh.add("EXPLORATORY: Spearman of listening range against attention distance, all heads", r["corner_vs_mean_distance"]["spearman"],
               f"{rel}#correlations.corner_vs_mean_distance")
        sh.add("EXPLORATORY: the same, within layers", r["corner_vs_mean_distance"]["within_layer_r"], f"{rel}#correlations.corner_vs_mean_distance")
    p = raw("probe-bandmute-*/bandmute.json")
    if p:
        r = json.load(open(p))["summary"]; rel = os.path.relpath(p, repo); rc = r["relative_change"]
        sh.add("EXPLORATORY band mute: heads per band", r["k"], f"{rel}#summary.k")
        for band in ("short", "long"):
            sh.add(f"EXPLORATORY band mute: loss change, all tokens, muting the {band}-range heads", rc[band]["all"],
                   f"{rel}#summary.relative_change.{band}")
            ctl = [rc[n]["all"] for n in rc if n.startswith(f"control_{band}_")]
            sh.add(f"EXPLORATORY band mute: loss change, all tokens, largest of the {band} band's random controls", max(ctl),
                   f"{rel}#summary.relative_change")
            sh.add(f"EXPLORATORY band mute: loss change on far-repeat tokens, muting the {band}-range heads", rc[band]["far_repeat"],
                   f"{rel}#summary.relative_change.{band}.far_repeat")
