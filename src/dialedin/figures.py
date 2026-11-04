"""The README's figures, drawn from the result files. No GPU.

    PYTHONPATH=src python3 -m dialedin.figures            # writes every figure under docs/figures/

Every number drawn is read here from a summary file under results/derived/, named in each figure's footer.
"""
import argparse
import json
import os

C5 = "results/derived/c5v1-20261029T025343Z/c5_summary.json"
C7 = "results/derived/c7v1-20261029T154201Z/c7_summary.json"
C8 = "results/derived/c8v1-20261029T222946Z/c8_summary.json"
C9 = "results/derived/c9v1-20261030T023935Z/c9_summary.json"
C1 = "results/derived/c1v5-20261027T152040Z/c1_summary.json"
HEADMAP = "results/derived/headmap-c1v5-20261027T152040Z/headmap.json"
BEHAVIOR = "results/raw/probe-behavior-20261030T022546Z/behavior.json"
TEAL = "#2e9e7a"
ORANGE, BLUE, GRAY, INK, MUTED, RULE = "#c2610c", "#2a6fb0", "#9aa3ab", "#15202a", "#56636f", "#e3e7ea"


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": RULE,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})
    return plt


def the_dial(repo, out):
    plt = setup()
    d = json.load(open(os.path.join(repo, C5)))
    grid = d["grid"]
    norm = {k: [g / d["curves"][k][-1] * 100 for g in d["curves"][k]] for k in d["curves"]}
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
    ax.fill_between(grid, norm["S0"], norm["S2"], color=ORANGE, alpha=0.12, linewidth=0)
    ax.plot(grid, norm["S0"], color=BLUE, lw=2, ls=(0, (5, 3)), marker="o", ms=6, label="dial at 0× (the original model)")
    ax.plot(grid, norm["S2"], color=ORANGE, lw=2.6, marker="o", ms=7, label="dial at 2×")
    ax.set_xscale("log", base=2)
    ax.set_xticks(grid, [str(g) for g in grid])
    ax.set_xlabel("how slow the test wave is (tokens per cycle; fast on the left, slow on the right)")
    ax.set_ylabel("how strongly it gets through\n(% of the slowest wave)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", color=RULE, lw=1)
    ax.set_ylim(min(norm["S0"]) - 8, 106)
    for k, col, dy in (("S0", BLUE, -14), ("S2", ORANGE, 10)):
        ax.annotate(f"{norm[k][0]:.0f}%", (grid[0], norm[k][0]), xytext=(10, dy), textcoords="offset points",
                    color=INK, fontsize=12, fontweight="bold")
    change = d["C5"]["relative_change"]
    ppl = d["C5"]["ppl_ratios"]["S2"]
    fig.suptitle(f"Turn the dial up and the fast waves get through more: the lean flattens by {abs(change) * 100:.0f}%",
                 x=0.06, ha="left", fontsize=15, fontweight="bold", color=INK)
    ax.set_title(f"Muse Glimmer-30B, 5 books, 32,768 tokens each. Cost: perplexity ×{ppl:.3f}.",
                 loc="left", fontsize=11, color=MUTED)
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    fig.text(0.06, 0.01, f"Source: {C5} (sealed experiment C5)", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out)
    plt.close(fig)


def where(repo, out):
    """C7 to C9: about half of the dial's effect sits in one matrix, layer 2's v."""
    plt = setup()
    c7, c8, c9 = (json.load(open(os.path.join(repo, p))) for p in (C7, C8, C9))
    on = [("Layer 2 q, k, v (3 of 32 parts)", "C7, set B", c7["C7"]["share_QKV2"], c7["bootstrap_95"]["share_QKV2"], GRAY),
          ("Layer 2 v alone (1 of 32)", "C8, set C", c8["C8"]["share_V2"], c8["bootstrap_95"]["share_V2"], ORANGE),
          ("Layer 2 v alone (1 of 32)", "C9, set A", c9["C9"]["retained_V2"], c9["bootstrap_95"]["retained_V2"], ORANGE),
          ("Layer 4 v alone, control", "C8, set C", c8["C8"]["share_V4"], c8["bootstrap_95"]["share_V4"], BLUE)]
    off = [("All but layer 2 v", "C9, set A", c9["C9"]["retained_NOV2"], c9["bootstrap_95"]["retained_NOV2"], ORANGE),
           ("All but layer 2's gate, control", "C9, set A", c9["C9"]["retained_NOG2"], c9["bootstrap_95"]["retained_NOG2"], BLUE),
           ("All but layer 4 v, control", "C9, set A", c9["C9"]["retained_NOV4"], c9["bootstrap_95"]["retained_NOV4"], BLUE)]
    panels = [("Turn on only this part: how much of the effect do you get?", on, 1, "On its own: about half"),
              ("Turn off only this part: how much of the effect is left?", off, 0, "Without it: about half left")]
    fig = plt.figure(figsize=(12, 8.6), dpi=160)
    left = 0.045                                     # every heading starts on this edge
    fig.text(left, 0.935, "One matrix carries about half the dial", fontsize=24, fontweight="bold", color=INK, va="top")
    fig.text(left, 0.875, "Layer 2's value matrix (v) is 1 of the dial's 32 parts.  100% = the full dial (all 32 parts at 2×).",
             fontsize=12.5, color=MUTED, va="top")
    for (head, rows, key, note), top in zip(panels, (0.77, 0.375)):
        fig.text(left, top + 0.012, head, fontsize=14, fontweight="bold", color=INK, va="bottom")
        h = 0.068 * len(rows)
        ax = fig.add_axes([0.335, top - h - 0.01, 0.475, h])
        tr = ax.get_yaxis_transform()
        ys = list(range(len(rows)))[::-1]
        for y, (lab, tag, v, ci, col) in zip(ys, rows):
            ax.barh(y, max(v * 100, 0.4), color=col, height=0.62)
            ax.plot([ci[0] * 100, ci[1] * 100], [y, y], color=INK, lw=1.6, solid_capstyle="butt")
            ax.text(max(v, ci[1]) * 100 + 1.6, y, f"{v * 100:.0f}%", va="center", fontsize=13, fontweight="bold", color=INK)
            ax.text(-0.025, y, lab, transform=tr, ha="right", va="center", fontsize=12, color=INK)
            ax.text(-0.025, y - 0.36, tag, transform=tr, ha="right", va="center", fontsize=9, color=MUTED,
                    family="DejaVu Sans Mono")
        ax.text(1.0, ys[key], note, va="center", fontsize=11, fontweight="bold", color="white")   # inside the key bar
        ax.axvline(100, color=MUTED, lw=1, ls=":")
        ax.set_xlim(-3, 108)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks([])
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.grid(axis="x", color=RULE, lw=1)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", length=0, pad=6)
    fig.text(left, 0.055, "Orange: layer 2 v. Blue: controls. Gray: layer 2's q, k and v together.  "
             "Black lines: 95% intervals over 5 books; each experiment used its own 5 books.", fontsize=10, color=MUTED, va="bottom")
    fig.text(left, 0.012, f"Sources: {C7}, {C8},\n         {C9}", fontsize=8, color=MUTED, va="bottom",
             family="DejaVu Sans Mono", linespacing=1.5)
    fig.savefig(out)
    plt.close(fig)


def fingerprint(repo, out):
    """C1: the frequency response of the whole model, its sliding-window layers and its global layers, per book set."""
    plt = setup()
    import numpy as np
    d = json.load(open(os.path.join(repo, C1)))
    grid = d["grid"]
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=160)
    ends = {}
    for k, col, lw, lab in (("local", BLUE, 2.2, "sliding-window layers (39 of 52)"),
                            ("global", TEAL, 2.2, "global layers (13 of 52)")):
        for i, st in enumerate("ABC"):
            c = np.array([np.nan if x is None else x for x in d["curves"][st][k]], dtype=float)
            y = c / c[-1] * 100
            ax.plot(grid, y, color=col, lw=lw, alpha=0.9, label=lab if i == 0 else None)
            ends.setdefault(k, []).append(y[0])
    ax.set_xscale("log", base=2)
    ticks = [2, 8, 32, 128, 512, 2048, 4096]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylim(20, 108)
    ax.grid(axis="y", color=RULE, lw=1)
    ax.set_xlabel("how slow the test wave is (tokens per cycle; fast on the left, slow on the right)")
    ax.set_ylabel("how strongly it gets through\n(% of the slowest wave)")
    for k, col, dy in (("local", BLUE, -18), ("global", TEAL, 14)):
        ax.annotate(f"{min(ends[k]):.0f} to {max(ends[k]):.0f}%", (grid[0], sum(ends[k]) / 3), xytext=(8, dy),
                    textcoords="offset points", color=INK, fontsize=12, fontweight="bold")
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    fig.suptitle("The model's fingerprint: fast waves fade in the sliding-window layers,\nmuch less in the global layers",
                 x=0.06, ha="left", fontsize=15, fontweight="bold", color=INK)
    ax.set_title("Muse Glimmer-30B, three sets of 5 books (one line each; they nearly overlap). No dial.",
                 loc="left", fontsize=11, color=MUTED)
    fig.text(0.06, 0.01, f"Source: {C1} (sealed experiment C1, curves per set)", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out)
    plt.close(fig)


def heads(repo, out):
    """Exploratory head map: each resolved head's listening range, by layer."""
    plt = setup()
    import numpy as np
    h = json.load(open(os.path.join(repo, HEADMAP)))
    layers = h["per_head"]["layers"]
    glob = {3 + 4 * i for i in range(13)}
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
    shown = [l for l, row in zip(layers, h["per_head"]["corner_tokens"]) if sum(v is not None for v in row) >= 3]
    pos = {l: i for i, l in enumerate(shown)}
    for l, row in zip(layers, h["per_head"]["corner_tokens"]):
        v = np.array([x for x in row if x is not None], dtype=float)
        if l not in shown:
            continue
        col = TEAL if l in glob else BLUE
        x = pos[l]
        ax.scatter(x + rng.uniform(-0.28, 0.28, len(v)), v, s=22, color=col, alpha=0.75, edgecolors="white", linewidths=0.4)
        ax.plot([x - 0.35, x + 0.35], [np.median(v)] * 2, color=INK, lw=2)
    ax.set_yscale("log", base=2)
    yt = [2, 8, 32, 128, 512, 2048]
    ax.set_yticks(yt, [str(t) for t in yt])
    ax.set_xticks(range(len(shown)), [str(l) for l in shown], fontsize=9)
    ax.grid(axis="y", color=RULE, lw=1)
    ax.set_xlabel("layer (only layers with at least 3 heads above the noise floor)")
    ax.set_ylabel("listening range of a head\n(tokens)")
    ax.scatter([], [], color=BLUE, s=30, label="head in a sliding-window layer")
    ax.scatter([], [], color=TEAL, s=30, label="head in a global layer")
    ax.plot([], [], color=INK, lw=2, label="median of the layer")
    ax.set_ylim(0.55, 5200)
    ax.legend(frameon=False, loc="lower right", fontsize=10, ncol=3)
    s = h["summary"]
    pc = s["corner_percentiles_10_25_50_75_90"]
    fig.suptitle("Each attention head listens at its own timescale", x=0.06, ha="left", fontsize=15,
                 fontweight="bold", color=INK)
    ax.set_title(f"{s['heads_with_corner']} heads, 15 books. Median range {pc[2]:.0f} tokens; 1 in 10 heads listens past "
                 f"{pc[4]:.0f}. Exploratory.", loc="left", fontsize=11, color=MUTED)
    fig.text(0.06, 0.01, f"Source: {HEADMAP} (exploratory, from sealed C1 data)", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out)
    plt.close(fig)


def behavior(repo, out):
    """Exploratory behavior probe: how much harder the new book gets after a switch, per pair and condition."""
    plt = setup()
    import numpy as np
    b = json.load(open(os.path.join(repo, BEHAVIOR)))
    conds = [("S2", "trained dial, 2×", ORANGE), ("S4", "trained dial, 4×", ORANGE),
             ("R2", "random add-on, same size, 2×", BLUE), ("R4", "random add-on, same size, 4×", BLUE)]
    base = b["summary"]["S0"]["mean_excess_loss"]
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=160)
    ys = list(range(len(conds)))[::-1]
    for y, (c, lab, col) in zip(ys, conds):
        mean = b["summary"][c]["mean_excess_loss"] / base - 1
        per = [p["conditions"][c]["excess_loss"] / p["conditions"]["S0"]["excess_loss"] - 1 for p in b["pairs"].values()]
        ax.barh(y, mean * 100, color=col, height=0.55, alpha=0.9)
        ax.scatter(np.array(per) * 100, y + np.linspace(-0.18, 0.18, len(per)), s=18, color="white",
                   edgecolors=INK, linewidths=0.8, zorder=3)
        med = float(np.median(per))
        ax.plot([med * 100] * 2, [y - 0.3, y + 0.3], color=INK, lw=2.5)
        ax.text(max(per) * 100 + 1.2, y, f"mean {mean * 100:+.1f}%, median {med * 100:+.1f}%, {sum(x > 0 for x in per)} of {len(per)} up",
                va="center", fontsize=10.5, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(ys, [c[1] for c in conds], fontsize=11, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.grid(axis="x", color=RULE, lw=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("how much harder the new book is to predict after a switch, compared with the dial at 0×")
    ax.set_xlim(-8, 88)
    fig.suptitle("After a switch between books, the trained dial tends to make the new book harder to predict;\n"
                 "a random change of the same size barely does. The mean leans on one large pair.",
                 x=0.02, ha="left", fontsize=13, fontweight="bold", color=INK)
    ax.set_title(f"{len(b['pairs'])} pairs of books: white dots, one per pair; bar = mean; black tick = median. "
                 "Exploratory, not sealed.", loc="left", fontsize=10.5, color=MUTED)
    fig.text(0.02, 0.01, f"Source: {BEHAVIOR}", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out)
    plt.close(fig)


def dial_gif(repo, out):
    """The dial stepping through its five measured settings (no in-between frames: only measured curves are drawn)."""
    plt = setup()
    import numpy as np
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Wedge
    d = json.load(open(os.path.join(repo, C5)))
    grid = d["grid"]
    keys = ["S0", "S0.5", "S1", "S1.5", "S2"]
    scales = [0, 0.5, 1, 1.5, 2]
    norm = {k: [g / d["curves"][k][-1] * 100 for g in d["curves"][k]] for k in keys}
    seq = keys + keys[-2:0:-1]
    fig = plt.figure(figsize=(9, 5), dpi=110)
    ax = fig.add_axes([0.1, 0.14, 0.58, 0.7])
    gx = fig.add_axes([0.72, 0.42, 0.26, 0.42])
    gx.set_xlim(-1.2, 1.2); gx.set_ylim(-0.35, 1.2); gx.axis("off"); gx.set_aspect("equal")
    gx.add_patch(Wedge((0, 0), 1, 0, 180, width=0.18, color=RULE))
    for sc in scales:
        a = np.pi * (1 - sc / 2)
        gx.text(1.2 * np.cos(a), 1.2 * np.sin(a), f"{sc:g}×", ha="center", va="center", fontsize=10, color=MUTED)
    arc = gx.add_patch(Wedge((0, 0), 1, 180, 180, width=0.18, color=ORANGE))
    needle, = gx.plot([0, 0], [0, 0.8], color=INK, lw=3, solid_capstyle="round")
    gx.plot([0], [0], "o", color=INK, ms=8)
    readout = fig.text(0.85, 0.3, "", ha="center", fontsize=12, color=INK, fontweight="bold")
    ax.plot(grid, norm["S0"], color=BLUE, lw=2, ls=(0, (5, 3)), label="dial at 0×")
    line, = ax.plot(grid, norm["S0"], color=ORANGE, lw=3, marker="o", ms=6, label="dial setting")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 16, 128, 1024, 4096], ["2", "16", "128", "1024", "4096"])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylim(30, 104)
    ax.grid(axis="y", color=RULE, lw=1)
    ax.set_xlabel("tokens per cycle (fast → slow)")
    ax.set_ylabel("gets through (% of slowest)")
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    fig.text(0.1, 0.92, "Turning the dial on Muse Glimmer-30B", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.1, 0.02, f"Measured settings only. Source: {C5}", fontsize=7.5, color=MUTED)

    def draw(i):
        k = seq[i]
        sc = scales[keys.index(k)]
        line.set_ydata(norm[k])
        a = np.pi * (1 - sc / 2)
        needle.set_data([0, 0.8 * np.cos(a)], [0, 0.8 * np.sin(a)])
        arc.set_theta1(180 - 180 * sc / 2)
        ch = d["tilts"][k] / d["tilts"]["S0"] - 1
        readout.set_text(f"{sc:g}×   lean {ch * 100:+.0f}%\nperplexity ×{d['perplexity_ratio_median'][k]:.3f}")
        return line, needle, arc, readout

    anim = FuncAnimation(fig, draw, frames=len(seq), blit=False)
    anim.save(out, writer=PillowWriter(fps=1.2))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    out = os.path.join(repo, "docs", "figures")
    os.makedirs(out, exist_ok=True)
    the_dial(repo, os.path.join(out, "the-dial.png"))
    where(repo, os.path.join(out, "where-the-dial-acts.png"))
    fingerprint(repo, os.path.join(out, "fingerprint.png"))
    heads(repo, os.path.join(out, "heads.png"))
    behavior(repo, os.path.join(out, "behavior.png"))
    dial_gif(repo, os.path.join(out, "the-dial.gif"))
    print("== written: " + ", ".join(sorted(os.listdir(out))))


if __name__ == "__main__":
    main()
