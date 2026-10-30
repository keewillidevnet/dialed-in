"""The README's figures, drawn from the result files. No GPU.

    PYTHONPATH=src python3 -m dialedin.figures            # writes docs/figures/the-dial.png and where-the-dial-acts.png

Every number drawn is read here from a summary file under results/derived/, named in each figure's footer.
"""
import argparse
import json
import os

C5 = "results/derived/c5v1-20261029T025343Z/c5_summary.json"
C7 = "results/derived/c7v1-20261029T154201Z/c7_summary.json"
C8 = "results/derived/c8v1-20261029T222946Z/c8_summary.json"
C9 = "results/derived/c9v1-20261030T023935Z/c9_summary.json"
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
    plt = setup()
    c7, c8, c9 = (json.load(open(os.path.join(repo, p))) for p in (C7, C8, C9))
    on = [("Layer 2 q, k, v (3 of 32 parts)   C7, books set B", c7["C7"]["share_QKV2"], c7["bootstrap_95"]["share_QKV2"], GRAY),
          ("Layer 2 v alone (1 of 32)   C8, set C", c8["C8"]["share_V2"], c8["bootstrap_95"]["share_V2"], ORANGE),
          ("Layer 2 v alone (1 of 32)   C9, set A", c9["C9"]["retained_V2"], c9["bootstrap_95"]["retained_V2"], ORANGE),
          ("Layer 4 v alone, control   C8, set C", c8["C8"]["share_V4"], c8["bootstrap_95"]["share_V4"], BLUE)]
    off = [("All but layer 2 v   C9, set A", c9["C9"]["retained_NOV2"], c9["bootstrap_95"]["retained_NOV2"], ORANGE),
           ("All but layer 2's gate, control   C9", c9["C9"]["retained_NOG2"], c9["bootstrap_95"]["retained_NOG2"], BLUE),
           ("All but layer 4 v, control   C9", c9["C9"]["retained_NOV4"], c9["bootstrap_95"]["retained_NOV4"], BLUE)]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.8), dpi=160, gridspec_kw={"height_ratios": [4, 3], "hspace": 0.55})
    for ax, rows, title in ((axes[0], on, "Switch ONE part on: how much of the dial's effect do you get?"),
                            (axes[1], off, "Switch ONE part off: how much of the dial's effect is left?")):
        ys = list(range(len(rows)))[::-1]
        for y, (lab, v, ci, col) in zip(ys, rows):
            ax.barh(y, v * 100, color=col, height=0.62)
            if ci:
                ax.plot([ci[0] * 100, ci[1] * 100], [y, y], color=INK, lw=1.5)
            ax.text(max(v, ci[1] if ci else v) * 100 + 1.5, y, f"{v * 100:.0f}%", va="center", fontsize=12,
                    fontweight="bold", color=INK)
        ax.set_yticks(ys, [r[0] for r in rows], fontsize=11, color=INK)
        ax.set_xlim(0, 108)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.set_xlabel("share of the full dial's effect on the tilt", fontsize=10)
        ax.grid(axis="x", color=RULE, lw=1)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", fontsize=12.5, color=INK, fontweight="bold")
        ax.axvline(100, color=MUTED, lw=1, ls=":")
    fig.suptitle("About half of the dial is one matrix, layer 2's v:\nenough alone for about half, and needed for about half",
                 x=0.02, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.02, 0.025, "100% = the full dial (all 32 parts at 2×). Black lines: 95% intervals over 5 books. Each experiment "
             "used a different set of 5 books.\nSources: " + ", ".join((C7, C8, C9)), fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.36, right=0.97, top=0.85, bottom=0.14)
    fig.savefig(out)
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
    print(f"== written: docs/figures/the-dial.png and docs/figures/where-the-dial-acts.png")


if __name__ == "__main__":
    main()
