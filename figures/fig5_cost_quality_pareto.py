"""Figure 5 (P2 RLxF) --- Cost-quality Pareto plot:
wall-clock-multiplier x bias-reduction-ratio with RAC Pareto-dominant.

Authored  Improvement 3: "Add cost-quality tradeoff
figure (2-D scatter: wall-clock x bias-reduction with RAC
Pareto-dominant)."

Anchored on Tab 1 "K=2 baseline cmp." block:
    naive   ( 1.0x wall-clock,   1.0x reduction )
    Retrace-A( 1.2x wall-clock,   1.5x reduction )  -- gamma^Delta age decay
    wait    (26.0x wall-clock,  27.1x reduction )  -- bare-additive floor
    RAC     ( 1.0x wall-clock,  47.9x reduction )  -- Pareto-dominant

The Pareto front is RAC alone: it has the same wall-clock as naive
and a higher reduction than wait-for-slow at 26x lower cost.

Tol 2018 + Wong 2011 colorblind-safe palettes; identical-discipline
to fig3 RAC vs naive bar chart.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _figstyle import WONG, set_pub_style, save_fig, here


def main() -> None:
    set_pub_style()

    # Wider + taller for legend-above-axes layout per user feedback:
    # ("legend is on the plot itself and hindering it") .
    fig, ax = plt.subplots(figsize=(4.4, 3.2))

    # Data anchored on Tab 1 "K=2 baseline cmp." block.  All four
    # correctors rendered with the same marker size; the axis-arrow
    # direction labels carry the "which corner is better" information
    # without need for in-figure decoration.
    methods = [
        ("naive PPO",          1.00,   1.0, WONG["grey"],       "o", 70),
        ("Retrace-A",          1.20,   1.5, WONG["sky_blue"],   "s", 70),
        ("wait-for-slow",     26.00,  27.1, WONG["vermillion"], "^", 80),
        ("RAC (ours)",         1.00,  47.9, "#117733",          "D", 80),
    ]

    for name, wc, red, color, marker, size in methods:
        ax.scatter(wc, red, c=color, s=size, marker=marker,
                   edgecolors="black", linewidths=0.8,
                   label=name, zorder=5)

    # Annotate each point with its (wc, red) tuple. Offsets chosen so that
    # text never lands underneath a marker; zorder=10 forces text above the
    # scatter markers (zorder=5) for double safety per user feedback:
    # ("blue dot covers the number") .
    annotations = [
        # name              wc      red    dx     dy   ha
        ("naive PPO",       1.00,   1.0,  -0.10,  -0.20,  "right"),  # below-left of marker
        ("Retrace-A",       1.20,   1.5,  +0.18,  +0.55,  "left"),   # clearly above-right
        ("wait-for-slow",  26.00,  27.1,  -0.40,  -7.0,   "right"),  # below-left
        ("RAC (ours)",      1.00,  47.9,  +0.10,  +5.5,   "left"),   # above-right
    ]
    for name, wc, red, dx, dy, ha in annotations:
        ax.annotate(f"{wc:.0f}x, {red:.1f}x", (wc, red),
                    xytext=(wc + dx, red + dy),
                    fontsize=7, color="black",
                    ha=ha, zorder=10,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, pad=0.6))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.7, 50)
    ax.set_ylim(0.7, 90)
    ax.set_xticks([1, 2, 5, 10, 26])
    ax.set_xticklabels(["1$\\times$", "2$\\times$", "5$\\times$",
                        "10$\\times$", "26$\\times$"])
    ax.set_yticks([1, 5, 10, 30, 50, 80])
    ax.set_yticklabels(["1$\\times$", "5$\\times$", "10$\\times$",
                        "30$\\times$", "50$\\times$", "80$\\times$"])
    ax.set_xlabel("Wall-clock cost relative to naive PPO  ($\\downarrow$ better)",
                  fontsize=8.5)
    ax.set_ylabel("Bias-reduction ratio vs naive PPO  ($\\uparrow$ better)",
                  fontsize=8.5)
    # Legend ABOVE the plot in a single horizontal row so it never overlaps
    # data points or the Pareto-dominance arrow.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=4, fontsize=7.5, frameon=False,
              handletextpad=0.4, borderaxespad=0.2,
              columnspacing=1.0)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, which="major", alpha=0.20, linewidth=0.4)
    fig.tight_layout(pad=0.6)

    save_fig(fig, here(__file__) + "/fig5_cost_quality_pareto.pdf")


if __name__ == "__main__":
    main()
