"""Figure 1 (P2 RLxF) --- Retroactive Advantage Correction schematic.

VARIATION 1 (2026-05-15): RESTORE the original schematic design that the
user liked --- fast-reward down-arrows above the timeline, slow-reward
up-arrows below the timeline, dashed delay arcs visualizing the
origin-to-arrival journey, right-edge labels for each panel, RAC
forward-injection arc on Panel C.

The ONLY fix vs the original: move timestamps t_0..t_7 BELOW the slow
up-arrow shafts (instead of between them and the baseline) so the orange
arrows no longer occlude the timestamp labels.

Anonymity-clean.  Wong 2011 colorblind-safe palette.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from _figstyle import WONG, set_pub_style, save_fig, here, panel_label


def _draw_panel(ax, title, slow_origins, slow_arrivals,
                forward_arrows=None, color_slow=WONG["vermillion"]):
    """One subplot panel: a single horizontal timeline with arrows."""
    n = 8
    xs = np.arange(n)
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(-2.40, 2.00)
    ax.axis("off")

    # baseline timeline
    ax.hlines(0, -0.4, n - 0.6, color=WONG["grey"], lw=1.0)
    for i in xs:
        ax.plot(i, 0, "o", color="black", ms=3.5, zorder=3)
        # Timestamps placed BELOW the slow-return arrow tails AND below the
        # right-edge "slow returns" label band so the orange text near t_7
        # never visually merges with the timestamp.
        ax.text(i, -2.10, f"$t_{{{i}}}$", ha="center", va="center",
                fontsize=7, color=WONG["grey"])

    # Fast rewards (every step): thin blue down-arrows above the baseline.
    for i in xs:
        ax.add_patch(FancyArrowPatch((i, 1.05), (i, 0.10),
                                     arrowstyle="-|>", color=WONG["blue"],
                                     mutation_scale=8, lw=0.9))
    ax.text(n - 0.5, 1.20, "fast rewards", fontsize=7, color=WONG["blue"],
            ha="right", va="bottom")

    # Slow returns: orange/red up-arrows from below the baseline.
    for src, dst in zip(slow_origins, slow_arrivals):
        ax.add_patch(FancyArrowPatch((dst, -1.30), (dst, -0.10),
                                     arrowstyle="-|>", color=color_slow,
                                     mutation_scale=8, lw=0.95))
        # Dashed delay arc: shows the slow signal travelling from the
        # rollout origin (src) to the optimiser step where it arrives (dst).
        # Placed between baseline (y=0) and arrow tails (y=-1.30); modest
        # bulge so the arc stays in the [-0.40, -0.75] band and never
        # touches the timestamps below at y=-2.10.
        ax.annotate("", xy=(dst - 0.15, -0.55),
                    xytext=(src + 0.15, -0.55),
                    arrowprops=dict(arrowstyle="->", color=color_slow,
                                    lw=0.75, ls="--",
                                    connectionstyle="arc3,rad=-0.28"))
    if slow_origins:
        # "slow returns" label band: y in [-1.50, -1.70]; clearly above the
        # timestamps row at y=-2.10.
        ax.text(n - 0.5, -1.50, "slow returns", fontsize=7,
                color=color_slow, ha="right", va="top")

    # Forward-injection arrows (RAC only): green from arrival -> next step.
    if forward_arrows is not None:
        for src, dst in forward_arrows:
            ax.add_patch(FancyArrowPatch((src, 0.10), (dst, 0.95),
                                         arrowstyle="-|>",
                                         color=WONG["blu_green"],
                                         mutation_scale=10, lw=1.3,
                                         connectionstyle="arc3,rad=0.40"))
        # "RAC forward-injection" label moved further up (y=1.70 vs prior
        # 1.45) so the green text has clear vertical separation from the
        # blue "fast rewards" text at y=1.20 (gap ~0.50 instead of ~0.25).
        ax.text(n - 0.5, 1.70, "RAC forward-injection",
                fontsize=7, color=WONG["blu_green"], ha="right", va="bottom",
                fontweight="bold")

    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold",
                 pad=4)


def main(out_path: str | None = None) -> None:
    set_pub_style()

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 4.5),
                              gridspec_kw={"hspace": 0.55})

    _draw_panel(axes[0], "  Synchronous PPO",
                slow_origins=[], slow_arrivals=[],
                forward_arrows=None, color_slow=WONG["vermillion"])
    panel_label(axes[0], "A", x=-0.045, y=1.08, fontsize=11)

    _draw_panel(axes[1], "  Slow channel (naive PPO)",
                slow_origins=[0, 2, 4], slow_arrivals=[3, 5, 7],
                forward_arrows=None, color_slow=WONG["vermillion"])
    panel_label(axes[1], "B", x=-0.045, y=1.08, fontsize=11)

    _draw_panel(axes[2], "  RAC: forward-inject $\\delta_i$ next step",
                slow_origins=[0, 2, 4], slow_arrivals=[3, 5, 7],
                forward_arrows=[(3, 4), (5, 6)],
                color_slow=WONG["orange"])
    panel_label(axes[2], "C", x=-0.045, y=1.08, fontsize=11)

    if out_path is None:
        out_path = here(__file__) + "/fig1_rac_schematic.pdf"
    save_fig(fig, out_path)


if __name__ == "__main__":
    main()
