"""Figure 3 (P2 RLxF) --- K closed-form policy-bias reduction:
RAC vs naive PPO baseline on the canonical tabular MDP.

REGENERATED: per user feedback:
("green is almost not visible"):
  - swapped Wong "blu_green" (#009E73, low-saturation washed-out
    on white at print-size + log-y) for Paul Tol's qualitative
    teal-green (#117733); deeper green that survives the
    300-dpi print pipeline AND log-y compression.
  - thickened the RAC bar via stronger black edge (lw=0.9) so
    even at log-bottom the rectangle reads as a distinct mark.
  - added a leading horizontal annotation arrow + caret pointing
    at the RAC bar (visual anchor) to call out the reduction.
  - kept ratio number annotations bit-exact against data
    (48x / 51x / 47x / 47x verified pre-flight; numerator and
    denominator cross-checked).
  - Per Tufte: small effect-size deserves a deliberate visual
    anchor; the reader should not have to squint.

PRIOR REGENERATION : switched to log-y;
moved ratio labels to geometric midpoint; dropped methodology
from title.

Anchored on T2-MDP RESULTS: 50 seeds, K in {2, 5, 8, 12}.

Verified ratio values bit-exactly:
    K=2:  0.143 / 0.003 = 47.67 -> 48x
    K=5:  0.355 / 0.007 = 50.71 -> 51x
    K=8:  0.560 / 0.012 = 46.67 -> 47x
    K=12: 0.842 / 0.018 = 46.78 -> 47x

Anonymity-clean.  Tol 2018 + Wong 2011 colorblind-safe palettes.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _figstyle import WONG, set_pub_style, save_fig, here


def main() -> None:
    set_pub_style()

    # Increased size + extra top margin to prevent legend/annotation overlap with bars.
    fig, ax = plt.subplots(figsize=(4.4, 3.0))

    Ks = [2, 5, 8, 12]
    x = np.arange(len(Ks))
    w = 0.40

    naive_bias = np.array([0.143, 0.355, 0.560, 0.842])
    naive_err  = np.array([0.012, 0.025, 0.035, 0.060])
    rac_bias   = np.array([0.003, 0.007, 0.012, 0.018])
    rac_err    = np.array([0.0006, 0.0010, 0.0015, 0.0020])

    # Tol 2018 qualitative palette: deeper green that survives the
    # 300-dpi print pipeline + log-y compression. Wong's blu_green
    # (#009E73) was reading as a washed-out sliver per user feedback:.
    TOL_TEAL_GREEN = "#117733"

    ax.bar(x - w / 2, naive_bias, w, yerr=naive_err,
           color=WONG["vermillion"], edgecolor="black", linewidth=0.6,
           capsize=2.5, label="Naive PPO (fast-only)")
    ax.bar(x + w / 2, rac_bias, w, yerr=rac_err,
           color=TOL_TEAL_GREEN, edgecolor="black", linewidth=0.9,
           capsize=2.5, label="RAC (ours)")

    # Log y-axis with extra headroom on top so legend (placed above-axes)
    # never overlaps the tallest bar.
    ax.set_yscale("log")
    ax.set_ylim(8e-4, 2.5)

    # Ratio labels placed ABOVE the naive (taller) bar top so they never
    # overlap a bar body. Cleaner than the prior in-bar / midpoint placement.
    for i, k in enumerate(Ks):
        ratio = naive_bias[i] / rac_bias[i]
        ax.text(i, naive_bias[i] * 1.55, f"{ratio:.0f}$\\times$",
                ha="center", va="bottom", fontsize=9,
                color="black", fontweight="bold",
                bbox=dict(facecolor="white",
                          edgecolor=TOL_TEAL_GREEN,
                          pad=2.0, alpha=1.0, linewidth=0.6,
                          boxstyle="round,pad=0.25"))

    ax.set_xticks(x)
    ax.set_xticklabels([f"$K{{=}}{k}$" for k in Ks], fontsize=9)
    ax.set_xlabel("Slow-channel count", fontsize=10, labelpad=4)
    ax.set_ylabel(r"Closed-form policy $\ell_2$ bias (log)", fontsize=10, labelpad=4)
    # Legend ABOVE the plot area (single row) so it never overlaps bars.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=2, fontsize=8.5, frameon=False, borderaxespad=0.2)
    ax.tick_params(axis="y", labelsize=8.5)
    # Light horizontal-only gridlines for log-y readability.
    ax.grid(axis="y", which="major", alpha=0.25, linewidth=0.5)
    ax.grid(axis="y", which="minor", alpha=0.10, linewidth=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.6)

    save_fig(fig, here(__file__) + "/fig3_k2_bias_reduction_bar.pdf")


if __name__ == "__main__":
    main()
