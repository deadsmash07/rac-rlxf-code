"""Figure 4 (P2 RLxF) --- T2 cross-MDP-topology K-sweep heatmap.

REGENERATED 2026-05-12: replaced prior synthetic-label placeholder
(topologies {chain, branch, star, tree, lattice} with hand-picked
cell values) with the actual cross-topology experiment data from
track2_K_sweep_cross_mdp_topology/topology_*.json.

Source data: per_K_topology_aggregate.<K>.reduction_topology_mean for
each topology, computed as the mean reduction-factor across 5 mdp
seeds (1024, 42, 7777, 1337, 31337) for each K in {2,3,5,7,10,15,20}.

The five real topologies span structural diversity (state count,
action count, and reachability graph):
  canonical_3s2a : baseline 3-state 2-action MDP (fully connected)
  chain_5s2a     : 5-state linear chain, 2 actions per state
  cyclic_4s3a    : 4-state cyclic graph, 3 actions per state
  dense_5s3a     : 5-state densely connected, 3 actions per state
  terminal_3s2a  : 3-state with absorbing terminal, 2 actions

honest-disclosure (peak-K varies by topology, K=2 NOT universal): a
yellow border on the per-row peak K + a triangle pointer caret above
the column makes the peak unambiguous without overlaying cell text.

Anonymity-clean.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _figstyle import WONG, set_pub_style, save_fig, here


import os

# Source-data root: cross-MDP-topology K-sweep aggregate jsons.
# Override via RAC_RESULTS_DIR env var; default is the in-repo results dir.
DATA_ROOT = os.environ.get(
    "RAC_RESULTS_DIR",
    "results/track2_K_sweep_cross_mdp_topology",
)

# Ordered (display-label, json-filename) pairs.  Display labels
# include a shape annotation in parens for reviewer clarity (state x
# action count); these match the labels used in tab:t2ksweep + body
# §3 prose so the appendix figure does not collide with the table.
TOPOLOGIES = [
    ("canonical (3$\\times$2)", "topology_canonical_3s2a.json"),
    ("chain (5$\\times$2)",     "topology_chain_5s2a.json"),
    ("cyclic (4$\\times$3)",    "topology_cyclic_4s3a.json"),
    ("dense (5$\\times$3)",     "topology_dense_5s3a.json"),
    ("terminal (3$\\times$2)",  "topology_terminal_3s2a.json"),
]

K_GRID = [2, 3, 5, 7, 10, 15, 20]


def load_reduction_matrix() -> np.ndarray:
    """Build the (5 topologies x 7 K) reduction matrix from json."""
    M = np.zeros((len(TOPOLOGIES), len(K_GRID)), dtype=float)
    for i, (_, fname) in enumerate(TOPOLOGIES):
        path = os.path.join(DATA_ROOT, fname)
        with open(path, "r") as fh:
            blob = json.load(fh)
        agg = blob["per_K_topology_aggregate"]
        for j, K in enumerate(K_GRID):
            cell = agg[str(K)]
            M[i, j] = float(cell["reduction_topology_mean"])
    return M


def main() -> None:
    set_pub_style()

    M = load_reduction_matrix()

    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    # Cividis is colorblind-safe (linear perceptual ramp; tested for
    # deuteranopia + protanopia).  vmax tracks the empirical maximum
    # (~135) so the colormap covers the full range without saturating
    # the high-end cells.
    vmin = float(np.floor(M.min() / 5.0) * 5.0)
    vmax = float(np.ceil(M.max() / 5.0) * 5.0)
    im = ax.imshow(M, aspect="auto", cmap="cividis",
                    vmin=vmin, vmax=vmax, interpolation="none")

    # Per-row peak: yellow rectangle border on the peak cell + a
    # triangle-pointer caret OUTSIDE the heatmap (above the top edge)
    # at the column of the peak.  This communicates the peak K
    # without obscuring the cell number.
    peak_cols = []
    for i in range(M.shape[0]):
        peak = int(np.argmax(M[i]))
        peak_cols.append(peak)
        rect = Rectangle((peak - 0.48, i - 0.48), 0.96, 0.96,
                         linewidth=1.8, edgecolor=WONG["yellow"],
                         facecolor="none", zorder=4)
        ax.add_patch(rect)

    # Triangle-pointer carets above the top edge, one per row's peak,
    # at horizontal positions corresponding to the unique peak K's.
    # (Multiple rows share K=5 as peak so duplicates are collapsed.)
    seen_cols = set()
    for col in peak_cols:
        if col in seen_cols:
            continue
        seen_cols.add(col)
        ax.scatter(col, -0.65, marker="v", s=22,
                    color=WONG["yellow"], edgecolor=WONG["black"],
                    linewidth=0.4, zorder=5, clip_on=False)

    # Cell value annotations.  Color-flip threshold around mid-range
    # so dark cividis cells get white text and bright cells get black.
    mid = 0.55 * (vmin + vmax)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.1f}$\\times$",
                    ha="center", va="center", fontsize=7,
                    color="white" if v < mid else "black")

    ax.set_xticks(range(len(K_GRID)))
    ax.set_xticklabels([f"$K{{=}}{k}$" for k in K_GRID], fontsize=8)
    ax.set_yticks(range(len(TOPOLOGIES)))
    ax.set_yticklabels([label for label, _ in TOPOLOGIES], fontsize=8)
    ax.set_xlabel("Slow-channel count $K$", fontsize=9)
    ax.set_ylabel("MDP topology (states$\\times$actions)", fontsize=9)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)

    cbar = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.025)
    cbar.set_label(r"Bias-reduction ratio (mean over 5 mdp seeds)",
                    fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_visible(False)

    save_fig(fig, here(__file__) + "/fig4_k_sweep_heatmap.pdf")


if __name__ == "__main__":
    main()
