"""Figure for EXP-C — MDP-size scaling of the RAC K=2 bias-reduction.

Consumes the JSON produced by
`2_Delay_Aware_RLHF/scripts/adv_mdp_scaling.py` and emits a 2-panel
figure: (A) bias-reduction vs MDP size with per-Delta box-plots over
the MDP-seed ensemble; (B) per-Delta reduction across sizes (line
plot).

Skill citation: research-grade-plots.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))
from _figstyle import WONG, set_pub_style, save_fig  # noqa: E402


def load_latest_results(adv_results_root: Path) -> dict:
    candidates = sorted(adv_results_root.glob("adv_mdp_scaling_*/results.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No adv_mdp_scaling_*/results.json under {adv_results_root}",
        )
    non_smoke = [p for p in candidates if "smoke" not in p.parent.name]
    return json.loads((non_smoke[-1] if non_smoke else candidates[-1]).read_text())


def main() -> int:
    set_pub_style()
    t2_root = Path(
        "/Users/arnav/Documents/research/nw/IMPLEMENTATION/"
        "2_Delay_Aware_RLHF/results"
    )
    data = load_latest_results(t2_root)

    sizes = list(data["results"].keys())
    deltas = sorted([int(k) for k in
                     data["results"][sizes[0]]["per_delta_per_seed_reduction"].keys()])

    # Aggregate
    per_size_all = {s: [] for s in sizes}
    per_size_per_delta = {s: {d: [] for d in deltas} for s in sizes}
    for s in sizes:
        cell = data["results"][s]
        for d in deltas:
            per_size_per_delta[s][d] = cell["per_delta_per_seed_reduction"][str(d)]
            per_size_all[s].extend(cell["per_delta_per_seed_reduction"][str(d)])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), dpi=150)
    delta_colors = [WONG["vermillion"], WONG["blu_green"], WONG["orange"]]

    # Panel A: violin/box of all (delta x seed) reductions per size
    ax = axes[0]
    box_data = [per_size_all[s] for s in sizes]
    bp = ax.boxplot(
        box_data, labels=sizes, patch_artist=True, widths=0.55,
        medianprops=dict(color="black", linewidth=1.0),
        boxprops=dict(linewidth=0.6),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
        flierprops=dict(marker="o", markersize=2.0,
                        markerfacecolor=WONG["grey"],
                        markeredgecolor=WONG["grey"]),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(WONG["sky_blue"])
        patch.set_alpha(0.55)
    ax.axhline(1.0, linestyle="--", color=WONG["grey"], linewidth=0.7, alpha=0.7)
    # Mean markers
    means = [np.mean(per_size_all[s]) for s in sizes]
    ax.plot(np.arange(1, len(sizes) + 1), means, "D",
            color=WONG["vermillion"], markersize=4.5, label="mean")
    ax.set_xlabel(r"MDP size $|S|\!\times\!|A|$")
    ax.set_ylabel(r"bias-reduction $\uparrow$")
    ax.set_yscale("log")
    ax.set_title(r"(A) Bias-reduction across MDP sizes (pooled $\Delta$, seeds)")
    ax.legend(loc="best", fontsize=6.5)
    ax.grid(alpha=0.25, linewidth=0.4, axis="y")

    # Panel B: per-Delta lines across sizes
    ax = axes[1]
    x = np.arange(len(sizes))
    for color, d in zip(delta_colors, deltas):
        ys = [np.mean(per_size_per_delta[s][d]) for s in sizes]
        ax.plot(x, ys, "o-", color=color, label=f"$\\Delta\\!=\\!{d}$", markersize=4)
    ax.axhline(1.0, linestyle="--", color=WONG["grey"], linewidth=0.7, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_xlabel(r"MDP size $|S|\!\times\!|A|$")
    ax.set_ylabel(r"bias-reduction (mean over seeds) $\uparrow$")
    ax.set_yscale("log")
    ax.set_title("(B) Per-$\\Delta$ reduction across MDP sizes")
    ax.legend(loc="best", fontsize=6.5)
    ax.grid(alpha=0.25, linewidth=0.4)

    fig.tight_layout()
    out_path = Path(__file__).resolve().parent / "mdp_scaling.pdf"
    save_fig(fig, str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
