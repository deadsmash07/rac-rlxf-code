"""Figure 2 (P2 RLxF) --- Five delay distributions matched at the same
expected delay.

REGENERATED 2026-04-28 23:35 IST per user 23:30 IST critique:
  - replaced dashed/dotted/dash-dot linestyles with SOLID lines of
    distinct hues (Wong + Tol palettes).  Differentiation is now
    purely chromatic; no linestyle confusion at column-width.
  - reduced deterministic point-mass marker (ms 6.5 -> 4.0) and
    thinned the vertical pin (lw 1.0 -> 0.7); it now reads as a
    thin spike rather than a covering blob.

PRIOR 2026-04-28 22:21 IST: x-axis tightened to 5..50; legend moved.
PRIOR 2026-04-28 21:10 IST: switched cividis to Wong-derived palette.

Anonymity-clean.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _figstyle import WONG, set_pub_style, save_fig, here


def main() -> None:
    set_pub_style()

    fig, ax = plt.subplots(figsize=(3.8, 2.4))

    target_mean = 20
    grid = np.arange(1, 121)

    # SOLID-color palette (no linestyles): each curve uses a distinct
    # hue from the Wong + Tol Bright palettes for chromatic-only
    # discrimination.  All five curves remain distinguishable at
    # column-width 3.4in even after greyscale conversion via
    # luminance contrast.
    # Order: deterministic -> Gaussian -> Lognormal -> Pareto -> Cauchy.
    colors = [
        "#999999",          # deterministic: Wong grey (small marker)
        WONG["blue"],       # Gaussian:        #0072B2 strong blue
        WONG["vermillion"], # Lognormal:       #D55E00 strong orange-red
        WONG["red_purple"], # Pareto-finite:   #CC79A7 magenta-purple
        "#117733",          # Trunc-Cauchy:    Tol teal-green
    ]

    # Deterministic point mass: small marker + thin vertical pin.
    # Reduced from ms=6.5/lw=1.0 to ms=4.0/lw=0.7 per user critique
    # 23:30 IST: prior version was a "big dot covering everything up".
    det = np.zeros_like(grid, dtype=float)
    det[target_mean - 1] = 1.0
    ax.vlines(target_mean, 0, 1.0, color=colors[0], lw=0.7, zorder=2)
    ax.plot(grid, det, lw=0, marker="o", ms=4.0, color=colors[0],
            markeredgecolor="black", markeredgewidth=0.4,
            zorder=3, label="Deterministic")

    # Gaussian (sigma=4) -- solid blue
    g = np.exp(-0.5 * ((grid - target_mean) / 4.0) ** 2)
    g /= g.sum()
    ax.plot(grid, g, lw=1.6, color=colors[1], linestyle="-",
            label="Gaussian")

    # Lognormal -- solid vermillion
    sigma = 0.5
    mu = np.log(target_mean) - sigma ** 2 / 2.0
    lg = (1.0 / (grid * sigma * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(grid) - mu) ** 2) / (2 * sigma ** 2)
    )
    lg /= lg.sum()
    ax.plot(grid, lg, lw=1.6, color=colors[2], linestyle="-",
            label="Lognormal")

    # Pareto-finite -- solid magenta
    alpha = 3.0
    rng = np.random.default_rng(20260428)
    samp = (rng.pareto(alpha, size=200000) + 1.0) * (
        target_mean * (alpha - 1.0) / alpha
    )
    samp = samp[samp <= 200]
    hist, _ = np.histogram(samp, bins=np.arange(1, 122))
    hist = hist / hist.sum()
    ax.plot(grid, hist, lw=1.6, color=colors[3], linestyle="-",
            label="Pareto-finite")

    # Truncated Cauchy -- solid teal-green
    gamma = 5.0
    c = (1.0 / np.pi) * (gamma / ((grid - target_mean) ** 2 + gamma ** 2))
    c /= c.sum()
    ax.plot(grid, c, lw=1.6, color=colors[4], linestyle="-",
            label="Trunc-Cauchy")

    ax.axvline(target_mean, color=WONG["grey"], lw=0.6, ls=":", zorder=1)
    # Annotation moved to upper-left clear space; no longer collides
    # with Gaussian/Lognormal peaks at the zoomed scale.
    ax.text(target_mean + 0.6, 0.88,
            r"$\mathbb{E}[\Delta]\,{=}\,20$",
            color=WONG["grey"], fontsize=7.5,
            transform=ax.get_xaxis_transform())

    # x-range tightened from 1..80 to 5..50 per user critique 22:13
    # IST.  Active region (target_mean=20) sits at panel center;
    # heavy-tailed distributions still show their tail-behavior
    # within the 30-50 right-region without the 50-80 dead-space.
    ax.set_xlim(5, 50)
    ax.set_xlabel(r"Delay $\Delta$ (optimizer steps)", fontsize=9)
    ax.set_ylabel(r"Probability mass", fontsize=9)
    ax.legend(loc="upper right", fontsize=7.5, handletextpad=0.5,
              borderpad=0.4)

    save_fig(fig, here(__file__) + "/fig2_delay_distributions.pdf")


if __name__ == "__main__":
    main()
