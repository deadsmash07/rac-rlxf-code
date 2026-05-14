"""Shared figure-style helpers for the P2 RLxF figure suite.

Identical-discipline mirror of paper/workshop_T1_pluralistic/figures/
_figstyle.py so the two papers share typography, palette, and
aspect-ratio defaults.
"""
from __future__ import annotations

import os
import matplotlib.pyplot as plt


WONG = {
    "black":      "#000000",
    "orange":     "#E69F00",
    "sky_blue":   "#56B4E9",
    "blu_green":  "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "red_purple": "#CC79A7",
    "grey":       "#606060",
    "light_grey": "#B0B0B0",
}


def set_pub_style() -> None:
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset":  "stix",
        "font.size":         8,
        "axes.labelsize":    9,
        "axes.titlesize":    10,
        "axes.titleweight":  "bold",
        "axes.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":    WONG["grey"],
        "axes.labelcolor":   "black",
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "xtick.color":       WONG["grey"],
        "ytick.color":       WONG["grey"],
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "legend.fontsize":   7.5,
        "legend.frameon":    False,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.03,
    })


def panel_label(ax, label: str, x: float = -0.10, y: float = 1.04,
                fontsize: int = 11) -> None:
    ax.text(x, y, label, transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=fontsize, fontweight="bold", color="black")


def save_fig(fig, path: str) -> None:
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {path}")


def here(file: str) -> str:
    return os.path.dirname(os.path.abspath(file))
