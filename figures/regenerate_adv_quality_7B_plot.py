import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WONG = {
    "orange":     "#E69F00",
    "sky_blue":   "#56B4E9",
    "blu_green":  "#009E73",
    "vermillion": "#D55E00",
    "grey":       "#606060",
    "blue":       "#0072B2",
}
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size":        8,
    "axes.labelsize":   8.5,
    "axes.titlesize":   9.5,
    "axes.titleweight": "bold",
    "axes.linewidth":   0.6,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.edgecolor":   WONG["grey"],
    "xtick.labelsize":  7.5,
    "ytick.labelsize":  7.5,
    "xtick.color":      WONG["grey"],
    "ytick.color":      WONG["grey"],
    "legend.fontsize":  7,
    "legend.frameon":   False,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

import os
_results_dir = os.environ.get("RAC_RESULTS_DIR", "results")
with open(os.path.join(_results_dir, "adv_quality_7B", "results.json")) as f:
    data = json.load(f)
per_channel = data["per_channel"]

channels = list(per_channel.keys())
chan_labels = {
    "deterministic": "Det $\\Delta{=}5$",
    "lognormal":     "Lognormal",
    "pareto":        "Pareto $\\alpha{=}2.5$",
}
labels   = [chan_labels.get(c, c) for c in channels]
bias_red = [per_channel[c]["bias_reduction"] for c in channels]
cos_ctrl = [per_channel[c]["cos_control"]    for c in channels]
cos_rac  = [per_channel[c]["cos_rac"]        for c in channels]
x = np.arange(len(channels))

# Wider figure so the y-axis label has room and Panel B legend can sit
# above the bars without overlap.
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

# Panel A: bias reduction.  Shorter y-label so it does not crowd the title.
ax = axes[0]
ax.bar(x, bias_red, color=WONG["blue"], width=0.55, edgecolor="black", linewidth=0.4)
for xi, v in zip(x, bias_red):
    ax.text(xi, v * 1.02, f"{v:.2f}$\\times$", ha="center", va="bottom", fontsize=7.5)
ax.axhline(1.0, color=WONG["grey"], linestyle="--", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("Bias-reduction ratio")
ax.set_title("A. $\\ell_2$-error reduction (oracle target)", pad=8)
ax.set_ylim(0, max(bias_red) * 1.20)
ax.text(0.97, 0.93,
        "$\\tau_{age}{=}1000$, $N{=}500$",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
        color=WONG["grey"])
ax.yaxis.set_label_coords(-0.16, 0.5)

# Panel B: cosine sim (control vs RAC), now correctly y-limited to include
# negative control bars; legend pushed above the axis so it cannot land on
# the bars.
ax    = axes[1]
width = 0.36
ax.bar(x - width / 2, cos_ctrl, width=width, color=WONG["vermillion"],
       edgecolor="black", linewidth=0.4, label="Control (delayed-fast)")
ax.bar(x + width / 2, cos_rac, width=width, color=WONG["blu_green"],
       edgecolor="black", linewidth=0.4, label="\\RAC{}".replace("\\RAC{}", "RAC"))
ax.axhline(0, color=WONG["grey"], linewidth=0.4)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("Cosine similarity to $A_{\\mathrm{sync}}$")
ax.set_title("B. Cosine alignment with oracle", pad=10)
ymin = min(min(cos_ctrl), 0) - 0.10
ymax = max(max(cos_rac), 0)  + 0.32
ax.set_ylim(ymin, ymax)
ax.yaxis.set_label_coords(-0.13, 0.5)
# Legend INSIDE the panel, just below the title and above the bars.
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 0.99),
    ncol=2,
    frameon=False,
    columnspacing=1.5,
    fontsize=7,
)

fig.subplots_adjust(wspace=0.32, top=0.88)
fig.savefig(
    os.environ.get("RAC_FIGURES_DIR", "figures") + "/adv_quality_7B.pdf",
    format="pdf", bbox_inches="tight", pad_inches=0.04,
)
plt.close(fig)
print("wrote adv_quality_7B.pdf")
