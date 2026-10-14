# -*- coding: utf-8 -*-
"""Decompose DOME's headline gains: raw forecasting vs normalized by reconstruction quality."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Table 2 (-O) numbers, verified from the paper
methods = ["OccWorld-O", "OccLLaMA-O", "DOME-O"]
recon_miou = np.array([66.38, 75.20, 83.08])
fore_miou = np.array([17.14, 19.93, 27.10])
recon_iou = np.array([62.29, 63.76, 77.25])
fore_iou = np.array([26.63, 29.17, 36.36])
ratio_miou = fore_miou / recon_miou * 100
ratio_iou = fore_iou / recon_iou * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
colors = ["#6e737b", "#1a6fa3", "#d97757"]

for ax, vals, title, ylabel in [
    (axes[0], fore_miou, "forecasting mIoU (avg 1s/2s/3s), %", "mIoU (%)"),
    (axes[1], ratio_miou, "forecasting / reconstruction  (head quality, normalized)", "ratio (%)"),
]:
    bars = ax.bar(methods, vals, color=colors, edgecolor="white", linewidth=1.2, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.35, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10, weight="bold")
    ax.set_title(title, fontsize=11, weight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

# annotate relative gains on left panel
axes[0].annotate("", xy=(2, 27.10), xytext=(1, 19.93),
                 arrowprops=dict(arrowstyle="<->", color="#3d3d3a", lw=1.4))
axes[0].text(1.5, 24.6, "+36.0% rel.", ha="center", fontsize=9.5, weight="bold",
             bbox=dict(boxstyle="round,pad=0.25", fc="#f8ecd2", ec="#d97757"))
axes[1].annotate("", xy=(2, ratio_miou[2]), xytext=(1, ratio_miou[1]),
                 arrowprops=dict(arrowstyle="<->", color="#3d3d3a", lw=1.4))
axes[1].text(1.5, (ratio_miou[2] + ratio_miou[1]) / 2 + 0.9, "+23% rel.\n(tokenizer gain removed)",
             ha="center", fontsize=9.5, weight="bold",
             bbox=dict(boxstyle="round,pad=0.25", fc="#e2eff5", ec="#1a6fa3"))

fig.suptitle("Headline +36.0% mIoU mixes two wins: a better tokenizer AND a better forecaster",
             fontsize=12, weight="bold", y=1.02)
plt.tight_layout()
plt.savefig("fig-decompose.png", dpi=170, bbox_inches="tight")
print("saved fig-decompose.png")
print("ratios:", np.round(ratio_miou, 2))
