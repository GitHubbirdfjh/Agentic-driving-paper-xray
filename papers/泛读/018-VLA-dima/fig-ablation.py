# -*- coding: utf-8 -*-
"""DiMA Fig A: Table 4 ablation as incremental gains (ID1 -> ID8).

ID1 VAD-Tiny baseline; ID2 +VQA w/ BEV tokens only; ID3 +map; ID4 all BEAM;
ID5 +distillation; ID6 +masked recon; ID7 +future pred; ID8 +scene editing.
Values are Ave_all over 1s/2s/3s, VAD evaluation, nuScenes val.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

labels = ["ID1\nbaseline", "ID2\n+VQA\n(BEV)", "ID3\n+map", "ID4\nall BEAM",
          "ID5\n+distill", "ID6\n+masked\nrecon", "ID7\n+future\npred", "ID8\n+scene\nediting"]
l2 = [0.60, 0.62, 0.56, 0.52, 0.48, 0.42, 0.39, 0.38]
col = [0.29, 0.26, 0.25, 0.21, 0.19, 0.18, 0.16, 0.15]

fig, axes = plt.subplots(2, 1, figsize=(10.5, 7), sharex=True)
ax = axes[0]
bars = ax.bar(labels, l2, color="#1a6fa3", edgecolor="black", linewidth=0.6)
ax.set_ylabel("Trajectory L2 (m), Ave_all")
ax.set_title("DiMA ablation (Table 4, DiMA-VAD-Tiny, VAD evaluation): every component earns its place", fontsize=12)
ax.set_ylim(0, 0.7)
for b, v in zip(bars, l2):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
ax.annotate("naive VQA hurts slightly:\n0.60 -> 0.62", xy=(1, 0.62), xytext=(0.2, 0.68),
            fontsize=9, color="#b34a06", arrowprops=dict(arrowstyle="->", color="#b34a06"))
ax.annotate("all BEAM tokens\nin, not just BEV:\n0.56 -> 0.52", xy=(3, 0.52), xytext=(4.4, 0.60),
            fontsize=9, color="#0b7a5c", arrowprops=dict(arrowstyle="->", color="#0b7a5c"))

ax = axes[1]
bars = ax.bar(labels, col, color="#9e5c00", edgecolor="black", linewidth=0.6)
ax.set_ylabel("Collision rate (%), Ave_all")
ax.set_ylim(0, 0.34)
for b, v in zip(bars, col):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.007, f"{v:.2f}", ha="center", fontsize=9)
ax.annotate("ID1 -> ID8: 0.60 -> 0.38 L2 (-37%), 0.29 -> 0.15 collision (-48%)",
            xy=(7, 0.15), xytext=(4.2, 0.26), fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8ecd2"),
            arrowprops=dict(arrowstyle="->", color="black"))
ax.tick_params(axis="x", labelsize=9)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/018-VLA-dima/fig-ablation.png", dpi=150)
print("saved fig-ablation.png")
