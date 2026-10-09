# -*- coding: utf-8 -*-
"""Fig A: PDUS progressive trajectory expansion.
Same quantity (lateral offset) shown as transformation: training distribution
starts at original trajectory, each update expands by stride Dy; DriveRestorer
only has to repair artifacts within one stride each round.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "cm",
})
C_IN, C_IN_FILL = "#1a6fa3", "#e2eff5"
C_OUT, C_OUT_FILL = "#9e5c00", "#f8ecd2"
C_OP, C_OP_FILL = "#0b7a5c", "#dcf2e8"
C_ALT, C_ALT_FILL = "#6a55c2", "#ebe7f8"
C_WIRE = "#6e737b"
C_LOSS, C_LOSS_FILL = "#b34a06", "#f9e3d6"

fig, ax = plt.subplots(figsize=(12.6, 4.6))
ax.set_xlim(-0.6, 12.6); ax.set_ylim(-1.5, 3.6); ax.axis("off")

DY = 1.5
xs = [0, DY, 2 * DY, 3 * DY, 4 * DY]  # k = 0..4 updates, y = k*Dy
lane_ys = [0.9, -0.9]
for ly in lane_ys:
    ax.plot([-0.5, 12.5], [ly, ly], color="#c9c6bb", lw=1.4, ls=(0, (6, 4)))
for i, x in enumerate(xs):
    # ego trajectory arrow at this lateral offset
    c = C_IN if i == 0 else (C_ALT if i < 4 else C_OUT)
    cf = C_IN_FILL if i == 0 else (C_ALT_FILL if i < 4 else C_OUT_FILL)
    box = plt.Rectangle((x - 0.62, -0.45), 1.24, 0.9, facecolor=cf,
                        edgecolor=c, lw=1.6, zorder=3)
    ax.add_patch(box)
    ax.text(x, 0.0, f"$T^{{{i}}}_{{\\mathrm{{novel}}}}$" if i else "$T_{\\mathrm{ori}}$",
            ha="center", va="center", fontsize=12, color=c, zorder=4)
    label = f"${i * DY:.1f}\\,\\mathrm{{m}}$"
    ax.text(x, -1.15, label, ha="center", va="top", fontsize=11, color=C_WIRE)
    if i < 4:
        ax.annotate("", xy=(x + DY - 0.62, 1.9), xytext=(x + 0.62, 1.9),
                    arrowprops=dict(arrowstyle="->", color=C_OP, lw=1.6,
                                    mutation_scale=12))
        ax.text(x + DY / 2, 2.15, f"$\\Delta y = {DY}\\,\\mathrm{{m}}$",
                ha="center", fontsize=11, color=C_OP)
        ax.text(x + DY / 2, 1.35, "render + restore", ha="center", fontsize=10,
                color=C_OP, style="italic")
ax.text(-0.35, 2.75, "lateral offset $y = k\\Delta y$", fontsize=13, color=C_WIRE)
ax.text(6.0, -1.15 - 0.0, "", fontsize=1)
# restoration capacity band
band = plt.Rectangle((xs[0], -1.32), xs[-1] - xs[0], 0.22,
                     facecolor=C_OP_FILL, edgecolor=C_OP, lw=1.2, ls=(0, (4, 3)))
ax.add_patch(band)
ax.text(xs[-1] + 0.15, -1.21, "each step stays inside\nDriveRestorer repair capacity",
        fontsize=10, color=C_OP, va="center")

ax.set_title("PDUS: from one big 6 m jump to four 1.5 m steps", fontsize=15, pad=18)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-40/world-model-driving/recondreamer-xray/fig-pdus.png",
            dpi=150, facecolor="#faf9f5")
print("ok")
