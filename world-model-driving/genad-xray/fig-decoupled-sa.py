"""Figure: why temporal attention alone fails on drastic view shift;
decoupled spatial attention propagates along horizontal then vertical axes."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

H = W = 6
qy, qx = 2, 2          # query grid cell
# car moves far between frames: same object appears at (2,2) at t and (4,4) at t+1
oy, ox = 4, 4

fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))

def draw_grid(ax, title):
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(-0.5, H - 0.5)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for yy in range(H):
        for xx in range(W):
            ax.add_patch(Rectangle((xx - 0.45, yy - 0.45), 0.9, 0.9,
                                   facecolor="#f2f0e8", edgecolor="#c9c6bb", lw=0.8))
    ax.set_title(title, fontsize=13, pad=10)

# Panel 1: temporal-only attention: query cell at (2,2) at time t, object at (4,4) at t+1 -> miss
ax = axes[0]
draw_grid(ax, "Temporal attention only\n(same grid cell across time)")
ax.add_patch(Rectangle((qx - 0.45, qy - 0.45), 0.9, 0.9, facecolor="#f0cf8a",
                       edgecolor="#9e5c00", lw=2.0))
ax.text(qx, qy, "q", ha="center", va="center", fontsize=14, color="#9e5c00")
ax.text(ox, oy, "car\nat t+1", ha="center", va="center", fontsize=11, color="#1a6fa3")
ax.add_patch(Rectangle((ox - 0.45, oy - 0.45), 0.9, 0.9, facecolor="none",
                       edgecolor="#1a6fa3", lw=2.0, linestyle=(0, (4, 3))))
ax.annotate("", xy=(ox - 0.1, oy - 0.55), xytext=(qx + 0.35, qy + 0.45),
            arrowprops=dict(arrowstyle="->", color="#b34a06", lw=1.6,
                            linestyle=(0, (4, 3))))
ax.text(4.6, 1.2, "too far:\nno overlap", fontsize=11, color="#b34a06", ha="center")

# Panel 2: decoupled spatial attention, horizontal pass
ax = axes[1]
draw_grid(ax, "Decoupled SA, pass 1\n(horizontal)")
for xx in range(W):
    ax.add_patch(Rectangle((xx - 0.45, qy - 0.45), 0.9, 0.9, facecolor="#ebe7f8",
                           edgecolor="#6a55c2", lw=1.2))
ax.add_patch(Rectangle((qx - 0.45, qy - 0.45), 0.9, 0.9, facecolor="#cdc2f0",
                       edgecolor="#6a55c2", lw=2.0))
ax.text(qx, qy, "q", ha="center", va="center", fontsize=14, color="#6a55c2")
ax.text(2.75, -1.15, "row info flows into q", fontsize=11, color="#6a55c2", ha="center")

# Panel 3: decoupled spatial attention, vertical pass
ax = axes[2]
draw_grid(ax, "Decoupled SA, pass 2\n(vertical)")
for yy in range(H):
    ax.add_patch(Rectangle((qx - 0.45, yy - 0.45), 0.9, 0.9, facecolor="#ebe7f8",
                           edgecolor="#6a55c2", lw=1.2))
ax.add_patch(Rectangle((qx - 0.45, qy - 0.45), 0.9, 0.9, facecolor="#cdc2f0",
                       edgecolor="#6a55c2", lw=2.0))
ax.text(qx, qy, "q", ha="center", va="center", fontsize=14, color="#6a55c2")
ax.add_patch(Rectangle((ox - 0.45, oy - 0.45), 0.9, 0.9, facecolor="none",
                       edgecolor="#1a6fa3", lw=2.0))
ax.text(ox, oy, "car", ha="center", va="center", fontsize=11, color="#1a6fa3")
ax.text(2.75, -1.15, "column info flows in;\nthen temporal attention can reach it", fontsize=11,
        color="#6a55c2", ha="center")

fig.suptitle("Why temporal attention alone is not enough under drastic view shift", fontsize=15, y=1.02)
fig.tight_layout()
out = "/home/hatch/workspace/goals/cross-domain-frontier-paper-reading-vla-world-models-vln-multi-agent-driving/files/paper-xray-40/world-model-driving/genad-xray/fig-decoupled-sa.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("saved", out)
