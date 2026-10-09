"""RSSM vs MSSM: why the 1D latent bottleneck loses BEV context.
Saves fig-bottleneck.png next to this script. Light theme, cm mathtext."""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size": 11,
})
import matplotlib.pyplot as plt
import matplotlib.patches as patches

IN = "#1a6fa3"; OUT = "#9e5c00"; OP = "#0b7a5c"; ALT = "#6a55c2"
WIRE = "#6e737b"; BGF = {"facecolor": "#faf9f5", "edgecolor": "none"}

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), facecolor="#faf9f5")
for ax, title in zip(axes, ["(a) RSSM: 1D latent squeezes everything",
                            "(b) MSSM: dynamics / statics take two roads"]):
    ax.set_facecolor("#faf9f5")
    ax.set_xlim(0, 100); ax.set_ylim(0, 60)
    ax.axis("off")
    ax.set_title(title, fontsize=12, color="#141413", pad=10)

def box(ax, x, y, w, h, text, color, fs=10):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=2.5",
                 facecolor=color + "22", edgecolor=color, linewidth=1.5))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", color=color, fontsize=fs,
            style="italic" if text.startswith("$") else "normal")

def arrow(ax, x1, y1, x2, y2, color=WIRE):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, shrinkA=2, shrinkB=2))

# ---------------- (a) RSSM ----------------
ax = axes[0]
box(ax, 4, 25, 20, 12, "$b_t$\nBEV 200x200x256", IN)
arrow(ax, 24, 31, 34, 31)
box(ax, 34, 27, 24, 8, "encoder -> 1D", OP, fs=9)
arrow(ax, 58, 31, 66, 31)
box(ax, 66, 25, 12, 12, "$s_t$\n1D vector", ALT)
arrow(ax, 78, 31, 86, 31)
box(ax, 86, 25, 12, 12, "decoder", OP, fs=9)
# cross-out mark over the 1D vector to show loss of context
ax.plot([66.5, 77.5], [36.5, 24.5], color="#b34a06", lw=2.2)
ax.plot([66.5, 77.5], [24.5, 36.5], color="#b34a06", lw=2.2)
ax.text(72, 8, "spatial layout of roads,\nlanes, distant objects: lost", ha="center",
        va="center", color="#b34a06", fontsize=9.5)
ax.text(50, 48, "Table 1: RSSM-only pre-training  mAP 0.381  <  no-pretrain 0.416",
        ha="center", color="#3d3d3a", fontsize=9.5)

# ---------------- (b) MSSM ----------------
ax = axes[1]
box(ax, 2, 25, 16, 12, "$b_t$", IN)
arrow(ax, 18, 33, 25, 33)          # up to static path
arrow(ax, 18, 29, 25, 29)          # down to dynamic path
box(ax, 25, 33, 26, 10, "random frame $b'$\n$\\hat b=z_\\theta(b')$", OP, fs=8.5)
box(ax, 25, 19, 26, 10, "$x_t\\in\\mathbb{R}^{512}$ -> $s_t$ (1D)", ALT, fs=8.5)
arrow(ax, 51, 38, 60, 38)
arrow(ax, 51, 24, 60, 24)
box(ax, 60, 36, 20, 6, "static context", OP, fs=8.5)
box(ax, 60, 21, 20, 6, "dynamic state\n+ memory $h_{1:t}$", ALT, fs=8)
arrow(ax, 80, 39, 86, 39); arrow(ax, 80, 24, 86, 24)
box(ax, 86, 25, 13, 14, "$\\hat y_t=\\,l_\\theta(m_\\theta(\\tilde h_t,s_t),\\,\\hat b)$", OUT, fs=8.5)
ax.text(35, 48, "BEV spatial layout preserved on the static road",
        ha="center", color="#0b7a5c", fontsize=9.5)
ax.text(63, 8, "channel-wise concat: decoder sees both", ha="center",
        color="#3d3d3a", fontsize=9.5)

plt.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-40/world-model-driving/driveworld-xray/fig-bottleneck.png"
plt.savefig(out, dpi=170, bbox_inches="tight", facecolor="#faf9f5")
print("saved", out)
