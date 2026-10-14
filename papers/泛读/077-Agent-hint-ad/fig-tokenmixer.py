"""Holistic token mixer: shapes at every step for a concrete frame (N_det=4, T_p=6)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 8

IN, OUT, OP, ALT, WIRE = "#1a6fa3", "#9e5c00", "#0b7a5c", "#6a55c2", "#6e737b"

fig, ax = plt.subplots(figsize=(13, 5.6))
ax.set_xlim(0, 13)
ax.set_ylim(0, 5.6)
ax.axis("off")

def box(x, y, w, h, label, sub, fc, ec, fs=8.5):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", fc=fc, ec=ec, lw=1.3)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2 + 0.16, label, ha="center", va="center",
            fontsize=fs, weight="bold", color=ec)
    ax.text(x + w/2, y + h/2 - 0.32, sub, ha="center", va="center",
            fontsize=7.2, color=WIRE)
    return b

def arrow(x1, y1, x2, y2, label=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.2, color=WIRE))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.12, label, ha="center", fontsize=7,
                color=WIRE, style="italic")

ax.text(6.5, 5.25, "Holistic token mixer: AD tokens -> LLM-readable context tokens",
        fontsize=11.5, weight="bold", ha="center")
ax.text(6.5, 4.92, "concrete frame: N_det = 4 objects, T_p = 6 planning steps  (paper dims; code uses 256)",
        fontsize=8, color=WIRE, ha="center", style="italic")

y0 = 3.55
# BEV path
box(0.2, y0, 2.2, 1.05, "F_bev", "200x200x256", "#e2eff5", IN)
arrow(2.4, y0+0.5, 2.9, y0+0.5, "conv enc")
box(2.9, y0, 2.0, 1.05, "3x3 -> P_bev", "9 x 728", "#e2eff5", IN)
arrow(4.9, y0+0.5, 5.4, y0+0.5, "MHSA x8")
box(5.4, y0, 2.2, 1.05, "F_bev''", "9 x 728", "#dcf2e8", OP)

# instance path
y1 = 2.1
box(0.2, y1, 1.55, 1.05, "F_track", "4 x 256", "#e2eff5", IN)
box(1.85, y1, 1.55, 1.05, "F_motion", "4 x 256", "#e2eff5", IN)
arrow(3.4, y1+0.5, 3.8, y1+0.5, "concat")
box(3.8, y1, 2.0, 1.05, "instance mixer", "MLP: 512 -> 728", "#ebe7f8", ALT)
box(5.9, y1, 1.7, 1.05, "F_instance", "4 x 728", "#ebe7f8", ALT)
arrow(7.6, y1+0.5, 8.0, y1+0.5)
box(8.0, y1, 2.6, 1.05, "instance blocks", "5 learned Q x-attend 4\nMHCA x5 -> 5 x 728", "#dcf2e8", OP)

# ego + planning
y2 = 0.65
box(0.2, y2, 2.2, 1.05, "F_instance^ego", "1 x 728", "#e2eff5", IN)
box(2.9, y2, 1.9, 1.05, "F_plan", "6 x 2 (x, y)", "#e2eff5", IN)
arrow(4.8, y2+0.5, 5.2, y2+0.5, "PE+MLP")
box(5.2, y2, 2.0, 1.05, "F_plan'", "6 x 728", "#e2eff5", IN)

# context concat
box(8.0, 0.35, 4.6, 1.65, "F_context = concat(...)", "9 + 5 + 1 + 6 = 21 tokens x 728\n-> project to LLaMA dim",
    "#f8ecd2", OUT, fs=9)
arrow(6.5, y0+0.15, 9.2, 1.6)
arrow(9.3, y1+0.15, 9.6, 1.7)
arrow(3.0, y2+0.9, 8.6, 1.6)
arrow(6.2, y2+0.9, 8.9, 1.6)

plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/077-Agent-hint-ad/fig-tokenmixer.png",
            dpi=150, bbox_inches="tight")
print("saved fig-tokenmixer.png")
