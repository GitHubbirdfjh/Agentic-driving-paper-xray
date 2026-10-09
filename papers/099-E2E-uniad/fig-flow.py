# -*- coding: utf-8 -*-
"""UniAD query-flow diagram: modules as boxes, arrows labeled with tensor shapes."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(13, 7.5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7.5)
ax.axis("off")

IN, OP, OUT, WIRE = "#1a6fa3", "#0b7a5c", "#9e5c00", "#6e737b"
IN_F, OP_F, OUT_F = "#e2eff5", "#dcf2e8", "#f8ecd2"

def box(x, y, w, h, title, lines, fc, ec, tsize=11):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       facecolor=fc, edgecolor=ec, linewidth=1.6)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center",
            fontsize=tsize, fontweight="bold", color="#141413")
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.72 - i * 0.34, ln, ha="center", va="center",
                fontsize=8.6, color="#3d3d3a", family="monospace")

def arrow(x1, y1, x2, y2, label, dy=0.0, color=WIRE, fsize=8.6):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=13, linewidth=1.4, color=color,
                        connectionstyle="arc3,rad=0.0")
    ax.add_patch(a)
    ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, ha="center", va="center",
            fontsize=fsize, color=color, family="monospace",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=color, linewidth=0.8))

# layout: two rows
# row 1 (top): cameras -> BEV -> TrackFormer -> MotionFormer -> Planner
# row 2 (bot): MapFormer -> OccFormer (feeds planner)
box(0.2, 5.3, 1.9, 1.5, "6 cameras", ["images"], IN_F, IN)
box(2.6, 5.3, 1.9, 1.5, "BEV encoder", ["B: 200x200x256"], IN_F, IN)
box(5.0, 5.3, 2.1, 1.5, "TrackFormer", ["QA: Na x 256", "ego query"], OP_F, OP)
box(7.6, 5.3, 2.3, 1.5, "MotionFormer", ["x-hat: Na x 12 x 2", "K=6 modes"], OP_F, OP)
box(10.4, 5.3, 2.4, 1.5, "Planner", ["tau-hat: 6 x 2", "Newton opt."], OUT_F, OUT)

box(2.6, 2.2, 1.9, 1.5, "MapFormer", ["QM: 300 x 256"], OP_F, OP)
box(5.0, 2.2, 2.1, 1.5, "OccFormer", ["O-hat: Na x 200 x 200", "To=5 blocks"], OP_F, OP)

arrow(2.1, 6.05, 2.6, 6.05, "")
arrow(4.5, 6.05, 5.0, 6.05, "")
arrow(7.1, 6.05, 7.6, 6.05, "QA", dy=0.22)
arrow(9.9, 6.05, 10.4, 6.05, "ego queries", dy=0.22)
arrow(3.55, 5.3, 3.55, 3.7, "B", dy=0.0)
arrow(6.05, 5.3, 6.05, 3.7, "QA, QX", dy=0.0)
arrow(4.5, 2.95, 5.0, 2.95, "QM", dy=0.22)
arrow(7.1, 2.95, 7.6, 2.55, "O-hat (merged)", dy=-0.28, color=OUT)
arrow(11.6, 5.3, 11.6, 4.4, "", color=OUT)
ax.text(11.6, 4.05, "tau* : 6 x 2\nplanned waypoints", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color=OUT, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=OUT_F, edgecolor=OUT))

ax.text(0.2, 1.2, "command embedding: 3 discrete cmds (left / right / forward)",
        fontsize=9, color=WIRE, style="italic")
ax.text(0.2, 0.7, "All inter-module currency: 256-d query vectors (soft interface, no hard boxes).",
        fontsize=9.5, color="#141413", weight="bold")

ax.set_title("UniAD: one forward pass from 6 cameras to a planned trajectory",
             fontsize=13, fontweight="bold", color="#141413", pad=12)

out = "/home/hatch/workspace/your_files/paper-xray-40/e2e-mainstream/uniad-xray/fig-flow.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#faf9f5")
print("saved", out)
