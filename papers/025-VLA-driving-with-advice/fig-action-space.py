# -*- coding: utf-8 -*-
"""Discrete Action Space (DAS) grid: Adir x Aspd = 5 x 4 = 20 motion primitives.
Highlights the worked example <Straight, Stop> used in the writeup."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dirs = ["Straight", "Left", "Right", "Lane\nChange\nLeft", "Lane\nChange\nRight"]
spds = ["Keep", "Accelerate", "Decelerate", "Stop"]

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 5)
ax.set_ylim(0, 4)
ax.set_xticks([i + 0.5 for i in range(5)], dirs, fontsize=11)
ax.set_yticks([i + 0.5 for i in range(4)], list(reversed(spds)), fontsize=11)
ax.set_xlabel("Direction primitive  Adir (5)", fontsize=12)
ax.set_ylabel("Speed primitive  Aspd (4)", fontsize=12)
ax.set_title("Discrete Action Space  A = Adir x Aspd  (20 motion primitives)", fontsize=13, pad=12)

for r in range(4):
    for c in range(5):
        is_target = (c == 0 and r == 0)  # <Straight, Stop> is bottom-left (row reversed)
        face = "#f0cf8a" if is_target else "#eef3f7"
        edge = "#9e5c00" if is_target else "#b9c4cc"
        rect = plt.Rectangle((c, r), 1, 1, facecolor=face, edgecolor=edge, linewidth=2 if is_target else 1)
        ax.add_patch(rect)
        spd = list(reversed(spds))[r]
        ax.text(c + 0.5, r + 0.5, f"<{dirs[c].replace(chr(10),' ')}, {spd}>",
                ha="center", va="center", fontsize=9,
                weight="bold" if is_target else "normal",
                color="#7a4a00" if is_target else "#33424e")

ax.text(0.5, -0.55, "Worked example: expert a* = <Straight, Stop>  ->  BAR = 1 only on exact match",
        ha="left", va="top", fontsize=10, color="#555")
ax.text(0.5, -0.85, "A = {Straight, Left, Right, LC, RC} x {Keep, Acc, Dec, Stop}   (paper Eq. 4)",
        ha="left", va="top", fontsize=10, color="#555", family="monospace")
plt.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/025-VLA-driving-with-advice/fig-action-space.png"
fig.savefig(out, dpi=150)
print("saved", out)
