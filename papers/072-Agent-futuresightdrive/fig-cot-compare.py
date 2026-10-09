# -*- coding: utf-8 -*-
"""fig-cot-compare.py: three CoT styles for driving VLA.
Text CoT: symbolic compression loses spatio-temporal relations.
Image-text CoT: modalities inconsistent.
Visual spatio-temporal CoT (FSDrive): one unified future frame carries both
spatial priors (red lane dividers + 3D boxes) and temporal evolution.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

WIRE = "#6e737b"
LOSS, LOSS_F = "#b34a06", "#f9e3d6"
IN, IN_F = "#1a6fa3", "#e2eff5"
OP, OP_F = "#0b7a5c", "#dcf2e8"
ALT, ALT_F = "#6a55c2", "#ebe7f8"

fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
for ax in axes:
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

def panel(ax, title, color):
    ax.text(5, 9.3, title, fontsize=10.5, weight="bold", ha="center", color=color)

# ---- (a) text CoT ----
ax = axes[0]
panel(ax, "(a) Text CoT: symbolic compression", LOSS)
r = mpatches.FancyBboxPatch((1, 6.4), 8, 1.6, boxstyle="round,pad=0.06",
                            facecolor=LOSS_F, edgecolor=LOSS, linewidth=1.5)
ax.add_patch(r)
ax.text(5, 7.2, '"pedestrian at (3.2, 8.1),\ncar ahead slowing down"', ha="center", va="center",
        fontsize=9, style="italic", color=LOSS)
ax.annotate("", xy=(5, 5.9), xytext=(5, 6.4), arrowprops=dict(arrowstyle="->", color=WIRE, lw=1.6))
r2 = mpatches.FancyBboxPatch((2.5, 3.4), 5, 2.0, boxstyle="round,pad=0.06",
                             facecolor="white", edgecolor=WIRE, linewidth=1.3)
ax.add_patch(r2)
ax.text(5, 4.4, "trajectory\n(w1..wn)", ha="center", va="center", fontsize=9.5, color=WIRE)
ax.text(5, 2.4, "lost: exact geometry,\nrelative motion,\ntexture cues", ha="center", fontsize=8.5, color=LOSS)
ax.text(5, 1.2, "modality gap:\nplan in text,\ndrive in pixels", ha="center", fontsize=8.5, color=LOSS, style="italic")

# ---- (b) image-text CoT ----
ax = axes[1]
panel(ax, "(b) Image-text CoT: inconsistent modalities", IN)
r = mpatches.FancyBboxPatch((0.6, 6.4), 4.2, 1.6, boxstyle="round,pad=0.06",
                            facecolor=IN_F, edgecolor=IN, linewidth=1.5)
ax.add_patch(r)
ax.text(2.7, 7.2, "future\nframe", ha="center", va="center", fontsize=9, color=IN, weight="bold")
r = mpatches.FancyBboxPatch((5.4, 6.4), 4.0, 1.6, boxstyle="round,pad=0.06",
                            facecolor=LOSS_F, edgecolor=LOSS, linewidth=1.5)
ax.add_patch(r)
ax.text(7.4, 7.2, '"3 boxes:\ncar, ped..."', ha="center", va="center", fontsize=8.5,
        style="italic", color=LOSS)
ax.annotate("", xy=(5, 5.9), xytext=(5, 6.4), arrowprops=dict(arrowstyle="->", color=WIRE, lw=1.6))
r2 = mpatches.FancyBboxPatch((2.5, 3.4), 5, 2.0, boxstyle="round,pad=0.06",
                             facecolor="white", edgecolor=WIRE, linewidth=1.3)
ax.add_patch(r2)
ax.text(5, 4.4, "trajectory\n(w1..wn)", ha="center", va="center", fontsize=9.5, color=WIRE)
ax.text(5, 2.4, "frame says pixels,\ntext says symbols:\nplanner must bridge\ntwo languages", ha="center", fontsize=8.5, color=IN)

# ---- (c) visual spatio-temporal CoT ----
ax = axes[2]
panel(ax, "(c) FSDrive: one unified future frame", OP)
# draw a schematic future frame: road + red dividers + red 3D boxes
fr = mpatches.FancyBboxPatch((1.5, 5.6), 7, 2.6, boxstyle="round,pad=0.06",
                             facecolor="#dfe9f2", edgecolor=OP, linewidth=1.8)
ax.add_patch(fr)
# road trapezoid
road = plt.Polygon([[3.2, 5.6], [6.8, 5.6], [6.0, 8.2], [4.0, 8.2]], closed=True,
                   facecolor="#9aa7b5", edgecolor="none")
ax.add_patch(road)
# red lane dividers
for x0, x1 in [(4.35, 4.75), (5.25, 5.65)]:
    ax.plot([x0, x1 - 0.15], [5.7, 8.1], color="red", lw=2.2)
# red 3D boxes (schematic rectangles)
for (bx, by, bw, bh) in [(5.9, 6.4, 1.1, 0.7), (2.4, 6.9, 0.9, 0.6)]:
    br = mpatches.Rectangle((bx, by), bw, bh, facecolor="none", edgecolor="red", linewidth=2)
    ax.add_patch(br)
ax.text(5, 5.0, "unified future frame Q_CoT", ha="center", fontsize=8.5, color=OP, weight="bold")
ax.annotate("", xy=(5, 4.5), xytext=(5, 5.0), arrowprops=dict(arrowstyle="->", color=WIRE, lw=1.6))
r2 = mpatches.FancyBboxPatch((2.5, 2.2), 5, 1.8, boxstyle="round,pad=0.06",
                             facecolor="white", edgecolor=OP, linewidth=1.5)
ax.add_patch(r2)
ax.text(5, 3.1, "trajectory\n(w1..wn)", ha="center", va="center", fontsize=9.5, color=OP)
ax.text(5, 1.3, "spatial: red dividers + 3D boxes\ntemporal: the frame itself evolves",
        ha="center", fontsize=8.5, color=OP)

fig.suptitle("Three ways for a driving VLA to 'think before acting'", fontsize=12.5, weight="bold", y=1.02)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/072-Agent-futuresightdrive/fig-cot-compare.png",
            dpi=150, bbox_inches="tight")
print("saved")
