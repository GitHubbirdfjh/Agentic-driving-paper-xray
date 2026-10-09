# -*- coding: utf-8 -*-
"""RDA-Driver Fig A: the reasoning-decision misalignment as four quadrants.

The paper's Fig 1(a) shows that a vanilla-finetuned LLaVA has no correlation
between CoT score and decision error. This figure makes the same point as a
quadrant map: x = decision (trajectory L2 error), y = reasoning (CoT score).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(8.5, 6.2))

ax.set_xlim(0, 3.0)
ax.set_ylim(40, 100)
ax.set_xlabel("Decision error: trajectory L2 (m)  -> lower is better")
ax.set_ylabel("Reasoning score: CoT score (0-100)  -> higher is better")

# quadrant shading
ax.axhspan(40, 70, 0, 0.5, color="#f9e3d6", alpha=0.6)   # bad-bad
ax.axhspan(70, 100, 0, 0.5, color="#f1bd9c", alpha=0.55)  # good-reasoning bad-decision
ax.axhspan(40, 70, 0.5, 1, color="#e2eff5", alpha=0.6)    # bad-reasoning good-decision
ax.axhspan(70, 100, 0.5, 1, color="#dcf2e8", alpha=0.65)  # good-good

# reference point: what alignment would look like
xs = np.linspace(0.2, 2.8, 50)
ax.plot(xs, 95 - 15 * xs, color="#6e737b", linestyle="--", linewidth=1.4,
        label="what alignment would look like (negative correlation)")

rng = np.random.default_rng(7)
# misaligned cloud: no correlation, matches paper's Fig 1(a) observation
x_cloud = rng.uniform(0.3, 2.8, 60)
y_cloud = rng.uniform(45, 95, 60)
ax.scatter(x_cloud, y_cloud, s=26, color="#7a6b00", alpha=0.55, zorder=3,
           label="vanilla-finetuned model samples (no correlation)")

# the notorious case from Fig 1(b): right reasoning, wrong decision
ax.scatter([0.55], [88], s=200, color="#b34a06", edgecolors="black", zorder=5)
ax.annotate("Fig 1(b): reasoning right,\ndecision wrong", xy=(0.55, 88),
            xytext=(1.45, 90), fontsize=10,
            arrowprops=dict(arrowstyle="->", color="black"))

ax.axvline(1.5, color="#6e737b", linewidth=0.8)
ax.axhline(70, color="#6e737b", linewidth=0.8)

ax.text(0.06, 0.94, "Q1: right reasoning\nbut wrong decision\n<- the misalignment", transform=ax.transAxes,
        fontsize=10, va="top", color="#b34a06",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85))
ax.text(0.68, 0.94, "Q2: reasoning and\ndecision agree\n<- the goal", transform=ax.transAxes,
        fontsize=10, va="top", color="#0b7a5c",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85))
ax.text(0.68, 0.30, "Q3: right decision\nfor wrong reasons\n(lucky)", transform=ax.transAxes,
        fontsize=10, va="top", color="#1a6fa3",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85))
ax.text(0.06, 0.30, "Q4: both wrong", transform=ax.transAxes,
        fontsize=10, va="top", color="#73726c",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85))

ax.legend(loc="center left", fontsize=9)
ax.set_title("RDA-Driver's diagnosis: reasoning score and decision error do not move together",
             fontsize=12, pad=12)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/017-VLA-rda-driver/fig-misalign-quadrant.png",
            dpi=150)
print("saved fig-misalign-quadrant.png")
