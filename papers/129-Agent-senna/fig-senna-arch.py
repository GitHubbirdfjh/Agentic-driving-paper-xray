# -*- coding: utf-8 -*-
"""Senna dual-system architecture diagram (self-drawn)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(14, 7.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 7.5); ax.axis("off")

def box(x, y, w, h, text, fc, ec="black", fs=9, sub=None):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       facecolor=fc, edgecolor=ec, linewidth=1.2)
    ax.add_patch(b)
    ax.text(x+w/2, y+h/2+(0.18 if sub else 0), text, ha="center", va="center",
            fontsize=fs, weight="bold", color="black")
    if sub:
        ax.text(x+w/2, y+h/2-0.28, sub, ha="center", va="center",
                fontsize=7.5, color="#333333")
    return b

def arrow(x1, y1, x2, y2, label=None, ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                        linewidth=1.4, color="black", linestyle=ls)
    ax.add_patch(a)
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2+0.22, label, ha="center", fontsize=7.5,
                color="#333333", style="italic")

IN, OP, OUT, ALT = "#e2eff5", "#dcf2e8", "#f8ecd2", "#ebe7f8"

# inputs
box(0.3, 5.6, 2.0, 1.0, "6-view images", IN, sub="Nimg x 224 x 224 x 3")
box(0.3, 4.2, 2.0, 1.0, "User instruction\n+ nav command", IN, sub="text tokens")

# Senna-VLM block
ax.text(3.4, 6.9, "Senna-VLM  (slow, deliberative)", fontsize=11, weight="bold", color="#0b7a5c")
box(3.2, 5.6, 2.2, 1.0, "CLIP ViT-L/14", OP, sub="576 tokens / image")
box(5.9, 5.6, 2.6, 1.0, "Driving Vision Adapter", OP, sub="learnable queries: 576 -> 128 tokens")
box(3.2, 4.2, 2.2, 1.0, "Text Encoder", OP, sub="Etxt")
box(5.9, 4.2, 2.6, 1.0, "Vicuna-v1.5-7B", OP, sub="autoregressive decode")
box(9.0, 4.9, 1.8, 1.0, "Meta-action", ALT, sub='"Decelerate, Turn Left"\n12-class discrete')
box(11.2, 4.9, 1.8, 1.0, "Meta-action\nEncoder", ALT, sub="learnable Eact -> eact")

# Senna-E2E block
ax.text(3.4, 3.4, "Senna-E2E  (fast, reactive)", fontsize=11, weight="bold", color="#9e5c00")
box(3.2, 1.6, 2.6, 1.0, "VADv2 backbone", OUT, sub="perception + motion pred.")
box(6.3, 1.6, 2.6, 1.0, "Planning tokens\n+ cross-attn", OUT, sub="conditioned on eact, enav")
box(9.4, 1.6, 2.6, 1.0, "Trajectory V", OUT, sub="V in R^{T x 2}")

arrow(2.3, 6.1, 3.2, 6.1)
arrow(2.3, 4.7, 3.2, 4.7)
arrow(5.4, 6.1, 5.9, 6.1)
arrow(5.4, 4.7, 5.9, 4.7)
arrow(8.5, 6.1, 8.5, 5.2); arrow(8.5, 5.2, 9.0, 5.2)
arrow(7.2, 4.7, 9.0, 5.4, ls=(0,(3,2)))
arrow(10.8, 5.4, 11.2, 5.4)
arrow(12.1, 4.9, 12.1, 2.6); arrow(12.1, 2.6, 8.9, 2.1, label="eact (condition)")
arrow(5.8, 1.6, 6.3, 2.1); arrow(5.8, 2.6, 6.3, 2.1)
arrow(8.9, 2.1, 9.4, 2.1)
arrow(3.2, 5.6, 3.2, 2.6, ls=(0,(3,2)))
ax.text(2.85, 4.0, "multi-view\nimages", fontsize=7.5, color="#333333", ha="center", style="italic")

ax.text(0.3, 0.5, "Core idea: never let the VLM do arithmetic. It only picks a discrete meta-action;\n"
        "the end-to-end model turns that decision into a precise trajectory.  Train with GT meta-actions, infer with predicted ones.",
        fontsize=8.5, color="#333333")
ax.text(0.3, 0.05, "Self-drawn for x-ray 129. Token counts from paper Table VI / Sec III-B.", fontsize=7, color="#888888")
plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/129-Agent-senna/fig-senna-arch.png", dpi=150)
print("saved")
