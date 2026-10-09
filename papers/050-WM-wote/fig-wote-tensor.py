# -*- coding: utf-8 -*-
"""自绘图：WoTE 的 BEV 世界模型一次递推步内的张量形状变化。

展示式 (2) 的核心运算：B_t (h×w×c, h=w=8) 展平成 64 个向量，与动作嵌入 a_i 拼接成 65×c，
过一个 TransformerEncoder，输出重新分成未来 BEV 状态 B_{t+1}^i (8×8×c) 与未来动作 a_{t+1}^i。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(figsize=(15, 4.6))
ax.set_xlim(0, 15); ax.set_ylim(0, 6); ax.axis("off")
ax.set_title("WoTE world model: one recurrent step, tensor shapes (h=w=8, c=256)", fontsize=12, weight="bold", pad=12)

IN = "#1a6fa3"; IN_FILL = "#e2eff5"
OP = "#0b7a5c"; OP_FILL = "#dcf2e8"
OUT = "#9e5c00"; OUT_FILL = "#f8ecd2"
WIRE = "#6e737b"

def block(ax, x, y, w, h, fc, ec, label, sub=""):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1.6)
    ax.add_patch(b)
    ax.text(x+w/2, y+h/2+0.18, label, ha="center", va="center", fontsize=10.5)
    if sub:
        ax.text(x+w/2, y+h/2-0.45, sub, ha="center", va="center", fontsize=9, color=WIRE)

# 画 8x8 网格示意 BEV
def grid(ax, x, y, s=1.6, n=8, ec=IN):
    g = s / n
    for i in range(n+1):
        ax.plot([x, x+s], [y+i*g, y+i*g], color=ec, lw=0.6)
        ax.plot([x+i*g, x+i*g], [y, y+s], color=ec, lw=0.6)

def arr(p1, p2, c=WIRE):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, lw=1.5, color=c))

# B_t
ax.text(1.6, 5.2, "$B_t$: future? no — current", fontsize=9, color=WIRE, ha="center")
grid(ax, 0.8, 2.8, s=1.6)
ax.text(1.6, 2.2, "$B_t \\in \\mathbb{R}^{8\\times8\\times256}$", fontsize=10.5, color=IN, ha="center")
ax.text(1.6, 1.75, "current BEV state\n64 spatial vectors", fontsize=8.5, color=WIRE, ha="center")

# a_i
arr((2.5, 3.6), (3.4, 3.6))
block(ax, 3.6, 2.9, 1.4, 1.4, IN_FILL, IN, "$a_t^i$", "$\\in \\mathbb{R}^{256}$")
ax.text(4.3, 2.2, "action embedding\n(trajectory encoded)", fontsize=8.5, color=WIRE, ha="center")

# concat -> 65 x 256
arr((5.0, 3.6), (5.9, 3.6))
block(ax, 6.0, 2.75, 2.0, 1.7, "#f8f4cf", "#7a6b00", "$F_i \\in \\mathbb{R}^{65\\times256}$", "flatten + concat")
ax.text(7.0, 2.1, "64 BEV tokens\n+ 1 action token", fontsize=8.5, color=WIRE, ha="center")

# transformer
arr((8.0, 3.6), (8.9, 3.6))
block(ax, 9.0, 2.75, 2.4, 1.7, OP_FILL, OP, "TransformerEncoder", "2 layers, 8 heads\nd_model=256 (code)")
ax.text(10.2, 2.1, "$F_{i+1} = \\mathrm{Trans}(F_i)$", fontsize=9.5, color=WIRE, ha="center")

# split output
arr((11.4, 3.6), (12.3, 3.6))
ax.text(13.4, 5.2, "future state + future action", fontsize=9, color=WIRE, ha="center")
grid(ax, 12.35, 3.4, s=1.3)
ax.text(13.0, 2.9, "$B_{t+1}^i$", fontsize=10.5, color=OUT, ha="center")
block(ax, 13.9, 3.3, 0.9, 1.0, OUT_FILL, OUT, "$a_{t+1}^i$", "")
ax.text(14.35, 2.9, "$\\in \\mathbb{R}^{256}$", fontsize=9, color=WIRE, ha="center")

ax.text(7.5, 0.9, "recurrent:  $(B_{t+K}^i,\\,a_{t+K}^i)=\\mathrm{WorldModel}(B_{t+K-1}^i,\\,a_{t+K-1}^i)$  — done for all N trajectories in parallel",
        fontsize=10, ha="center", color=WIRE, style="italic")

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/050-WM-wote/fig-wote-tensor.png", dpi=150)
print("saved fig-wote-tensor.png")
