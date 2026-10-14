"""双向目标：p(A|V,L) 生成 vs p(L|V,A) 理解。
同一批 (V,L,A) 数据，换一下拼接顺序，同一个 decoder 做两个方向。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.font_manager import fontManager

fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Serif CJK JP", "DejaVu Sans"]

IN = "#1a6fa3"
OUT = "#9e5c00"
OP = "#0b7a5c"
ALT = "#6a55c2"
WIRE = "#6e737b"

fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")

for k, (title, seq, target, tcolor) in enumerate([
    ("(a) 生成 generation：$p(A|V,L)$",
     ["V", "L", "A"], "A", OUT),
    ("(b) 理解 understanding：$p(L|V,A)$",
     ["V", "A", "L"], "L", IN),
]):
    ax = axes[k]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    # 三个 token 块
    colors = {"V": IN, "L": IN, "A": OUT}
    for i, s in enumerate(seq):
        x = 1.5 + i * 2.6
        hl = (s == target)
        ax.add_patch(mpatches.FancyBboxPatch((x, 5.5), 2.0, 1.6,
                                            boxstyle="round,pad=0.08",
                                            fc=colors[s], ec=colors[s],
                                            alpha=0.35 if hl else 0.15,
                                            lw=2.4 if hl else 1.2))
        ax.text(x + 1.0, 6.3, s, ha="center", va="center", fontsize=16,
                color=colors[s], weight="bold" if hl else "normal")
        lab = {"V": "图像", "L": "\"向左变道\"", "A": "轨迹词串"}[s]
        ax.text(x + 1.0, 5.0, lab, ha="center", va="center", fontsize=9,
                color=colors[s])
    # decoder
    ax.add_patch(mpatches.FancyBboxPatch((3.4, 2.6), 3.2, 1.5,
                                        boxstyle="round,pad=0.1",
                                        fc=OP, ec=OP, alpha=0.18))
    ax.text(5, 3.35, "same decoder", ha="center", color=OP, fontsize=10)
    ax.annotate("", xy=(5, 4.1), xytext=(5, 5.5),
                arrowprops=dict(arrowstyle="->", color=WIRE, lw=1.4))
    # 监督目标
    ax.text(5, 1.7, f"监督：{target}", ha="center", color=tcolor, fontsize=11,
            weight="bold")
    loss = (r"$\mathcal{L}_{generation}$" if k == 0
            else r"$\mathcal{L}_{understanding}$")
    ax.text(5, 0.9, loss, ha="center", color=tcolor, fontsize=11)

axes[0].text(5, 8.6, "输入 [V, L]，预测 A", ha="center", color=WIRE, fontsize=10)
axes[1].text(5, 8.6, "输入 [V, A]，把轨迹“说”回去", ha="center", color=WIRE,
             fontsize=10)

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/088-Agent-linkvla/fig-bidirectional.png",
            dpi=150)
print("saved fig-bidirectional.png")
