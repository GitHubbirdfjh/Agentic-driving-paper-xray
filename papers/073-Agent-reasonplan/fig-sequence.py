# -*- coding: utf-8 -*-
"""ReasonPlan 单次前向的统一序列结构：
一次 forward 按顺序生成 [BOS][BOI]{image t+3}[EOI][BOT]{reasoning}[EOT]{trajectory}[EOS]。
只讲一件事：单条序列布局（NSP 图像段 + DeCoT 文本段共用一次自回归）。
"""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
_font = [f for f in font_manager.fontManager.ttflist if f.name == "Noto Sans CJK JP"]
assert _font, "no CJK font"
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.patches import FancyBboxPatch

IN = "#1a6fa3"; OUT = "#9e5c00"; ALT = "#6a55c2"; WIRE = "#6e737b"

fig, ax = plt.subplots(figsize=(12.5, 3.4))
ax.set_xlim(0, 12.5); ax.set_ylim(0, 3.0); ax.axis("off")

ax.text(6.25, 2.62, "ReasonPlan 的一次前向回答序列（按生成顺序从左到右）", ha="center",
        fontsize=12.5, color="#141413", weight="bold")

# (label, width, fill, edge)
blocks = [
    ("[BOS]", 0.85, "#e2eff5", IN),
    ("[BOI]", 0.85, "#ebe7f8", ALT),
    ("image tokens\nt+3", 1.55, "#ebe7f8", ALT),
    ("[EOI]", 0.85, "#ebe7f8", ALT),
    ("[BOT]", 0.85, "#f8ecd2", OUT),
    ("reasoning steps\n（场景理解→交规→风险→元动作）", 2.9, "#f8ecd2", OUT),
    ("[EOT]", 0.85, "#f8ecd2", OUT),
    ("trajectory\n（6 个未来路点）", 1.55, "#f8ecd2", OUT),
    ("[EOS]", 0.85, "#f3f1ea", WIRE),
]
x = 0.25
gap = 0.12
for label, w, fill, edge in blocks:
    ax.add_patch(FancyBboxPatch((x, 1.05), w, 1.05, boxstyle="round,pad=0.03",
                                facecolor=fill, edgecolor=edge, linewidth=1.7))
    ax.text(x + w / 2, 1.05 + 0.525, label, ha="center", va="center",
            fontsize=10.5, color=edge, weight="bold", linespacing=1.4)
    x += w + gap

# 监督信号标注
ax.annotate("MSE 监督\n(潜空间)", xy=(3.02, 1.05), xytext=(3.02, 0.28),
            ha="center", fontsize=10, color=ALT,
            arrowprops=dict(arrowstyle="->", color=ALT, lw=1.8))
ax.annotate("交叉熵监督\n(文本空间)", xy=(8.9, 1.05), xytext=(8.9, 0.28),
            ha="center", fontsize=10, color=OUT,
            arrowprops=dict(arrowstyle="->", color=OUT, lw=1.8))
ax.text(6.25, 2.15, "NSP 图像预测（紫）与 DeCoT 文本推理（橙）不是两轮对话，而是一次自回归生成的两段",
        ha="center", fontsize=11, color="#3d3d3a")

ax.text(0.25, 0.06, "图：ReasonPlan 的单次前向序列布局 —— 一次 forward 同时完成场景预测与决策推理",
        fontsize=10, color="#73726c")
plt.tight_layout(pad=0.3)
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/073-Agent-reasonplan/fig-sequence.png",
            dpi=150, bbox_inches="tight")
print("saved fig-sequence.png")
