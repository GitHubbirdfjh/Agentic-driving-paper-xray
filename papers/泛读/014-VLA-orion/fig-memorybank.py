# -*- coding: utf-8 -*-
"""QT-Former 的长期记忆 bank: FIFO 滚动示意。

M ∈ R^{(Nh×n)×Cq}: Nh=16 个 history query × n=16 帧。
每帧: Qh 先从 M 里按相对时间戳取历史, 再与当前场景 Qs 交叉注意力,
得到更新后的 Qh_hat, 按 FIFO 写回 M(顶端丢弃最老的一帧)。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax in axes:
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")

def draw_bank(ax, title, highlight_new=False):
    ax.text(5, 9.3, title, ha="center", fontsize=12, weight="bold")
    # 16 帧, 每帧一个格子
    for i in range(16):
        y = 8.2 - i * 0.48
        color = "#f0cf8a" if (highlight_new and i == 15) else "#e2eff5"
        ec = "#9e5c00" if (highlight_new and i == 15) else "#1a6fa3"
        ax.add_patch(FancyBboxPatch((3.2, y - 0.36), 3.6, 0.42,
                                    boxstyle="round,pad=0.02",
                                    fc=color, ec=ec, lw=1.2))
        if i in (0, 15):
            label = "t-15 (最老)" if i == 0 else ("t (最新, 刚写入)" if highlight_new else "t-0")
            ax.text(7.1, y - 0.15, label, fontsize=9, color="#3d3d3a", va="center")
        elif i == 1:
            ax.text(5, y - 0.15, "⋮", fontsize=12, ha="center", color="#6e737b")
    ax.text(5, 0.35, "$M$: 16 帧 × 每帧 16 个 $Q_h$\n共 256 个向量, 每帧覆盖写", ha="center",
            fontsize=9.5, color="#3d3d3a",
            bbox=dict(boxstyle="round", fc="#f5f4ed", ec="#deddd8"))

draw_bank(axes[0], "t 时刻写入前")
draw_bank(axes[1], "t 时刻写入后 (FIFO)", highlight_new=True)

# 箭头: 丢弃最老, 整体上移
axes[1].annotate("", xy=(3.0, 8.0), xytext=(3.0, 7.4),
                 arrowprops=dict(arrowstyle="->", color="#b34a06", lw=2))
axes[1].text(2.2, 7.7, "丢弃\n最老帧", fontsize=9, color="#b34a06", ha="center")

fig.suptitle("QT-Former 长期记忆 bank 的 FIFO 滚动: $M=[\\hat{Q}^{t-n}_h,\\dots,\\hat{Q}^{t-1}_h,\\hat{Q}^t_h]$",
             fontsize=14)
fig.text(0.5, -0.02,
         "每帧两步: ① $Q_h = CA(Q_h, M+P_t, M+P_t)$ —— 用 history query 从 bank 里按相对时间戳捞历史;\n"
         "② $\\hat{Q}_h = CA(Q_h, Q_s, Q_s)$ —— 再与当前场景特征 $Q_s$ 交叉, 捞出与当前最相关的历史, 写回 bank。",
         ha="center", fontsize=10.5, color="#3d3d3a",
         bbox=dict(boxstyle="round", fc="#f5f4ed", ec="#deddd8"))
plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/014-VLA-orion/fig-memorybank.png",
            dpi=150, bbox_inches="tight")
print("saved fig-memorybank.png")
