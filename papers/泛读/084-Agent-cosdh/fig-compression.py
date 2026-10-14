# -*- coding: utf-8 -*-
"""附录表 A：不同压缩-融合方案在 OPV2V 上的精度与带宽（数字全部可核实）。
SF=单尺度融合，MF/SC=多尺度融合+单尺度压缩（CoAlign 式），MF/MC=多尺度压缩+融合（CoSDH）。
结论：MF/MC 在两种压缩器下都是精度最高、带宽最低。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# 表 A 数字：AP@0.7 / BD(Mbps)
schemes = ["SF", "MF/SC", "MF/MC"]
ae_ap = [91.18, 92.31, 93.00]
ae_bd = [76.2, 76.2, 33.4]
cb_ap = [84.32, 87.40, 88.17]
cb_bd = [0.4, 1.0, 0.4]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
x = np.arange(3)
w = 0.38

for ax, ap, bd, title in zip(axes,
                             [ae_ap, cb_ap], [ae_bd, cb_bd],
                             ["自编码器压缩（c0=16）", "Codebook 压缩 [2]"]):
    bars = ax.bar(x, ap, width=w * 2, color=["#b9c2cc", "#e8b96a", "#0b7a5c"],
                  edgecolor="#37474f", linewidth=1.2)
    for i, (v, b) in enumerate(zip(ap, bd)):
        ax.text(i, v + 0.25, f"{v:.2f}%\n{b} Mbps", ha="center", fontsize=10,
                color="#0b5345" if i == 2 else "#333333",
                fontweight="bold" if i == 2 else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels(schemes, fontsize=11)
    ax.set_ylabel("AP@0.7 (OPV2V)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylim(80, 96)

axes[0].text(1, 81.2,
             "MF/MC：每层尺度独立压缩\n"
             "传的是小特征图（H/2、H/4、H/8）\n→ 带宽 76.2→33.4 Mbps",
             ha="center", fontsize=9.5, color="#333333",
             bbox=dict(boxstyle="round,pad=0.4", fc="#faf9f5", ec="#c9c6bb"))
axes[1].text(1, 81.2,
             "MF/MC 精度反超 MF/SC：\n"
             "单尺度压缩大特征图再解压，\n信息损失更大",
             ha="center", fontsize=9.5, color="#333333",
             bbox=dict(boxstyle="round,pad=0.4", fc="#faf9f5", ec="#c9c6bb"))

fig.suptitle("多尺度压缩+融合（MF/MC）：精度更高、带宽反而更低", fontsize=13,
             fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/084-Agent-cosdh/fig-compression.png",
            dpi=150, bbox_inches="tight")
print("saved fig-compression.png")
