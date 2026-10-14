# -*- coding: utf-8 -*-
"""World4Drive 的 World Model Selector：训练用 argmin 距离定 j，推理该用 ScoreNet（论文），
但代码推理时用的却是 argmin(组合损失)。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from matplotlib.font_manager import FontProperties
_ZH = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
plt.rcParams["font.family"] = _ZH.get_name()
plt.rcParams["font.size"] = 10.5
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(11, 5.6))
ax.set_xlim(0, 11)
ax.set_ylim(0, 6.2)
ax.axis("off")

K = 4  # 示意用 4 个意图分支（论文 K=6）
dist = [0.9, 0.35, 1.25, 0.62]   # 各分支预测潜变量与真实潜变量的距离（示意）
scores = [0.18, 0.46, 0.11, 0.25]  # ScoreNet 输出（示意）

# 左侧：K 个分支
ax.text(0.7, 5.6, "K 个意图分支", fontsize=12, fontweight="bold")
for k in range(K):
    y = 4.7 - k * 1.0
    box = mpatches.FancyBboxPatch((0.3, y - 0.32), 2.2, 0.64,
                                  boxstyle="round,pad=0.03",
                                  fc="#ebe7f8" if k != 1 else "#cdc2f0",
                                  ec="#6a55c2", lw=1.6 if k == 1 else 1.0)
    ax.add_patch(box)
    ax.text(1.4, y, f"意图 {k+1}：轨迹 T$^{{{k+1}}}$\n预测潜变量 L$^{{{k+1}}}_{{t+n}}$",
            ha="center", va="center", fontsize=9.5)

# 中：距离计算
ax.text(4.35, 5.6, "与真实潜变量 L̂$_{t+n}$ 的距离", fontsize=12, fontweight="bold")
for k in range(K):
    y = 4.7 - k * 1.0
    d = dist[k]
    w = d * 2.0
    bar = mpatches.Rectangle((3.1, y - 0.18), w, 0.36, fc="#f0cf8a", ec="#9e5c00")
    ax.add_patch(bar)
    ax.text(3.1 + w + 0.12, y, f"{d:.2f}", va="center", fontsize=10,
            fontweight="bold" if k == 1 else "normal", color="#9e5c00")
    # 箭头从左盒子到条
    ax.annotate("", xy=(3.05, y), xytext=(2.55, y),
                arrowprops=dict(arrowstyle="->", color="#6e737b", lw=1.1))

ax.text(4.6, 0.35, "距离度量：MSE（论文）／代码另加 KL + 余弦", fontsize=9.5,
        color="#555", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f4ed", ec="#deddd8"))

# 右上：训练路径
ax.text(8.6, 5.6, "训练", fontsize=12, fontweight="bold", color="#0b7a5c")
ax.annotate("", xy=(7.3, 4.35), xytext=(6.05, 4.0),
            arrowprops=dict(arrowstyle="->", color="#0b7a5c", lw=1.6))
ax.text(8.6, 4.6, "j = argmin 距离  → 意图 2", fontsize=11, color="#0b7a5c",
        fontweight="bold", ha="center")
ax.text(8.6, 4.0, "① L$_{recon}$：只在分支 j 上\n    回传预测潜变量误差", fontsize=10, ha="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="#dcf2e8", ec="#0b7a5c"))
ax.text(8.6, 2.9, "② L$_{score}$：focal loss\n    训练 ScoreNet 去预测 j", fontsize=10, ha="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="#dcf2e8", ec="#0b7a5c"))

# 右下：推理路径（论文 vs 代码）
ax.text(8.6, 2.05, "推理", fontsize=12, fontweight="bold", color="#b34a06")
ax.text(8.6, 1.55, "论文写法：选 ScoreNet\n分数最高的分支", fontsize=10, ha="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec="#b34a06"))
ax.text(8.6, 0.75, "代码实际：best = argmin\n(距离+KL+余弦)，\nScoreNet 推理时未被使用", fontsize=10, ha="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f9e3d6", ec="#b34a06"))
ax.annotate("", xy=(7.3, 1.35), xytext=(6.05, 2.6),
            arrowprops=dict(arrowstyle="->", color="#b34a06", lw=1.6, ls="--"))

ax.set_title("World Model Selector：训练的“裁判”和推理的“裁判”不是同一个", fontsize=13, pad=12)
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/047-WM-world4drive/fig-selector.png",
            dpi=160, bbox_inches="tight")
print("saved fig-selector.png")
