# -*- coding: utf-8 -*-
"""ReSim 不平衡噪声采样：均匀采样 vs 论文声称 vs 代码实际。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.font_manager import FontProperties
_ZH = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
plt.rcParams["font.family"] = _ZH.get_name()
plt.rcParams["font.size"] = 11
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})

# 左：三种采样下 P(t in [500,1000]) 的对比
ax = axes[0]
labels = ["均匀采样\n(uniform)", "论文声称\n(from 1/2 to 2/3)", "代码实际\n(weight 3.0 : 1.0)"]
vals = [0.5, 2 / 3, 0.75]
colors = ["#9aa0a8", "#1a6fa3", "#b34a06"]
bars = ax.bar(labels, vals, color=colors, width=0.62, edgecolor="white")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.1%}",
            ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylim(0, 0.92)
ax.set_ylabel("P( t ∈ [500, 1000) )")
ax.set_title("噪声步 t 落在高噪声区间的概率", fontsize=13, pad=10)
ax.grid(axis="y", alpha=0.25)
ax.annotate("代码与论文不一致：\n3/4 ≠ 2/3",
            xy=(2, 0.75), xytext=(1.35, 0.85),
            fontsize=10, color="#b34a06",
            arrowprops=dict(arrowstyle="->", color="#b34a06", lw=1.4))

# 右：权重函数形状
ax = axes[1]
t = np.arange(0, 1000)
w = np.where(t >= 500, 3.0, 1.0)
ax.fill_between(t, 0, w, step="mid", color="#e2eff5", edgecolor="#1a6fa3", lw=1.6)
ax.set_xlim(0, 1000)
ax.set_ylim(0, 3.6)
ax.set_xlabel("扩散步 t")
ax.set_ylabel("采样权重 w(t)")
ax.set_title("代码中的 my_weight_function", fontsize=13, pad=10)
ax.text(250, 1.5, "w = 1.0", ha="center", fontsize=11)
ax.text(750, 3.25, "w = 3.0", ha="center", fontsize=11)
ax.axvline(500, color="#b34a06", ls="--", lw=1.2)
ax.text(500, 0.15, "t = 500", ha="center", fontsize=10, color="#b34a06")

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/046-WM-resim/fig-noise-sampling.png",
            dpi=160, bbox_inches="tight")
print("saved fig-noise-sampling.png")
