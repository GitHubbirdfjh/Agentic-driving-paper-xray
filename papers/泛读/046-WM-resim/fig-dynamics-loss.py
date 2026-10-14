# -*- coding: utf-8 -*-
"""ReSim 动力学一致性损失示意：匹配的不是帧本身，而是帧间的"运动"。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.font_manager import FontProperties
_ZH = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
plt.rcParams["font.family"] = _ZH.get_name()
plt.rcParams["font.size"] = 10.5
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(10.5, 4.6))
ax.set_xlim(0, 10.5)
ax.set_ylim(0, 5.2)
ax.axis("off")

# 两条时间线：真值 latent x（上）与预测 latent d（下）
t = np.array([0, 1, 2, 3])
x = np.array([1.0, 1.6, 1.5, 2.3])   # 真值潜变量某通道随帧的变化（示意）
d = np.array([1.0, 1.3, 1.7, 2.1])   # 预测潜变量（示意）

def draw_track(y0, vals, color, label):
    xs = 1.2 + t * 2.2
    ys = y0 + vals * 0.9
    ax.plot(xs, ys, "o-", color=color, lw=2, ms=7, zorder=3)
    for xi, yi, ti in zip(xs, ys, t):
        ax.text(xi, yi + 0.28, f"{'x' if label=='GT' else 'd'}$_{{{ti}}}$",
                ha="center", fontsize=11, color=color, fontweight="bold")
    return xs, ys

ax.text(0.35, 4.35, "真值潜变量 x", fontsize=12, color="#0b7a5c", fontweight="bold")
ax.text(0.35, 1.35, "预测潜变量 d", fontsize=12, color="#b34a06", fontweight="bold")
xs, yx = draw_track(3.4, x, "#0b7a5c", "GT")
_, yd = draw_track(0.4, d, "#b34a06", "P")

# j=1 的运动差分：虚线箭头
for i in [0, 2]:
    # 真值运动 x_{i+1}-x_i
    ax.annotate("", xy=(xs[i+1], yx[i+1]), xytext=(xs[i], yx[i]),
                arrowprops=dict(arrowstyle="<->", color="#0b7a5c", lw=1.6,
                                linestyle=(0, (4, 3)), shrinkA=9, shrinkB=9))
    # 预测运动 d_{i+1}-d_i
    ax.annotate("", xy=(xs[i+1], yd[i+1]), xytext=(xs[i], yd[i]),
                arrowprops=dict(arrowstyle="<->", color="#b34a06", lw=1.6,
                                linestyle=(0, (4, 3)), shrinkA=9, shrinkB=9))

ax.text(xs[0] + 1.1, yx[0] + 0.62, "真值运动\nx$_{i+1}$−x$_i$", color="#0b7a5c",
        fontsize=10, ha="center")
ax.text(xs[2] + 1.1, yd[2] - 0.75, "预测运动\nd$_{i+1}$−d$_i$", color="#b34a06",
        fontsize=10, ha="center")

# j=2 的长程运动（只画一组，避免拥挤）
ax.annotate("", xy=(xs[3], yx[3]), xytext=(xs[1], yx[1]),
            arrowprops=dict(arrowstyle="<->", color="#0b7a5c", lw=1.2,
                            linestyle=(0, (6, 4)), shrinkA=9, shrinkB=9))
ax.text(xs[2], yx[2] + 0.75, "j=2 长程运动 x$_{i+2}$−x$_i$", color="#0b7a5c",
        fontsize=10, ha="center")

# 公式框
ax.text(10.15, 2.6,
        "L$_{dynamics}$ ∝\nΣ$_{j=1..K}$ Σ$_i$\n‖(d$_{i+j}$−d$_i$)\n −(x$_{i+j}$−x$_i$)‖² / s",
        fontsize=11, ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f8f4cf", ec="#7a6b00"))
ax.text(10.15, 1.15, "s = 该间隔下平均\n|运动幅度|（归一化）",
        fontsize=9.5, ha="right", va="center", color="#555")

ax.set_title("动力学一致性损失：监督的是“运动差分”，而不是帧本身", fontsize=13, pad=12)
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/046-WM-resim/fig-dynamics-loss.png",
            dpi=160, bbox_inches="tight")
print("saved fig-dynamics-loss.png")
