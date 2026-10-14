"""Figure 1 重绘：延迟导致的两种错位。
左：空间错位——协同车的特征停在 400ms 前的位置，自车视野盲区无参照。
右：语义错位——两车都看到同一物体，但位置差 10 个格子，融合认不出是同一个。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "sans-serif"
from matplotlib.font_manager import fontManager
fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
plt.rcParams["font.sans-serif"] = ["Noto Serif CJK JP", "DejaVu Sans"]

IN = "#1a6fa3"     # 输入/观测
OUT = "#9e5c00"    # 输出/预测位置
LOSS = "#b34a06"   # 错位/失败
WIRE = "#6e737b"
ALT = "#6a55c2"    # 新部件（轨迹场）

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor="white")

for ax, title in zip(axes, ["(a) 空间错位 spatial misalignment",
                            "(b) 语义错位 semantic misalignment"]):
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 30)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x (m)")
    # 自车
    ax.add_patch(mpatches.Rectangle((2, 2), 6, 4, fc="none", ec=IN, lw=2))
    ax.text(5, 4, "ego\n(t)", ha="center", va="center", color=IN, fontsize=10)
    # 自车视野扇形
    wedge = mpatches.Wedge((5, 4), 40, 20, 70, fc=IN, alpha=0.08, ec="none")
    ax.add_patch(wedge)

# ---------- (a) 空间错位：自车看不见，协同特征过期 ----------
ax = axes[0]
# 盲区（建筑物遮挡）
ax.add_patch(mpatches.Rectangle((20, 0), 10, 14, fc=WIRE, alpha=0.25, ec=WIRE, ls="--"))
ax.text(25, 7, "occlusion", ha="center", color=WIRE, fontsize=10)
# 真实车辆位置（t 时刻）
ax.add_patch(mpatches.Rectangle((33, 12), 4, 2, fc=OUT, ec=OUT))
ax.text(35, 16.5, "true pos @ t", ha="center", color=OUT, fontsize=10)
# 协同车在 s=t-400ms 看到的位置（过期特征）
ax.add_patch(mpatches.Rectangle((23, 12), 4, 2, fc="none", ec=LOSS, lw=2, ls="--"))
ax.text(25, 10.5, "stale feat @ s", ha="center", color=LOSS, fontsize=10)
ax.annotate("", xy=(33, 13), xytext=(27, 13),
            arrowprops=dict(arrowstyle="->", color=LOSS, lw=1.6))
ax.text(30, 14.2, "4 m gap", ha="center", color=LOSS, fontsize=10)
# 协同车
ax.add_patch(mpatches.Rectangle((42, 2), 6, 4, fc="none", ec=ALT, lw=2))
ax.text(45, 4, "agent i\n(s)", ha="center", va="center", color=ALT, fontsize=10)
ax.set_ylabel("y (m)")

# ---------- (b) 语义错位：两车都看到，但对不上 ----------
ax = axes[1]
# 自车看到的（t 时刻）
ax.add_patch(mpatches.Rectangle((33, 12), 4, 2, fc=IN, ec=IN, alpha=0.35))
ax.text(35, 16.5, "ego sees @ t", ha="center", color=IN, fontsize=10)
# 协同车传来的过期特征（s 时刻）
ax.add_patch(mpatches.Rectangle((23, 12), 4, 2, fc="none", ec=LOSS, lw=2, ls="--"))
ax.text(25, 10.5, "agent i feat @ s", ha="center", color=LOSS, fontsize=10)
ax.annotate("", xy=(23.5, 13), xytext=(32.5, 13),
            arrowprops=dict(arrowstyle="<->", color=LOSS, lw=1.6))
ax.text(28, 14.2, "same car?", ha="center", color=LOSS, fontsize=10,
        style="italic")
# 错误的融合结果：两个框
ax.add_patch(mpatches.Rectangle((33, 19), 4, 2, fc="none", ec=LOSS, lw=1.2))
ax.add_patch(mpatches.Rectangle((23, 19), 4, 2, fc="none", ec=LOSS, lw=1.2))
ax.text(28, 22.5, "fusion outputs 2 boxes\n(false positive)",
        ha="center", color=LOSS, fontsize=10)
ax.annotate("", xy=(28, 21), xytext=(28, 19.5),
            arrowprops=dict(arrowstyle="->", color=LOSS, lw=1.2))

for ax in axes:
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_yticks([0, 10, 20, 30])

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/087-Agent-traf-align/fig-delay-scenario.png",
            dpi=150)
print("saved fig-delay-scenario.png")
