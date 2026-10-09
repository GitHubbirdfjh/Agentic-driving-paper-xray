"""MindDriver: 三种推理路径的机制对比示意（对应论文图1的理解重绘）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_SEM = "#1a6fa3"   # 语义空间
C_MID = "#6a55c2"   # 中间想象空间
C_PHY = "#9e5c00"   # 物理空间
C_BAD = "#b34a06"
C_OK  = "#0b7a5c"

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False)
titles = ["(a) 纯文本 CoT：语义→物理 空间错位",
          "(b) 纯图像 CoT：无规划目标引导",
          "(c) 本文：渐进多模态推理（对齐平滑）"]

def box(ax, x, y, w, h, text, color, alpha=0.25):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=color, ec=color, alpha=alpha, lw=1.5))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9.5,
            color="#141413", weight="bold")

def arrow(ax, x1, y1, x2, y2, color, style="->", ls="-", lw=2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 color=color, lw=lw, linestyle=ls,
                                 mutation_scale=16))

for ax, t in zip(axes, titles):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(t, fontsize=11, weight="bold", pad=12)

# (a)
ax = axes[0]
box(ax, 0.06, 0.62, 0.38, 0.26, "文本推理\n语义空间", C_SEM)
box(ax, 0.56, 0.62, 0.38, 0.26, "轨迹\n物理空间", C_PHY)
arrow(ax, 0.46, 0.75, 0.55, 0.75, C_BAD, ls=(0, (4, 3)))
ax.text(0.5, 0.82, "直接跳跃\n空间错位", ha="center", fontsize=9, color=C_BAD, weight="bold")
ax.text(0.25, 0.45, "“前方有车，应直行”\n抽象语义 → 具体坐标\n无桥梁", ha="center", fontsize=8.5, color="#3d3d3a")
ax.text(0.75, 0.45, "Turn Left / Turn Right\n误判方向", ha="center", fontsize=8.5, color=C_BAD)

# (b)
ax = axes[1]
box(ax, 0.06, 0.62, 0.38, 0.26, "图像推理\n中间空间", C_MID)
box(ax, 0.56, 0.62, 0.38, 0.26, "轨迹\n物理空间", C_PHY)
arrow(ax, 0.46, 0.75, 0.55, 0.75, C_MID)
ax.text(0.5, 0.82, "Dream 下一帧\n无目标指引", ha="center", fontsize=9, color=C_MID, weight="bold")
ax.text(0.25, 0.45, "不知道该聚焦\n哪个物体\n（卡车？行人？灯？）", ha="center", fontsize=8.5, color="#3d3d3a")
ax.text(0.75, 0.45, "场景演化不准\n规划跟着偏", ha="center", fontsize=8.5, color=C_BAD)

# (c)
ax = axes[2]
box(ax, 0.01, 0.62, 0.31, 0.26, "文本推理\n语义空间", C_SEM)
box(ax, 0.345, 0.62, 0.31, 0.26, "未来想象\n中间空间", C_MID)
box(ax, 0.68, 0.62, 0.31, 0.26, "轨迹规划\n物理空间", C_PHY)
arrow(ax, 0.325, 0.75, 0.34, 0.75, C_OK)
arrow(ax, 0.66, 0.75, 0.675, 0.75, C_OK)
ax.text(0.332, 0.84, "文本引导", ha="center", fontsize=8.5, color=C_OK, weight="bold")
ax.text(0.668, 0.84, "细节映射", ha="center", fontsize=8.5, color=C_OK, weight="bold")
ax.text(0.5, 0.45, "“大卡车横穿路口\n→ 想象它 0.5s 后的位置\n→ 轨迹绕开”", ha="center", fontsize=8.5, color="#3d3d3a")
ax.text(0.5, 0.24, "渐进：语义 → 想象 → 物理，步步有据", ha="center",
        fontsize=9, color=C_OK, weight="bold")

fig.suptitle("MindDriver 的核心主张：推理要“渐进”，模态要“对齐”",
             fontsize=13, weight="bold", y=1.02)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/081-Agent-minddriver/fig-progressive.png",
            dpi=150, bbox_inches="tight")
print("saved fig-progressive.png")
