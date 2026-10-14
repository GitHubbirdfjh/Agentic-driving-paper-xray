"""重构 GEM 论文 Fig.4 的自回归噪声调度：共 6 帧、窗口 3 帧、每帧 3 步去噪。
行=迭代轮次，列=帧，格内数字=该轮开始时该帧的噪声等级（3最噪→0干净）。
绿色对勾=该轮结束时该帧被完全去噪并保存。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.font_manager import FontProperties
FP = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc")
plt.rcParams["font.family"] = FP.get_name()
plt.rcParams["axes.unicode_minus"] = False

# 噪声等级矩阵：rows=迭代(8轮), cols=帧(6帧)；-1 表示该帧不在窗口内
levels = np.full((8, 6), -1, dtype=int)
saved = np.zeros((8, 6), dtype=bool)
# init
levels[0, 0:3] = [3, 3, 3]
levels[1, 0:3] = [2, 2, 2]
levels[2, 0:3] = [1, 1, 2]
# iter2 结束 f0 干净
saved[2, 0] = True
levels[3, 1:4] = [1, 2, 3]
saved[3, 1] = True
levels[4, 2:5] = [1, 2, 3]
saved[4, 2] = True
levels[5, 3:6] = [1, 2, 3]
saved[5, 3] = True
levels[6, 4:6] = [1, 2]
saved[6, 4] = True
levels[7, 5:6] = [1]
saved[7, 5] = True

fig, ax = plt.subplots(figsize=(10, 7))
cmap = plt.cm.YlOrRd
for r in range(8):
    for c in range(6):
        lv = levels[r, c]
        if lv < 0:
            ax.add_patch(plt.Rectangle((c, 7 - r), 1, 1, fc="#eeeeee", ec="white", lw=2))
            continue
        color = cmap(lv / 3.0)
        ax.add_patch(plt.Rectangle((c, 7 - r), 1, 1, fc=color, ec="white", lw=2))
        ax.text(c + 0.5, 7 - r + 0.42, f"σ{lv}", ha="center", va="center",
                fontsize=11, fontweight="bold", color="black" if lv < 2 else "white")
        if saved[r, c]:
            ax.text(c + 0.88, 7 - r + 0.78, "✓", ha="center", va="center",
                    fontsize=13, color="#0b7a5c", fontweight="bold")

ax.set_xlim(0, 6)
ax.set_ylim(0, 8)
ax.set_xticks(np.arange(6) + 0.5)
ax.set_xticklabels([f"frame {i}" for i in range(6)], fontsize=11)
ax.set_yticks(np.arange(8) + 0.5)
ax.set_yticklabels([f"iter {7 - r}" for r in range(8)], fontsize=11)
ax.invert_yaxis()
ax.set_title("GEM 自回归噪声调度（重构论文 Fig.4：6帧 / 窗口3帧 / 每帧3步）",
             fontsize=13, fontweight="bold", pad=12)

# 阶段括号：数据行 r 占据 y∈[7-r, 8-r]；r0..r1 对应 y_lo=7-r1, y_hi=8-r0
for (r0, r1, label, col) in [(0, 2, "初始化\n(initialization)", "#1a6fa3"),
                             (3, 4, "自回归\n(autoregressive)", "#9e5c00"),
                             (5, 7, "终止\n(termination)", "#0b7a5c")]:
    y_lo, y_hi = 7 - r1 - 0.05, 8 - r0 + 0.05
    ax.annotate("", xy=(6.15, y_lo), xytext=(6.15, y_hi),
                arrowprops=dict(arrowstyle="-", color=col, lw=2))
    ax.text(6.35, (y_lo + y_hi) / 2, label, va="center", fontsize=10, color=col)

ax.text(0.02, 0.98,
        "每轮：窗口内每帧去噪一步；首帧干净(✓)即保存并滑窗，\n"
        "追加一帧纯噪声。8 轮共 8 次前向，每帧恰好去噪 3 步。",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(fc="white", ec="#cccccc", boxstyle="round,pad=0.4"))

plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/056-WM-gem/fig-schedule.png",
            dpi=150, bbox_inches="tight")
print("saved fig-schedule.png")
