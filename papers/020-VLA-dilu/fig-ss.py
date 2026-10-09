# -*- coding: utf-8 -*-
"""论文文字中报告的中位数 Success Steps (SS):
0-shot: 中位数 <5 (任务无一完成)
memory=40: 1-shot 中位数 25, 3-shot 中位数 27, 5-shot 中位数 >25 (几乎全过)
memory=20 (图6b, 跨环境 lane-4-density-2 -> lane-5-density-3):
  3-shot: 13 -> 5 ; 5-shot: 30 -> 23
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

# (a) memory 物品数 / few-shot 数对中位数 SS 的影响 (图6a 文字数字)
labels = ["0-shot\n(no memory)", "1-shot\n(40 mem)", "3-shot\n(40 mem)", "5-shot\n(40 mem)"]
vals = [4, 25, 27, 28]
colors = ["#b34a06", "#1a6fa3", "#1a6fa3", "#1a6fa3"]
ax = axes[0]
bars = ax.bar(labels, vals, color=colors, width=0.62)
ann = ["<5\n(0/10 success)", "25", "27", ">25\n(nearly all pass)"]
for b, a in zip(bars, ann):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, a,
            ha="center", fontsize=10)
ax.axhline(30, ls="--", color="gray", lw=1)
ax.text(3.4, 30.6, "SS=30: task complete", fontsize=9, color="gray")
ax.set_ylabel("median Success Steps (SS)", fontsize=11)
ax.set_title("(a) median SS vs memory size / few-shot number\n"
             "same env: lane-4-density-2", fontsize=11)
ax.set_ylim(0, 36)

# (b) 跨环境泛化 (图6b 文字数字), memory=20
labels2 = ["3-shot\nsame env", "3-shot\nlane-5-density-3",
           "5-shot\nsame env", "5-shot\nlane-5-density-3"]
vals2 = [13, 5, 30, 23]
colors2 = ["#1a6fa3", "#9e5c00", "#1a6fa3", "#9e5c00"]
ax = axes[1]
bars = ax.bar(labels2, vals2, color=colors2, width=0.62)
for b, v in zip(bars, vals2):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.6, str(v), ha="center", fontsize=10)
ax.axhline(30, ls="--", color="gray", lw=1)
ax.set_ylabel("median Success Steps (SS)", fontsize=11)
ax.set_title("(b) generalization: memory from lane-4-density-2\n"
             "tested on lane-5-density-3 (20 memory items)", fontsize=11)
ax.set_ylim(0, 36)

fig.suptitle("DiLu: median Success Steps reported in the paper text "
             "(box plots in Fig.6 not digitized)", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/"
            "020-VLA-dilu/fig-ss.png", dpi=150, bbox_inches="tight")
print("saved fig-ss.png")
