# -*- coding: utf-8 -*-
"""DriveGPT4-V2: CARLA Longest6 主结果 (表 2) Driving Score 对比.
数据来源: 论文表 2, DS 列. 颜色区分输入模态: 纯相机(C) vs 相机+激光雷达(C&L).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Noto Serif CJK JP"
plt.rcParams["axes.unicode_minus"] = False

methods = ["WOR", "LAV v1", "LMDrive*", "Interfuser", "TransFuser",
           "LAV v2", "PlanT", "Transfuser++*\n(同等数据)", "Transfuser++*\n(3倍数据)", "DriveGPT4-V2"]
ds = [21, 33, 36, 47, 47, 58, 58, 58, 65, 70]
modal = ["C", "C&L", "C&L", "C", "C&L", "C&L", "C&L", "C&L", "C&L", "C"]
colors = ["#1a6fa3" if m == "C" else "#6e737b" for m in modal]
colors[-1] = "#0b7a5c"  # DriveGPT4-V2 高亮

fig, ax = plt.subplots(figsize=(11, 4.6))
y = np.arange(len(methods))
bars = ax.barh(y, ds, color=colors, edgecolor="white", height=0.62)
ax.set_yticks(y)
ax.set_yticklabels(methods, fontsize=10)
ax.set_xlabel("Driving Score (DS, 越高越好)")
ax.set_title("CARLA Longest6 闭环评测：DriveGPT4-V2 vs 基线 (论文表 2)", fontsize=12, fontweight="bold")
ax.set_xlim(0, 78)
for b, v in zip(bars, ds):
    ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v}", va="center", fontsize=10, fontweight="bold")
ax.invert_yaxis()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

from matplotlib.patches import Patch
legend_elems = [Patch(facecolor="#0b7a5c", label="DriveGPT4-V2 (纯相机)"),
                Patch(facecolor="#1a6fa3", label="其他纯相机方法"),
                Patch(facecolor="#6e737b", label="相机+激光雷达方法")]
ax.legend(handles=legend_elems, loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/015-VLA-drivegpt4-v2/fig-longest6.png", dpi=150)
print("saved fig-longest6.png")
