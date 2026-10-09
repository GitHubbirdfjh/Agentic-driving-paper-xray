#!/usr/bin/env python3
"""表 2 的诚实读法：分布内（nuScenes）无上下文最好；零样本跨传感器（Waymo）图上下文最好。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]

IN   = "#1a6fa3"
OUT  = "#9e5c00"
WIRE = "#6e737b"
BG   = "#faf9f5"

labels = ["None\n(无 P1-3 上下文)", "Chain\n(只传 P3)", "Graph\n(传全部 P1-3)"]
nus  = [61.45, 50.43, 57.49]   # nuScenes behavior acc
waymo = [35.70, 34.62, 39.73]  # Waymo behavior acc

x = np.arange(len(labels))
wdt = 0.34
fig, ax = plt.subplots(figsize=(10.5, 5.4))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

b1 = ax.bar(x - wdt / 2, nus, wdt, color=IN, alpha=0.85, edgecolor=IN, label="nuScenes（分布内）behavior 准确率")
b2 = ax.bar(x + wdt / 2, waymo, wdt, color=OUT, alpha=0.85, edgecolor=OUT, label="Waymo（零样本跨传感器）behavior 准确率")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.2f}",
                ha="center", fontsize=10.5, color="#141413")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Behavior 准确率 (%)", fontsize=11)
ax.set_ylim(0, 72)
ax.legend(fontsize=10.5, frameon=True, facecolor="white", edgecolor=WIRE)
ax.set_title("图上下文的真实收益：分布内没有，零样本泛化才有（表 2）", fontsize=13,
             weight="bold", color="#141413", pad=12)
ax.text(1, 6, "分布内：None 最好（61.45）——图推理没带来额外好处，作者正文亦承认\n"
              "Waymo：Graph 最好（39.73），速度准确率 43.90 -> 54.29 是主要来源",
        ha="center", fontsize=10, color=WIRE, style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=WIRE, alpha=0.95))
ax.grid(axis="y", color=WIRE, alpha=0.25, linestyle="--")
plt.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/012-VLA-drivelm/fig-generalization.png"
plt.savefig(out, dpi=150, facecolor=BG)
print("saved", out)
