# -*- coding: utf-8 -*-
"""Fig: input-modality ablation (OpenDriveVLA Table IV, 0.5B, UniAD avg collision).
The skeptical point: dropping vision entirely (Ego+Hist+Cmd, no Visu) costs
almost nothing (0.29 vs 0.26), while dropping ego state (0.77) or history
(1.14) collapses. Open-loop nuScenes is dominated by ego+history, not vision.
"""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Noto Serif CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

BG = "#faf9f5"; INK = "#141413"
C_LOSS = "#b34a06"; C_LOSS_F = "#f9e3d6"
C_WIRE = "#6e737b"

labels = [
    "Visu+Hist+Cmd\n(无 Ego)",
    "Visu+Ego+Cmd\n(无 Hist)",
    "Ego+Hist+Cmd\n(无 Visu)",
    "Visu+Ego+Hist\n(无 Cmd)",
    "全输入",
]
coll = [0.77, 1.14, 0.29, 0.33, 0.26]
colors = [C_LOSS, C_LOSS, "#1a6fa3", C_LOSS, "#0b7a5c"]
fills = [C_LOSS_F, C_LOSS_F, "#e2eff5", C_LOSS_F, "#dcf2e8"]

fig, ax = plt.subplots(figsize=(12.6, 4.6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
x = np.arange(len(labels))
bars = ax.bar(x, coll, color=fills, edgecolor=colors, linewidth=1.8, width=0.62, zorder=3)
for i, v in enumerate(coll):
    ax.text(i, v + 0.035, f"{v:.2f}", fontsize=11, ha="center", weight="bold", color=colors[i])
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylabel("UniAD 平均碰撞率 (%) ↓", fontsize=11)
ax.set_ylim(0, 1.35)
ax.grid(axis="y", color="#deddd8", lw=0.8, alpha=0.7)
ax.set_title("去掉视觉输入，开环碰撞率只从 0.26 涨到 0.29——视觉 grounding 不是规划分数的主力",
             fontsize=12.5, color=INK, pad=12)

ax.annotate("去掉历史轨迹：\n直接崩到 1.14", xy=(1, 1.14), xytext=(1.9, 1.0),
            fontsize=9.5, color=INK, ha="center",
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
ax.annotate("去掉 Ego 状态：\n0.77，呼应 [53]", xy=(0, 0.77), xytext=(-0.02, 0.98),
            fontsize=9.5, color=INK, ha="center",
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
ax.annotate("去掉视觉：\n0.29 ≈ 0.26", xy=(2, 0.29), xytext=(2.9, 0.62),
            fontsize=9.5, color="#1a6fa3", ha="center", weight="bold",
            arrowprops=dict(arrowstyle="->", color="#1a6fa3", lw=1.5))

ax.text(4.32, 0.95,
        "读法：nuScenes 开环指标主要被 Ego 状态和历史轨迹“喂答案”主导\n"
        "（[53] 的结论：Ego 状态本身就能拟合出很低的开环误差）。\n"
        "这并不否定视觉 grounding 的价值——QA 任务和长尾场景\n"
        "仍需要它；但 Table I 的规划数字不能证明“3D 视觉起作用”。",
        fontsize=8.6, color=INK, ha="left", va="top",
        bbox=dict(fc="#ffffff", ec=C_WIRE, alpha=0.95, boxstyle="round,pad=0.4"))

plt.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/075-Agent-opendrivevla/fig-ablation.png"
plt.savefig(out, dpi=150, facecolor=BG)
print("saved", out)
