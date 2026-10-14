# -*- coding: utf-8 -*-
"""Fig: OpenDriveVLA multi-stage training pipeline (Stages 1/2/2.5/3).
Each stage: task, trainable params (from Table VIII), frozen/trainable modules,
and the UniAD avg-collision trajectory across stages (from Table V, 0.5B).
"""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams["font.family"] = ["Noto Serif CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

BG = "#faf9f5"; INK = "#141413"
C_OP, C_OP_F = "#0b7a5c", "#dcf2e8"
C_OUT, C_OUT_F = "#9e5c00", "#f8ecd2"
C_ALT, C_ALT_F = "#6a55c2", "#ebe7f8"
C_IN, C_IN_F = "#1a6fa3", "#e2eff5"
C_WIRE = "#6e737b"; C_LOSS = "#b34a06"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.55, 1]})
fig.patch.set_facecolor(BG)
for ax in (ax1, ax2):
    ax.set_facecolor(BG)

# ---------- left: pipeline ----------
ax1.set_xlim(0, 13.0); ax1.set_ylim(0, 5.0); ax1.axis("off")
ax1.set_title("四阶段训练：每阶段的任务与可训练参数（Table VIII，0.5B 版）", fontsize=11.5, color=INK, pad=10)

stages = [
    (1.65, "Stage 1\n分层视觉-语言对齐", "projector 可训练\n3.1M 参数\n视觉编码器、LLM 冻结",
     "输入：V_env + caption\n目标：LLM 生成对齐描述", C_IN, C_IN_F),
    (4.85, "Stage 2\n驾驶指令微调", "projector + LLM\n496.9M 参数\n视觉编码器冻结",
     "输入：V_env + S_ego + 问题\n目标：回答驾驶 QA\n（CoT 模式蒸馏进权重）", C_ALT, C_ALT_F),
    (8.05, "Stage 2.5\nAgent-Env-Ego 交互", "projector + LLM\n496.9M 参数\n视觉编码器冻结",
     "输入：V_env + S_ego + 目标 agent\n目标：预测该 agent 未来轨迹", C_OP, C_OP_F),
    (11.25, "Stage 3\n轨迹规划微调", "全模型（除 2D backbone）\n552.6M 参数",
     "输入：V_env + S_ego + 驾驶指令\n目标：自回归生成 6 个路点", C_OUT, C_OUT_F),
]
for x, title, params, io, col, fcol in stages:
    ax1.add_patch(patches.FancyBboxPatch(((x-1.45), 1.15), 2.9, 3.05, boxstyle="round,pad=0.04",
                                         fc=fcol, ec=col, lw=1.8))
    ax1.text(x, 3.92, title, fontsize=10, color=col, ha="center", va="top", weight="bold")
    ax1.text(x, 3.05, params, fontsize=8.3, color=INK, ha="center", va="top")
    ax1.text(x, 2.15, io, fontsize=8.0, color=C_WIRE, ha="center", va="top")
    if x < 11.25:
        ax1.annotate("", xy=(x+1.55, 2.7), xytext=(x+1.15, 2.7),
                     arrowprops=dict(arrowstyle="->", color=C_WIRE, lw=1.8))
ax1.text(6.2, 0.55, "关键设计：Stage 1 只训 3.1M 的投影器——视觉与 LLM 语义都冻结，\n"
         "对齐是“翻译”不是“重学”；Stage 2 把 CoT 推理模式蒸馏进权重，推理时不再显式推理。",
         fontsize=9, color=INK, ha="center",
         bbox=dict(fc="#ffffff", ec=C_WIRE, alpha=0.95, boxstyle="round,pad=0.35"))

# ---------- right: collision curve across stages ----------
ax2.set_title("每加一阶段，UniAD 平均碰撞率（Table V，0.5B）", fontsize=11.5, color=INK, pad=10)
stg = ["1", "1+2", "1+2+2.5", "1+2+2.5+3"]
coll = [0.37, 0.32, 0.31, 0.26]
l2 = [0.70, 0.69, 0.68, 0.68]
x = np.arange(len(stg))
ax2.plot(x, coll, "o-", color=C_LOSS, lw=2.2, ms=8, label="Avg. Collision (%) ↓")
for i, v in enumerate(coll):
    ax2.text(i, v + 0.012, f"{v:.2f}", fontsize=9.5, color=C_LOSS, ha="center", weight="bold")
ax2.set_xticks(x); ax2.set_xticklabels(stg, fontsize=9.5)
ax2.set_ylabel("碰撞率 (%)", fontsize=10, color=C_LOSS)
ax2.tick_params(axis="y", labelcolor=C_LOSS)
ax2.set_ylim(0.18, 0.44)
ax2.grid(axis="y", color="#deddd8", lw=0.8, alpha=0.7)
ax2.annotate("Stage 3 贡献最大\n(-0.05)", xy=(3, 0.26), xytext=(2.1, 0.21),
             fontsize=9, color=INK, ha="center",
             arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
ax2.annotate("Stage 2.5 几乎没动\n(-0.01)", xy=(2, 0.31), xytext=(1.0, 0.40),
             fontsize=9, color=INK, ha="center",
             arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
# note about missing baseline
ax2.text(1.5, 0.185, "注：没有“去掉 Stage 1”的对照行，\n正文“Stage 1 降碰撞最显著”缺乏直接证据",
         fontsize=8.5, color=C_WIRE, ha="center",
         bbox=dict(fc="#f5f4ed", ec="#deddd8", boxstyle="round,pad=0.3"))

plt.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/075-Agent-opendrivevla/fig-stages.png"
plt.savefig(out, dpi=150, facecolor=BG)
print("saved", out)
