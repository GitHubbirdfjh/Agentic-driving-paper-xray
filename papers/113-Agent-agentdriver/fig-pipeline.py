# -*- coding: utf-8 -*-
"""Agent-Driver 三件套架构图：神经模块 -> 工具库 -> LLM 动态调用 -> 认知记忆 -> 推理引擎四阶段."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
plt.rcParams["font.sans-serif"] = ["Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis("off")

def box(x, y, w, h, text, fc, fs=10, ec="#333333"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1.2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#1a1a1a",
            wrap=True, linespacing=1.4)

def arrow(x1, y1, x2, y2, label=None, ls="-", lw=1.6):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                        lw=lw, ls=ls, color="#333333", shrinkA=2, shrinkB=4)
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label, ha="center", fontsize=8.5, color="#555555")

ax.text(7, 7.55, "Agent-Driver: LLM as scheduler over neural modules", ha="center", fontsize=14, weight="bold")

# row 1: neural modules
mods = [("Detection\nModule", "#dcebf5"), ("Prediction\nModule", "#dcebf5"),
        ("Occupancy\nModule", "#dcebf5"), ("Map\nModule", "#dcebf5")]
for i, (t, c) in enumerate(mods):
    box(0.6 + i * 3.35, 5.9, 2.9, 1.0, t, c, fs=10)
ax.text(0.15, 6.4, "神经模块\n(数值输出)", fontsize=9, color="#555", va="center")

# tool library
box(4.3, 4.35, 5.4, 1.0, "Tool Library: 20+ 函数\n get_leading_object() / get_pred_trajs_for_object(i) /\n get_occ_at_loc_time(loc,t) / collision_check(traj) / get_lanes(loc) ...", "#fef3d8", fs=9)
for i in range(4):
    arrow(0.6 + i * 3.35 + 1.45, 5.9, 4.3 + (i + 0.5) * 5.4 / 4, 5.35, ls=":")

# LLM agent center
box(4.3, 2.55, 5.4, 1.15, "LLM Agent (gpt-3.5-turbo)\n动态函数调用: 先问\"要不要开检测?\"-> 选函数 -> 读文本结果 -> 再问\"要不要开预测?\"...", "#e7f4e4", fs=9)
arrow(7, 4.35, 7, 3.7, "多轮问答\n收集环境信息")

# cognitive memory right
box(10.6, 2.55, 3.0, 2.8, "Cognitive Memory\n\n常识记忆:\n交通规则文本\n\n经验记忆:\n(环境信息, 轨迹)\n两阶段检索:\n向量KNN -> LLM重排", "#efe6f7", fs=9)
arrow(9.7, 3.4, 10.6, 3.9, "检索")
arrow(10.6, 3.1, 9.7, 3.0, "查询", ls=":")

# reasoning engine bottom
stages = [("Chain-of-\nThought\n关键物体+影响", "#fde8d8"), ("Task\nPlanning\n行为+速度估计", "#fde8d8"),
          ("Motion\nPlanning\n微调LLM输出轨迹", "#fde8d8"), ("Self-\nReflection\n碰撞检查+优化", "#fbd9c8")]
for i, (t, c) in enumerate(stages):
    box(0.6 + i * 3.35, 0.5, 2.9, 1.35, t, c, fs=9.5)
    if i > 0:
        arrow(0.6 + i * 3.35 - 0.05, 1.17, 0.6 + i * 3.35 - 0.4, 1.17)
ax.text(0.15, 1.17, "推理引擎\n(四阶段)", fontsize=9, color="#555", va="center")
arrow(5.2, 2.55, 3.2, 1.85, "环境+记忆\n输入")
arrow(12.1, 0.5, 12.1, 0.08)
ax.text(12.1, 0.28, "  安全轨迹输出", fontsize=9.5, color="#333")

# self-reflection loop annotation
ax.annotate("", xy=(11.9, 0.6), xytext=(13.2, 0.6),
            arrowprops=dict(arrowstyle="->", color="#b34a06", lw=1.4, connectionstyle="arc3,rad=0.5"))
ax.text(13.35, 0.75, "碰撞->重优化\n式(1)", fontsize=8, color="#b34a06")

plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/113-Agent-agentdriver/fig-pipeline.png", dpi=150, bbox_inches="tight")
print("saved fig-pipeline.png")
