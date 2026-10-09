# -*- coding: utf-8 -*-
"""ELF-VLA 三阶段管线 + GRPO 反馈循环的框图。
忠实于论文图 2、图 3 与算法 1：两阶段 SFT 打底，RL 阶段老师只在 r_traj < s 时介入。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")

IN, OP, ALT, LOSS, WIRE = "#1a6fa3", "#0b7a5c", "#6a55c2", "#b34a06", "#6e737b"

def box(x, y, w, h, text, fc, ec, fs=10):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       fc=fc, ec=ec, linewidth=1.4)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color="#141413", linespacing=1.5)
    return p

def arrow(x1, y1, x2, y2, label="", c=WIRE, fs=8.5, style="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                        color=c, linewidth=1.5, linestyle=style,
                        connectionstyle="arc3,rad=0.0")
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center",
                fontsize=fs, color=c,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=c))

# ---- 阶段一/二：SFT ----
ax.text(2.1, 6.55, "阶段一 & 二：SFT（打地基）", fontsize=12, fontweight="bold", ha="center")
box(0.3, 5.3, 3.6, 0.9, "驾驶问答预训练\nDriveLM/LingoQA/...", "#e2eff5", IN, 9.5)
box(0.3, 4.1, 3.6, 0.9, "混合 SFT：(q_base, o_gt)\n+ (q_fb, o_gt) 学会\"被指点后修正\"", "#e2eff5", IN, 9.5)
arrow(2.1, 5.3, 2.1, 5.0, "")

# ---- 阶段三：RL with feedback ----
ax.text(8.0, 6.55, "阶段三：GRPO + 失败反馈（主战场）", fontsize=12, fontweight="bold", ha="center")
box(5.2, 5.3, 2.2, 0.9, "策略采样 n 条\nVLA(q_base)", "#dcf2e8", OP, 9.5)
box(8.0, 5.3, 2.2, 0.9, "打分 r_traj\nPDMS 仿真", "#f9e3d6", LOSS, 9.5)
box(5.2, 3.9, 2.2, 1.0, "r ≥ s=0.8？\n是→规则正反馈\n否→老师介入", "#ebe7f8", ALT, 9.5)
box(8.0, 3.9, 2.2, 1.0, "老师 Qwen3-VL-32B\n看 (q_base, o_w, o_gt)\n输出 5 段结构化诊断", "#ebe7f8", ALT, 9.5)
box(5.2, 2.5, 2.2, 0.9, "构造 q_fb\n策略再采样 o_fb\n选 k=1 条更优者", "#dcf2e8", OP, 9.5)
box(8.0, 2.5, 2.2, 0.9, "拼成 n+k 批\n目标函数式 (4)\n反馈分支用 shaping", "#dcf2e8", OP, 9.5)
box(5.2, 1.1, 5.0, 0.9, "数据先筛一遍：85k → 24k（去掉高均值低方差的\"已学会\"样本）", "#f5f4ed", WIRE, 9.5)

arrow(7.4, 5.75, 8.0, 5.75, "")
arrow(9.1, 5.3, 9.1, 4.9, "")
arrow(8.0, 4.4, 7.4, 4.4, "", c=ALT)
ax.text(7.7, 4.62, "否", fontsize=9, color=ALT, fontweight="bold")
arrow(6.3, 3.9, 6.3, 3.4, "")
arrow(7.4, 2.95, 8.0, 2.95, "")
# 回环箭头：梯度更新后回到采样（下一轮迭代）
ax.annotate("", xy=(5.0, 5.75), xytext=(9.1, 2.5),
            arrowprops=dict(arrowstyle="-|>", color=OP, linewidth=1.8,
                            mutation_scale=14,
                            connectionstyle="arc3,rad=0.28"))
ax.text(10.9, 4.6, "梯度更新\n下一轮迭代", fontsize=9, color=OP, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=OP))

# SFT -> RL
arrow(3.9, 4.55, 5.2, 4.55, "SFT 模型\n作为起点", c=IN)

ax.text(6.0, 0.35, "承重结构只有两处：① 老师诊断把 GT 翻译成策略能用的语言；② shaping 让修正样本的梯度不被条件错配冲掉。其余（格式奖励、数据筛选）是脚手架。",
        fontsize=9.5, color="#555550", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#c9c6bb"))
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/083-Agent-elf-vla/fig-pipeline.png",
            dpi=150, bbox_inches="tight")
print("saved fig-pipeline.png")
