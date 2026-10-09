# -*- coding: utf-8 -*-
"""GRPO 零奖励僵局 vs ELF-VLA 的反馈修正：优势信号从哪里来。
左：长尾场景下 n=8 条 rollout 全部得零分 -> 优势全零 -> 梯度消失。
右：混入一条老师指导修正后的高分样本 -> 归一化后它拿到大正优势 -> 定向梯度。
数字为示意（按论文 4.2 节 total-failure 场景的量级构造），机制忠实于式 (7)(8)。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

# ---- 左图：全部零分 ----
n = 8
r = np.zeros(n)
ax = axes[0]
bars = ax.bar(range(n), r, color="#b9c2cc", edgecolor="#5a626b", width=0.7)
ax.set_ylim(-0.05, 1.05)
ax.set_xticks(range(n))
ax.set_xticklabels([f"rollout {i+1}" for i in range(n)], fontsize=8)
ax.set_ylabel("奖励 r (PDMS 等)", fontsize=11)
ax.set_title("(a) 常规 GRPO：长尾场景下全部 rollout 零分", fontsize=12, pad=10)
ax.axhline(0, color="black", linewidth=0.8)
for i in range(n):
    ax.text(i, 0.06, "0.00", ha="center", fontsize=9, color="#333333")
ax.text(3.5, 0.82, "mean = 0, std = 0\n优势 A_i = 0/0 → 无学习信号\n策略梯度 ≈ 0：原地打转",
        ha="center", fontsize=10.5, color="#8a1f1f",
        bbox=dict(boxstyle="round,pad=0.5", fc="#fbe9e7", ec="#c97b6e"))

# ---- 右图：混入修正样本 ----
r2 = np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.92])  # 最后一条是反馈修正样本
labels = [f"r{i+1}" for i in range(8)] + ["修正样本"]
colors = ["#b9c2cc"] * 8 + ["#2e7d32"]
ax = axes[1]
bars = ax.bar(range(9), r2, color=colors, edgecolor="#37474f", width=0.65)
ax.set_ylim(-0.05, 1.05)
ax.set_xticks(range(9))
ax.set_xticklabels(labels, fontsize=8, rotation=20)
ax.set_ylabel("奖励 r", fontsize=11)
ax.set_title("右) ELF-VLA：把一条老师指点过的修正样本混进 batch", fontsize=12, pad=10)
mean, std = r2.mean(), r2.std()
ax.axhline(mean, color="#8a1f1f", linestyle="--", linewidth=1.2,
           label=f"合并集均值 ≈ {mean:.2f}")
adv = (r2 - mean) / std
for i, (v, a) in enumerate(zip(r2, adv)):
    ax.text(i, v + 0.04, f"{v:.2f}\nA={a:+.1f}", ha="center", fontsize=7.5,
            color="#1b5e20" if i == 8 else "#333333", fontweight="bold" if i == 8 else "normal")
ax.text(3.2, 0.55, "修正样本拿到大正优势 A≈+2.8\n"
        "→ 定向梯度把它\"拉\"进策略\n其余样本优势≈-0.35：轻微推开",
        ha="center", fontsize=10.5, color="#1b5e20",
        bbox=dict(boxstyle="round,pad=0.5", fc="#e8f5e9", ec="#66a370"))
ax.legend(fontsize=9, loc="upper right")

fig.suptitle("为什么 GRPO 会卡住、ELF-VLA 如何给出第一推动力（式 7–8 的机制示意）",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/083-Agent-elf-vla/fig-plateau.png",
            dpi=150, bbox_inches="tight")
print("saved fig-plateau.png")
