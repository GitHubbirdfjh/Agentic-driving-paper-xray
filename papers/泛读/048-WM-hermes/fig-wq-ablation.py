# -*- coding: utf-8 -*-
"""HERMES Fig 3: 每组 world query 数量 n 的消融 (论文表 S3, 1/4 训练数据).
双轴: 左轴 3s Chamfer 距离 (越低越好), 右轴 CIDEr (越高越好).
n=4 最优; n 太大反而掉点 —— 论文归因于冗余信息与优化困难.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# 表 S3 原始数据
n = [1, 2, 4, 8, 16]
cd3s = [1.725, 1.720, 1.718, 1.744, 1.748]
cider = [0.712, 0.714, 0.720, 0.713, 0.716]

fig, ax1 = plt.subplots(figsize=(8.5, 5))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

ax1.set_xlabel("每组 world queries 数量 n", fontsize=11)
ax1.set_ylabel("3s Chamfer 距离 ↓ (越低越好)", fontsize=11, color="#b34a06")
l1, = ax1.plot(n, cd3s, "o-", color="#b34a06", lw=2.2, ms=8, label="3s Chamfer 距离")
ax1.tick_params(axis="y", labelcolor="#b34a06")
ax1.set_xscale("log", base=2)
ax1.set_xticks(n)
ax1.set_xticklabels([str(x) for x in n])
ax1.set_ylim(1.70, 1.76)
for x, y in zip(n, cd3s):
    ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10),
                 ha="center", fontsize=9, color="#b34a06")

ax2 = ax1.twinx()
ax2.set_ylabel("CIDEr ↑ (越高越好)", fontsize=11, color="#1a6fa3")
l2, = ax2.plot(n, cider, "s--", color="#1a6fa3", lw=2.2, ms=8, label="CIDEr")
ax2.tick_params(axis="y", labelcolor="#1a6fa3")
ax2.set_ylim(0.705, 0.725)
for x, y in zip(n, cider):
    ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -16),
                 ha="center", fontsize=9, color="#1a6fa3")

ax1.axvline(4, color="#0b7a5c", ls=":", lw=1.8)
ax1.text(4, 1.757, "  n=4 (论文默认)", fontsize=10, color="#0b7a5c", weight="bold", va="top")

ax1.set_title("world query 数量 n: 生成呈 U 形, 理解几乎不动 (表 S3)", fontsize=12, weight="bold", pad=12)
ax1.grid(True, alpha=0.25, which="both")
fig.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/048-WM-hermes/fig-wq-ablation.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
