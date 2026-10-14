# -*- coding: utf-8 -*-
"""Voyager 主结果复刻:表 1 科技树掌握(解锁所需 prompting 轮数,越少越好)
与表 2 零样本泛化到新世界新任务.数字全部取自论文(实证)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

C_BG = "#f7f5ef"
fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
fig.patch.set_facecolor(C_BG)
for ax in axes:
    ax.set_facecolor(C_BG)

# ---- 左: 表 1, 对数坐标 ----
levels = ["木制工具", "石制工具", "铁制工具", "钻石工具"]
voyager = [6, 11, 21, 102]
autogpt = [92, 94, 135, np.nan]       # 钻石级 0/3 未解锁
no_skill = [7, 9, 29, np.nan]
x = np.arange(len(levels)); w = 0.26
ax = axes[0]
ax.bar(x - w, voyager, w, label="Voyager", color="#2e9e6b", edgecolor="none")
ax.bar(x, autogpt, w, label="AutoGPT", color="#9e5c00", edgecolor="none")
ax.bar(x + w, no_skill, w, label="Voyager 去技能库", color="#8a8f98", edgecolor="none")
for i, v in enumerate(voyager):
    ax.text(i - w, v * 1.35, f"{v}", ha="center", fontsize=9, color="#1d5c3a",
            weight="bold")
for i, v in enumerate(autogpt[:3]):
    ax.text(i, v * 1.35, f"{v}", ha="center", fontsize=8.5, color="#7a4a00")
ax.set_xticks(x); ax.set_xticklabels(levels, fontsize=10)
ax.set_yscale("log"); ax.set_ylim(3, 400)
ax.set_ylabel("解锁所需 prompting 轮数 (对数轴,越少越好)", fontsize=10)
ax.set_title("科技树掌握 (表 1):木制级快 15.3×;\n只有 Voyager 解锁钻石级(102 轮,1/3 成功)",
             fontsize=11.5, weight="bold", pad=10)
ax.legend(frameon=False, fontsize=9.5)
ax.grid(axis="y", alpha=0.25, linestyle="--", which="both")

# ---- 右: 表 2 零样本泛化 ----
tasks = ["钻石镐", "金剑", "岩浆桶", "指南针"]
voy = [19, 18, 21, 18]
voy_succ = ["3/3", "3/3", "3/3", "3/3"]
ag_lib = [39, 30, np.nan, 30]   # AutoGPT + Voyager 技能库
x = np.arange(len(tasks)); w = 0.32
ax = axes[1]
b1 = ax.bar(x - w/2, voy, w, label="Voyager (3/3 全过)", color="#2e9e6b",
            edgecolor="none")
b2 = ax.bar(x + w/2, [39, 30, 0, 30], w, label="AutoGPT+技能库 (部分过)",
            color="#6a55c2", edgecolor="none", hatch="//")
for b, s in zip(b1, voy_succ):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8, s, ha="center",
            fontsize=8.5, color="#1d5c3a", weight="bold")
for b, v in zip(b2, [39, 30, 0, 30]):
    if v > 0:
        ax.text(b.get_x() + b.get_width()/2, v + 0.8, f"{v}轮", ha="center",
                fontsize=8.5, color="#4a3f8a")
ax.set_xticks(x); ax.set_xticklabels(tasks, fontsize=10)
ax.set_ylim(0, 48); ax.set_ylabel("解出所需轮数 (50 轮内未解出记失败)", fontsize=10)
ax.set_title("新世界零样本泛化 (表 2):基线全军覆没;\n技能库即插即用,连 AutoGPT 都被带飞",
             fontsize=11.5, weight="bold", pad=10)
ax.legend(frameon=False, fontsize=9.5)
ax.grid(axis="y", alpha=0.25, linestyle="--")

plt.tight_layout()
out = "/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/104-agent-voyager/fig-voyager-results.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
print("saved", out)
