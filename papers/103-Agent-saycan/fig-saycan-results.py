# -*- coding: utf-8 -*-
"""SayCan 主结果复刻:表 2 七个指令家族的 plan/execute 成功率,
以及消融对比(SayCan vs No VF vs Generative;PaLM vs FLAN).
数字全部取自论文表 2/表 3(实证)."""
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

# ---- 左: 七个家族 plan/execute (mock kitchen, 表 2) ----
fams = ["单技能\nNL", "名词\nNL", "动词\nNL", "结构化\n语言", "具身\nEmbodiment",
        "众包\nCrowd", "长时程\nLong"]
plan = [100, 67, 100, 93, 64, 87, 73]
exe  = [100, 47, 93, 87, 55, 87, 47]
x = np.arange(len(fams)); w = 0.36
ax = axes[0]
b1 = ax.bar(x - w/2, plan, w, label="规划成功率 plan", color="#3b7dd8", edgecolor="none")
b2 = ax.bar(x + w/2, exe, w, label="执行成功率 execute", color="#d97b2e", edgecolor="none")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.2, f"{b.get_height():.0f}",
            ha="center", fontsize=8, color="#333")
ax.set_xticks(x); ax.set_xticklabels(fams, fontsize=9)
ax.set_ylim(0, 112); ax.set_ylabel("成功率 (%)", fontsize=10)
ax.set_title("101 条指令 × 7 家族 (表 2, mock kitchen)\n规划 84% / 执行 74%;真厨房执行掉到 60%",
             fontsize=11.5, weight="bold", pad=10)
ax.legend(frameon=False, fontsize=9.5)
ax.grid(axis="y", alpha=0.25, linestyle="--")

# ---- 右: 消融 ----
labels = ["PaLM-\nSayCan", "No VF\n(去affordance)", "Generative\n(生成+投影)",
          "FLAN-\nSayCan", "BC NL\n(整句喂策略)", "BC USE\n(投影到技能)"]
plan_ab = [84, 67, 74, 70, 0, 9]
exe_ab  = [74, np.nan, np.nan, 61, 0, 9]
x = np.arange(len(labels)); w = 0.36
ax = axes[1]
b1 = ax.bar(x - w/2, plan_ab, w, label="规划成功率", color="#2e9e6b", edgecolor="none")
vals = [74, 0, 0, 61, 0, 9]
b2 = ax.bar(x + w/2, [74, 0, 0, 61, 0, 9], w, label="执行成功率",
            color="#9e5c00", edgecolor="none", hatch="//")
for b in list(b1):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.2, f"{b.get_height():.0f}",
            ha="center", fontsize=8, color="#333")
for b, v in zip(b2, vals):
    if v > 0:
        ax.text(b.get_x() + b.get_width()/2, v + 1.2, f"{v:.0f}",
                ha="center", fontsize=8, color="#333")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylim(0, 100); ax.set_ylabel("成功率 (%)", fontsize=10)
ax.set_title("消融:去 affordance 掉 17 个点;\n换 FLAN 错误翻倍;纯 BC 指令跟随 0%",
             fontsize=11.5, weight="bold", pad=10)
ax.legend(frameon=False, fontsize=9.5)
ax.grid(axis="y", alpha=0.25, linestyle="--")

plt.tight_layout()
out = "/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/103-agent-saycan/fig-saycan-results.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
print("saved", out)
