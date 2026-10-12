# -*- coding: utf-8 -*-
"""ReAct 主结果复刻:表 1 (HotpotQA EM / Fever Acc) 与决策任务 (ALFWorld / WebShop 成功率).
全部数字取自论文正文表格,标注实证来源."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

C_BG = "#f7f5ef"
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
fig.patch.set_facecolor(C_BG)
for ax in axes:
    ax.set_facecolor(C_BG)

# ---- 左: 表 1 ----
methods = ["Standard", "CoT", "CoT-SC", "Act", "ReAct", "CoT-SC→ReAct", "ReAct→CoT-SC"]
hotpot = [28.7, 29.4, 33.4, 25.7, 27.4, 34.2, 35.1]   # EM, 表 1
fever  = [57.1, 56.3, 60.4, 58.9, 60.9, 64.6, 62.0]   # Acc, 表 1
x = np.arange(len(methods)); w = 0.36
ax = axes[0]
b1 = ax.bar(x - w/2, hotpot, w, label="HotpotQA (EM)", color="#3b7dd8", edgecolor="none")
b2 = ax.bar(x + w/2, fever, w, label="FEVER (Acc)", color="#d97b2e", edgecolor="none")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.7,
                f"{b.get_height():.1f}", ha="center", fontsize=8.2, color="#333")
ax.set_xticks(x); ax.set_xticklabels(methods, rotation=22, ha="right", fontsize=9)
ax.set_ylim(0, 72); ax.set_ylabel("分数 (%)", fontsize=10)
ax.set_title("知识推理: ReAct vs 基线 (表 1, PaLM-540B few-shot)\n组合 ReAct+CoT-SC 拿到 prompting 最优",
             fontsize=11.5, weight="bold", pad=10)
ax.legend(frameon=False, fontsize=9.5)
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.text(0.02, 0.86, "SOTA(领域专用方法): 67.5 / 89.5,差距仍大",
        transform=ax.transAxes, fontsize=8.5, color="#777", va="top")

# ---- 右: 决策任务 ----
tasks = ["ALFWorld\n成功率", "WebShop\n成功率", "WebShop\n平均分"]
# 表 3: ReAct best-of-6 71, Act best 45, BUTLER best 37 ; 表 4: ReAct 40.0/66.6, Act 30.1/62.3, IL 29.1/59.9
react = [71.0, 40.0, 66.6]
act   = [45.0, 30.1, 62.3]
other = [37.0, 29.1, 59.9]
other_lbl = ["BUTLER", "IL", "IL"]
x = np.arange(len(tasks)); w = 0.26
ax = axes[1]
b1 = ax.bar(x - w, react, w, label="ReAct (best)", color="#2e9e6b", edgecolor="none")
b2 = ax.bar(x, act, w, label="Act (best)", color="#9e5c00", edgecolor="none")
b3 = ax.bar(x + w, other, w, label="BUTLER / IL", color="#8a8f98", edgecolor="none")
for bars in (b1, b2, b3):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.0,
                f"{b.get_height():.1f}", ha="center", fontsize=8.6, color="#333")
ax.set_xticks(x); ax.set_xticklabels(tasks, fontsize=10)
ax.set_ylim(0, 80); ax.set_ylabel("成功率/分数 (%)", fontsize=10)
ax.set_title("交互决策: 1~3 个 in-context 例就超过\n1k~100k 样本训练的 IL/RL (表 3/表 4)",
             fontsize=11.5, weight="bold", pad=10)
ax.legend(frameon=False, fontsize=9.5)
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.text(0.02, 0.97, "WebShop 人类专家: 82.1 分 / 59.6 成功率,仍远高于所有模型",
        transform=ax.transAxes, fontsize=8.5, color="#777", va="top")

plt.tight_layout()
out = "/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/102-agent-react/fig-react-results.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
print("saved", out)
