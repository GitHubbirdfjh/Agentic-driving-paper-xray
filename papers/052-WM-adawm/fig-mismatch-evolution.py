# -*- coding: utf-8 -*-
"""AdaWM: 三种微调策略下两类失配 (TV 距离) 随微调步骤的演化示意。

复刻论文 Figure 4 的核心信息 (出自论文第 10 页, ROM03 任务):
  (a) model-only: 模型失配下降, 但策略失配一路上升 —— 策略跟不上新模型;
  (b) policy-only: 策略失配下降, 但模型失配一路上升 —— 决策依据越来越错;
  (c) AdaWM: 两条曲线都被按住, 没有哪一条失控。

曲线形状按论文 Figure 4 的走势手绘为示意 (schematic), 横轴与论文一致为
微调步数 (1.00e5 ~ 1.08e5), 纵轴为 TV 距离。目的不是复制数值, 而是让读者
一眼看到"单修一边会把另一边修坏"这个论文的核心论点。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

steps = np.linspace(1.00, 1.08, 200)

def smooth_down(a, b, k=6):
    return a + (b - a) * (1 - np.exp(-k * (steps - 1.00) / 0.08))

def smooth_up(a, b, k=6):
    return a + (b - a) * (1 - np.exp(-k * (steps - 1.00) / 0.08))

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True)

# (a) model-only: model TV down, policy TV up
axes[0].plot(steps, smooth_down(0.55, 0.18), color="#1a6fa3", lw=2.4, label="Dynamics-model TV")
axes[0].plot(steps, smooth_up(0.30, 0.92), color="#b34a06", lw=2.4, label="Policy TV")
axes[0].set_title("(a) Model-only finetuning", fontsize=12)
axes[0].annotate("policy can't keep up:\npolicy TV keeps rising",
                 xy=(1.065, 0.80), xytext=(1.015, 1.02),
                 arrowprops=dict(arrowstyle="->", color="#b34a06", lw=1.3),
                 fontsize=9.5, color="#b34a06")

# (b) policy-only: policy TV down, model TV up
axes[1].plot(steps, smooth_up(0.45, 0.95), color="#1a6fa3", lw=2.4, label="Dynamics-model TV")
axes[1].plot(steps, smooth_down(0.60, 0.15), color="#b34a06", lw=2.4, label="Policy TV")
axes[1].set_title("(b) Policy-only finetuning", fontsize=12)
axes[1].annotate("decisions rest on\nworsening predictions",
                 xy=(1.065, 0.83), xytext=(1.012, 1.02),
                 arrowprops=dict(arrowstyle="->", color="#1a6fa3", lw=1.3),
                 fontsize=9.5, color="#1a6fa3")

# (c) AdaWM: both controlled
axes[2].plot(steps, smooth_down(0.55, 0.28) + 0.06 * np.sin(steps * 60),
             color="#1a6fa3", lw=2.4, label="Dynamics-model TV")
axes[2].plot(steps, smooth_down(0.50, 0.30) + 0.06 * np.cos(steps * 55),
             color="#b34a06", lw=2.4, label="Policy TV")
axes[2].set_title("(c) AdaWM (alignment-driven)", fontsize=12)
axes[2].annotate("neither error\nruns away",
                 xy=(1.06, 0.35), xytext=(1.012, 0.62),
                 arrowprops=dict(arrowstyle="->", color="#0b7a5c", lw=1.3),
                 fontsize=9.5, color="#0b7a5c")

for ax in axes:
    ax.set_xlabel("Finetuning steps", fontsize=11)
    ax.set_xlim(1.00, 1.08)
    ax.set_ylim(0, 1.15)
    ax.grid(alpha=0.25)
axes[0].set_ylabel("TV distance", fontsize=11)
axes[0].legend(fontsize=10, loc="center right")
fig.suptitle("What Figure 4 of the paper shows: fixing one side alone breaks the other",
             fontsize=13, y=1.02)
fig.text(0.5, -0.02,
         "Schematic redraw of the paper's Figure 4 (p.10, ROM03 task); curves follow the paper's trends, not exact values.",
         ha="center", fontsize=9, style="italic", color="#6e737b")
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/052-WM-adawm/fig-mismatch-evolution.png",
            dpi=150, bbox_inches="tight")
print("saved fig-mismatch-evolution.png")
