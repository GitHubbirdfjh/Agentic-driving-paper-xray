"""HEAL 论文表 2 训练代价数据的两种读法：单步边际代价 vs 累计代价。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

steps = ["+C(384)_E", "+L(32)_S", "+C(336)_R"]
heal_step = np.array([20.25, 6.34, 7.15])
hmvit_step = np.array([47.71, 65.08, 83.34])
heal_cum = np.cumsum(heal_step)
hmvit_cum = np.cumsum(hmvit_step)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
x = np.arange(3)
w = 0.36

ax = axes[0]
b1 = ax.bar(x - w/2, hmvit_step, w, label="HM-ViT (retrain all)", color="#e07856", edgecolor="#8a3b28")
b2 = ax.bar(x + w/2, heal_step, w, label="HEAL (train new encoder only)", color="#4c9be0", edgecolor="#1f5b99")
for v in list(hmvit_step) + list(heal_step):
    pass
for i, v in enumerate(hmvit_step):
    ax.text(i - w/2, v + 1.5, f"{v:.2f}", ha="center", fontsize=9, color="#8a3b28")
for i, v in enumerate(heal_step):
    ax.text(i + w/2, v + 1.5, f"{v:.2f}", ha="center", fontsize=9, color="#1f5b99")
ax.annotate("7.15 vs 83.34\n= 91.5% fewer", xy=(2 + w/2, 7.15), xytext=(1.15, 55),
            arrowprops=dict(arrowstyle="->", color="#333"), fontsize=10, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff7e6", ec="#c98a1b"))
ax.set_xticks(x); ax.set_xticklabels(steps)
ax.set_ylabel("trainable params per step (M)")
ax.set_title("(a) Cost of EACH integration step\n(the paper's 91.5% is computed here)", fontsize=11)
ax.legend(frameon=False, fontsize=9)
ax.set_ylim(0, 100)

ax = axes[1]
labels = ["HEAL\n(all 4 steps)", "HM-ViT\n(all 4 steps)"]
vals = [heal_cum[-1], hmvit_cum[-1]]
bars = ax.bar(labels, vals, width=0.5, color=["#4c9be0", "#e07856"], edgecolor=["#1f5b99", "#8a3b28"])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 4, f"{v:.2f} M", ha="center", fontsize=10)
ax.annotate("33.74 vs 196.13\n= 82.8% fewer", xy=(0.5, 120), fontsize=10, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#eef4ff", ec="#1f5b99"))
ax.set_ylabel("total trainable params across all steps (M)")
ax.set_title("(b) CUMULATIVE cost over all integrations\n(includes HEAL's 20.25M base training)", fontsize=11)
ax.set_ylim(0, 230)

fig.suptitle("HEAL Table 2: what the '91.5% fewer training params' number actually measures", fontsize=12, y=1.02)
fig.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-40/multi-agent-driving/heal-xray/fig-train-cost.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
