"""Self-drawn: why the std in GRPO's advantage silently kills learning.

Two rollout groups from a weak SFT policy. Same "best rollout" logic,
different advantage under GRPO vs Dr. GRPO.

Group A (easy, e.g. straight road):   rewards ~ N(0.9525, 0.0106^2), best r = 0.97
Group B (learnable-hard, e.g. turn):  rewards ~ N(0.40, 0.141^2),   best r = 0.60

GRPO advantage  = (r - mean) / std
DrGRPO advantage = (r - mean)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- numbers (computed in docstring) ----
labels = ["Group A: easy\n(mean .95, std .01)", "Group B: learnable-hard\n(mean .40, std .14)"]
abs_gain = [0.0175, 0.20]
grpo_adv = [1.65, 1.42]
dr_adv = [0.0175, 0.20]

x = np.arange(len(labels))
w = 0.36

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
fig.patch.set_facecolor("white")

# left: absolute gain of the best rollout (what actually happened in the world)
ax = axes[0]
b = ax.bar(x, abs_gain, width=0.55, color=["#9db8d2", "#d98c5f"], edgecolor="black", linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("absolute reward gain of best rollout", fontsize=11)
ax.set_title("(a) What the rollouts actually achieved", fontsize=12, pad=10)
ax.set_ylim(0, 0.24)
for i, v in enumerate(abs_gain):
    ax.text(i, v + 0.008, f"+{v}", ha="center", fontsize=11, fontweight="bold")
ax.text(0.5, 0.93, "the hard sample improved 11x more in absolute terms",
        transform=ax.transAxes, ha="center", fontsize=10, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f4ed", ec="none"))

# right: advantage each method assigns to that same best rollout
ax = axes[1]
ax.bar(x - w/2, grpo_adv, width=w, color="#4a7bb8", edgecolor="black", linewidth=0.8, label="GRPO: (r-mean)/std")
ax.bar(x + w/2, dr_adv, width=w, color="#c26a3d", edgecolor="black", linewidth=0.8, label="Dr. GRPO: (r-mean)")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("advantage fed to the policy gradient", fontsize=11)
ax.set_title("(b) What each optimizer tells the policy", fontsize=12, pad=10)
ax.legend(frameon=False, fontsize=10, loc="lower right")
for i, (g, d) in enumerate(zip(grpo_adv, dr_adv)):
    ax.text(i - w/2, g + 0.05, f"{g}", ha="center", fontsize=10, fontweight="bold")
    ax.text(i + w/2, d + 0.06, f"{d}", ha="center", fontsize=10, fontweight="bold")
ax.text(0.98, 0.96, "GRPO amplifies the easy group;\nDr. GRPO lets the hard group dominate",
        transform=ax.transAxes, ha="right", va="top", fontsize=10, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f4ed", ec="none"))

for a in axes:
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/030-VLA-nord/fig-difficulty-bias.png",
            dpi=150, bbox_inches="tight")
print("saved fig-difficulty-bias.png")
