# -*- coding: utf-8 -*-
"""fig-reward.py: two panels.
(A) A worked weighted-Levenshtein example for R_speed on a 4-step sequence.
(B) The contrastive R_tool mechanism in AMS-RL: tool path vs text-only baseline.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

LOSS, LOSS_F = "#b34a06", "#f9e3d6"
ALT, ALT_F = "#6a55c2", "#ebe7f8"
IN, IN_F = "#1a6fa3", "#e2eff5"
OP, OP_F = "#0b7a5c", "#dcf2e8"
WIRE = "#6e737b"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.6), gridspec_kw={"width_ratios": [1.05, 1]})

# ================= (A) worked edit distance =================
ax1.set_xlim(0, 10); ax1.set_ylim(0, 10); ax1.axis("off")
ax1.text(5, 9.55, "(A) R_speed on a worked 4-step example", fontsize=11.5, weight="bold", ha="center")

gt = ["Keep", "Keep", "Decel", "Stop"]
pr = ["Keep", "Decel", "Decel", "Stop"]
ws = ["w=0.7", "w=0.7", "w=1.3", "w=1.1"]   # weights indexed by TRUE action at t (mean 1)
costs = ["1−0.7=0.3", "1−0=1.0", "1−1.3=−0.3", "1−1.1=−0.1"]
ax1.text(0.3, 8.8, "Ground truth (A):", fontsize=9.5, color=IN, weight="bold")
for i, (g, p, w, cst) in enumerate(zip(gt, pr, ws, costs)):
    x = 1.1 + i * 1.95
    match = (g == p)
    col = OP if match else LOSS
    face = OP_F if match else LOSS_F
    r = mpatches.FancyBboxPatch((x, 7.15), 1.7, 1.0, boxstyle="round,pad=0.04",
                                facecolor=face, edgecolor=col, linewidth=1.5)
    ax1.add_patch(r)
    ax1.text(x + 0.85, 7.95, g, ha="center", va="center", fontsize=9.5, weight="bold", color=col)
    ax1.text(x + 0.85, 7.32, w, ha="center", va="center", fontsize=8, color=WIRE)
    r2 = mpatches.FancyBboxPatch((x, 5.75), 1.7, 1.0, boxstyle="round,pad=0.04",
                                 facecolor=face, edgecolor=col, linewidth=1.5)
    ax1.add_patch(r2)
    ax1.text(x + 0.85, 6.4, p, ha="center", va="center", fontsize=9.5, weight="bold", color=col)
    ax1.text(x + 0.85, 5.35, "subst cost", ha="center", fontsize=7.5, color=WIRE)
    ax1.text(x + 0.85, 5.0, cst, ha="center", fontsize=8.5, weight="bold", color=col)
ax1.text(0.3, 6.9, "Prediction (A^):", fontsize=9.5, color=ALT, weight="bold")

ax1.text(0.3, 4.25, "t=2 mismatches: S = 1(Keep=Decel)·w = 0 → cost 1.\n"
         "Matches pay 1−w: rare Decel right → −0.3 bonus.\n"
         "D(A,A^) = 0.3 + 1.0 − 0.3 − 0.1 = 0.9\n"
         "R_speed = 1 − 0.9/max(4,4) = 0.775   (unweighted: 0.75)", fontsize=9, color="#3d3d3a")
ax1.text(0.3, 2.45, "R_acc = 0.7·R_speed + 0.3·R_traj;  del/ins cost 0.6.\n"
         "Weights: w ∝ (f̄t/fc,t)^γ, γ=0.5, clip [0.7,1.3], mean-1 per t.\n"
         "Incentive: get the RARE actions right — that is where the bonus is.",
         fontsize=8.8, color="#3d3d3a")

# ================= (B) contrastive tool reward =================
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis("off")
ax2.text(5, 9.55, "(B) Contrastive R_tool in AMS-RL (one group, G=4)", fontsize=11.5, weight="bold", ha="center")

labels = ["text-1", "text-2", "tool-1", "tool-2"]
accs = [0.62, 0.70, 0.88, 0.75]
calls = [0, 0, 2, 3]
base = 0.66  # mean of text paths
xs = np.linspace(0.9, 7.6, 4)
for i, (lab, a, n) in enumerate(zip(labels, accs, calls)):
    col = IN if n == 0 else ALT
    face = IN_F if n == 0 else ALT_F
    h = a * 5.2
    r = mpatches.FancyBboxPatch((xs[i], 1.6), 1.3, h, boxstyle="round,pad=0.03",
                                facecolor=face, edgecolor=col, linewidth=1.8)
    ax2.add_patch(r)
    ax2.text(xs[i] + 0.65, 1.25, lab, ha="center", fontsize=9, color=col, weight="bold")
    ax2.text(xs[i] + 0.65, 1.6 + h + 0.18, f"{a:.2f}", ha="center", fontsize=9, weight="bold", color=col)
    if n:
        ax2.text(xs[i] + 0.65, 1.6 + h/2, f"{n} calls\ncost {n}×0.125",
                 ha="center", va="center", fontsize=8, color=ALT)

ax2.plot([0.4, 9.2], [1.6 + base*5.2]*2, color=IN, lw=1.6, ls="--")
ax2.text(9.3, 1.6 + base*5.2, "text baseline\nR̄_acc = 0.66", fontsize=8.5, color=IN, va="center")

ax2.text(0.4, 8.0, "tool-1: (0.88−0.66) − 2×0.125 = −0.03 → ≤0 → no bonus\n"
         "tool-2: (0.75−0.66) − 3×0.125 = −0.285 → punished\n"
         "margin = N_i · 0.125  (\"min effective gain\": fix ≥1 of 4 steps = 0.25, / ~2 calls)",
         fontsize=8.6, color="#3d3d3a")
ax2.text(0.4, 6.15, "Positive case: gain 0.30 with 1 call → 0.30−0.125 = +0.175,\n"
         "× per-call quality (parse 0.2 + found 0.2 + params 0.2 + exec 0.4 ≤ 1.0)\n"
         "→ R_tool ∈ [−0.2, 0.2]. No text path in group → R_tool = 0.", fontsize=8.6, color="#3d3d3a")

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/071-Agent-driveagent-r1/fig-reward.png",
            dpi=150, bbox_inches="tight")
print("saved")
