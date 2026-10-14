# -*- coding: utf-8 -*-
"""Self-drawn figures for the ReAL-AD xray doc. English labels only (no CJK fonts)."""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

INK = "#141413"; GREY = "#6e737b"
BLUE = "#1a6fa3"; ORANGE = "#9e5c00"; GREEN = "#0b7a5c"
YELLOW = "#7a6b00"; RED = "#b34a06"; PURPLE = "#6a55c2"

def box(ax, x, y, w, h, label, color, fs=10, style="round"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size=0.12",
                       ec=color, fc=color, alpha=0.14, lw=1.6)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=fs,
            color=color, weight="bold", linespacing=1.4)

def arrow(ax, x1, y1, x2, y2, label=None, color=GREY, ls="-", lw=1.3):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                        color=color, lw=lw, linestyle=ls, shrinkA=2, shrinkB=4)
    ax.add_patch(a)
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.22, label, ha="center", va="bottom",
                fontsize=8.5, color=color, style="italic")

# ---------------- Figure 1: three-layer pipeline ----------------
fig, ax = plt.subplots(figsize=(15.5, 8.2))
ax.set_xlim(0, 15.5); ax.set_ylim(0, 8.2); ax.axis("off")

ax.text(1.4, 7.7, "STRATEGY  (VLM reasoning)", fontsize=12, weight="bold", color=PURPLE)
ax.text(1.4, 4.9, "TACTICS  (discrete commands)", fontsize=12, weight="bold", color=BLUE)
ax.text(1.4, 2.0, "OPERATION  (coarse -> fine trajectory)", fontsize=12, weight="bold", color=ORANGE)

# Strategy row
box(ax, 1.2, 6.35, 2.6, 0.95, "VLM (frozen)\nfront-view image", PURPLE)
box(ax, 4.4, 6.35, 2.5, 0.95, "strategy text\nT_strategy", PURPLE)
box(ax, 7.5, 6.35, 2.2, 0.95, "E_text + MLP\nF_strategy", GREEN)
box(ax, 10.3, 6.35, 2.4, 0.95, "ego query\nQ_ego", BLUE)
box(ax, 1.2, 5.25, 2.6, 0.8, "GT trajectory\nE_gt -> F_gt", RED, fs=9)
arrow(ax, 3.8, 6.82, 4.4, 6.82, label="M(V, P_strategy)")
arrow(ax, 6.9, 6.82, 7.5, 6.82)
arrow(ax, 9.7, 6.82, 10.3, 6.82, label="Q <- Q + F")
arrow(ax, 3.8, 5.65, 8.0, 6.35, label="L_sim = 1 - cos(.,.)", color=RED, ls="--")
ax.text(13.2, 6.82, "residual\ninjection", fontsize=9, color=GREY, ha="left", va="center")

# Tactics row
box(ax, 1.2, 3.55, 2.6, 0.95, "VLM (frozen)\nfront-view image", PURPLE)
box(ax, 4.4, 3.55, 2.5, 0.95, "command text\nT_command", PURPLE)
box(ax, 7.5, 3.55, 2.2, 0.95, "regex + one-hot\nc_k in {0,1}^n_k", GREEN)
box(ax, 10.3, 3.55, 1.35, 0.95, "g_reactive\n(dir+emerg)", BLUE, fs=9)
box(ax, 12.0, 3.55, 1.65, 0.95, "g_regulatory\n(lane+speed)", BLUE, fs=9)
arrow(ax, 3.8, 4.02, 4.4, 4.02, label="M(V, P_command)")
arrow(ax, 6.9, 4.02, 7.5, 4.02)
arrow(ax, 9.7, 4.02, 10.3, 4.02)
arrow(ax, 11.65, 4.02, 12.0, 4.02)

# Operation row
box(ax, 1.2, 0.75, 2.6, 0.95, "f_ego + g_reactive", BLUE)
box(ax, 4.4, 0.75, 2.5, 0.95, "DistEncoder_theta\nz^c ~ N(mu^c, sigma^c)", GREEN, fs=9)
box(ax, 7.5, 0.75, 2.6, 0.95, "DistEncoder_phi\nz^f ~ N(mu^f, sigma^f)", GREEN, fs=9)
box(ax, 10.7, 0.75, 2.2, 0.95, "GRU + MLP\npred_coarse / pred_fine", ORANGE, fs=9)
box(ax, 13.3, 0.75, 1.7, 0.95, "Bezier(gt)\ngt_coarse", YELLOW, fs=9)
arrow(ax, 3.8, 1.22, 4.4, 1.22)
arrow(ax, 6.9, 1.22, 7.5, 1.22, label="[f_ego; z^c; g_reg]")
arrow(ax, 10.1, 1.22, 10.7, 1.22)
arrow(ax, 12.9, 1.22, 13.3, 1.22, label="supervise", color=YELLOW, ls="--")
arrow(ax, 11.0, 3.55, 2.6, 1.7, color=BLUE, ls="--")
arrow(ax, 12.8, 3.55, 8.4, 1.7, color=BLUE, ls="--")
arrow(ax, 11.5, 6.35, 2.6, 1.7, color=BLUE, ls="--")

ax.text(0.2, 0.25, "fig-pipeline: ReAL-AD = VLM text -> strategy residual + discrete tactic commands -> coarse-to-fine variational decoder. "
        "Every layer has its own supervision (L_sim, command supervision, Bezier coarse GT).",
        fontsize=9, color=GREY, ha="left", va="bottom", wrap=True)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-40/vla-driving/real-ad-xray/fig-pipeline.png", dpi=150)
plt.close(fig)

# ---------------- Figure 2: Bezier coarse vs fine ----------------
# Construct a plausible 3s / 6-step fine trajectory: lane change with a wiggle
dP = np.array([[0.9, 0.02], [1.0, 0.10], [1.1, 0.35], [1.2, 0.75], [1.3, 0.85], [1.35, 0.80]])
P = np.cumsum(dP, axis=0)                      # absolute positions, eq. (21)
C = np.stack([P[0], P[2], P[3], P[5]])          # control points P1, P3, P4, P6
ts = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
B = sum(C[k] * (np.math.comb(3, k) * ts[:, None]**k * (1-ts[:, None])**(3-k)) for k in range(4))  # eq. (22)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
ax.plot(P[:, 0], P[:, 1], "o-", color=ORANGE, lw=2, ms=7, label="fine GT: 6 offsets")
ax.plot(C[:, 0], C[:, 1], "s--", color=GREY, lw=1.2, ms=6, label="control pts P1,P3,P4,P6")
ax.plot(B[:, 0], B[:, 1], "^-", color=YELLOW, lw=2, ms=7, label="Bezier coarse: 6 samples")
for i, (x, y) in enumerate(P):
    ax.text(x, y + 0.13, f"P{i+1}", fontsize=9, ha="center", color=ORANGE, weight="bold")
ax.set_aspect("equal"); ax.grid(alpha=0.25)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("coarse supervision = cubic Bezier through 4 control points", fontsize=11)
ax.legend(fontsize=9, loc="upper left")

ax = axes[1]
steps = np.arange(1, 7)
fine_yaw = np.degrees(np.arctan2(dP[:, 1], dP[:, 0]))
dB = np.diff(B, axis=0, prepend=B[0:1] - np.array([B[0,0]-dP[0,0], B[0,1]-dP[0,1]]))
coarse_yaw = np.degrees(np.arctan2(dB[:, 1], dB[:, 0]))
ax.plot(steps, fine_yaw, "o-", color=ORANGE, lw=2, ms=7, label="fine: per-step heading")
ax.plot(steps, coarse_yaw, "^-", color=YELLOW, lw=2, ms=7, label="coarse: smoothed heading")
ax.set_xticks(steps); ax.set_xlabel("timestep (0.5 s each)")
ax.set_ylabel("heading (deg)")
ax.set_title("what the coarse layer sees: global trend, no wiggle", fontsize=11)
ax.grid(alpha=0.25); ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-40/vla-driving/real-ad-xray/fig-bezier.png", dpi=150)
plt.close(fig)
print("figures saved")
