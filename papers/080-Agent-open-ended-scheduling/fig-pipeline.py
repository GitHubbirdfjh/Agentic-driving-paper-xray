# -*- coding: utf-8 -*-
"""080 Fig2: 三阶段管线 + 时间尺度分离（slow semantic vs fast control loop）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13, 6.4))
ax.set_xlim(0, 13); ax.set_ylim(0, 6.4); ax.axis("off")

C_IN, C_OP, C_OUT, C_ALT = "#1a6fa3", "#0b7a5c", "#9e5c00", "#6a55c2"

def box(x, y, w, h, text, sub="", fc="#e2eff5", ec=C_IN, fs=10):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1.6)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2 + (0.18 if sub else 0), text, ha="center", va="center",
            fontsize=fs, fontweight="bold", color="#141413")
    if sub:
        ax.text(x + w/2, y + h/2 - 0.38, sub, ha="center", va="center", fontsize=8.3, color="#3d3d3a")

def arrow(x1, y1, x2, y2, label="", color="#6e737b", style="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 lw=1.5, color=color, linestyle=style))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.2, label, ha="center", fontsize=8.5, color=color)

# 上层：LLM 慢回路
bg = FancyBboxPatch((0.2, 3.55), 12.6, 2.45, boxstyle="round,pad=0.05", fc="#f8ecd2", ec=C_OUT, lw=1.2, ls="--")
ax.add_patch(bg)
ax.text(0.5, 5.7, "HIGH-LEVEL, LOW-FREQUENCY: LLM semantic reasoning (called ONCE per instruction)", fontsize=10, fontweight="bold", color=C_OUT)

box(0.6, 4.0, 2.4, 1.4, "Stage 1\nInstruction intent", 'f_phi(gamma, o_0)\n-> [Decelerate, Right LC,\n   Lane Keeping]', fc="#f8ecd2", ec=C_OUT)
box(5.2, 4.0, 2.4, 1.4, "Stage 2\nScheduling script", "coroutine: yield from\nasync triggers (TCs)", fc="#ebe7f8", ec=C_ALT)
arrow(3.0, 4.7, 5.2, 4.7, label="behavior sequence")
box(9.6, 4.0, 2.8, 1.4, "Script = audit trail", "human-readable: text ->\nplanner switch, inspectable", fc="#f8ecd2", ec=C_OUT)
arrow(7.6, 4.7, 9.6, 4.7)

# 下层：快回路
bg2 = FancyBboxPatch((0.2, 0.2), 12.6, 2.9, boxstyle="round,pad=0.05", fc="#dcf2e8", ec=C_OP, lw=1.2, ls="--")
ax.add_patch(bg2)
ax.text(0.5, 2.82, "LOW-LEVEL, HIGH-FREQUENCY: real-time feedback-driven schedule-plan-control loop", fontsize=10, fontweight="bold", color=C_OP)

box(0.6, 0.9, 2.6, 1.5, "Async triggers\nmonitor scene graph", "gap > 20 m and ...\n-> switch planner", fc="#ebe7f8", ec=C_ALT)
box(4.0, 0.9, 2.6, 1.5, "5 MPC motion planners", "receding-horizon opt\nLaneKeep/LC/Accel/Decel", fc="#dcf2e8", ec=C_OP)
box(7.4, 0.9, 2.6, 1.5, "LQR controller", "trajectory ->\nsteer/throttle/brake", fc="#dcf2e8", ec=C_OP)
box(10.8, 0.9, 1.6, 1.5, "Vehicle", "numerical\ncontrol", fc="#e2eff5", ec=C_IN)
arrow(3.2, 1.65, 4.0, 1.65); arrow(6.6, 1.65, 7.4, 1.65); arrow(10.0, 1.65, 10.8, 1.65)
arrow(6.4, 4.0, 6.4, 2.75, label="script installs triggers + planners", style="--", color=C_ALT)

ax.text(0.3, 6.15, "Decoupling by timescale: LLM latency only degrades task completion, never safety",
        fontsize=12, fontweight="bold", color="#141413")

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/080-Agent-open-ended-scheduling/fig-pipeline.png",
            dpi=160, bbox_inches="tight", facecolor="white")
print("saved fig-pipeline.png")
