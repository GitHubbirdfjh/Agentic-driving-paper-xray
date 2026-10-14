# -*- coding: utf-8 -*-
"""080 Fig1: 三种 LLM 参与模式（Mode I/II/III）的调度时间线对比。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, axes = plt.subplots(3, 1, figsize=(13, 6.4), sharex=True)
fig.suptitle("Fig.3 of the paper: three modes of LLM involvement, from a scheduling perspective",
             fontsize=12, fontweight="bold")
T = 10
colors = {"llm": "#e38b6b", "pl": "#a9e3cc", "tc": "#cdc2f0"}

titles = [
    "(a) Mode I: Static Configuration -- LLM decides once at startup, parameters stay fixed",
    "(b) Mode II: Dynamic Decision -- LLM is queried continuously (e.g. every step)",
    "(c) Mode III (Ours): Hybrid Scheduling -- ONE LLM call emits a script; async triggers switch planners",
]
for ax, title in zip(axes, titles):
    ax.set_xlim(0, T); ax.set_ylim(0, 3); ax.set_yticks([])
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.set_xlabel("time (driving steps)" if title.startswith("(c)") else "")

# Mode I
ax = axes[0]
b = FancyBboxPatch((0.1, 1.2), 1.4, 0.9, boxstyle="round,pad=0.05", fc=colors["llm"], ec="#ad493e")
ax.add_patch(b); ax.text(0.8, 1.65, "LLM sets params", ha="center", va="center", fontsize=8.5, fontweight="bold")
for i in range(1, 11):
    b = FancyBboxPatch((i-0.35, 0.4), 0.7, 0.9, boxstyle="round,pad=0.02", fc=colors["pl"], ec="#0b7a5c")
    ax.add_patch(b)
ax.text(5.2, 0.85, "AD algorithm runs with fixed params", ha="center", fontsize=8.5)
ax.text(5.2, 2.35, "no context-aware switching", ha="center", fontsize=8.5, color="#ad493e")

# Mode II
ax = axes[1]
for i in range(0, 10):
    b = FancyBboxPatch((i+0.05, 1.5), 0.5, 0.9, boxstyle="round,pad=0.02", fc=colors["llm"], ec="#ad493e")
    ax.add_patch(b)
    b = FancyBboxPatch((i+0.6, 0.4), 0.35, 0.9, boxstyle="round,pad=0.02", fc=colors["pl"], ec="#0b7a5c")
    ax.add_patch(b)
ax.text(5.0, 2.6, "LLM call every step: high overhead, latency, coherence risk", ha="center", fontsize=8.5, color="#ad493e")
ax.text(0.3, 1.95, "LLM", fontsize=8, color="#ad493e", fontweight="bold")
ax.text(0.32, 0.85, "act", fontsize=8, color="#0b7a5c")

# Mode III
ax = axes[2]
b = FancyBboxPatch((0.1, 1.5), 1.2, 0.9, boxstyle="round,pad=0.05", fc=colors["llm"], ec="#ad493e")
ax.add_patch(b); ax.text(0.7, 1.95, "LLM: 1 call", ha="center", va="center", fontsize=8.5, fontweight="bold")
ax.text(0.7, 1.15, "behavior seq +\ntrigger script", ha="center", fontsize=7.5)
plans = [(1.5, "Decel", 1.5), (3.2, "Right LC", 1.6), (5.0, "Lane keep", 1.4)]
for (x0, name, w) in plans:
    b = FancyBboxPatch((x0, 0.4), w, 0.9, boxstyle="round,pad=0.02", fc=colors["pl"], ec="#0b7a5c")
    ax.add_patch(b); ax.text(x0+w/2, 0.85, name, ha="center", fontsize=8)
tcs = [(3.2, "TC: gap>20m"), (5.0, "TC: lane clear")]
for (x0, label) in tcs:
    ax.annotate(label, xy=(x0, 1.3), xytext=(x0, 2.5),
                arrowprops=dict(arrowstyle="-|>", color="#6a55c2", lw=1.5),
                fontsize=8, color="#6a55c2", ha="center")
ax.text(8.2, 1.95, "adaptive switching via async triggers\n(overhead of Mode I, responsiveness of Mode II)",
        fontsize=8.5, color="#141413")

for ax in axes:
    ax.spines[["top", "right", "left"]].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/080-Agent-open-ended-scheduling/fig-modes.png",
            dpi=160, bbox_inches="tight", facecolor="white")
print("saved fig-modes.png")
