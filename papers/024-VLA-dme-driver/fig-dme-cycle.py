"""Self-drawn figure for the DME-Driver xray doc.

A concrete walk-through of one decision cycle (the intersection / red-light
example from the paper's Figure 3), showing exactly what flows where:
the Decision-Maker's four text outputs, the Driver Logic Encoder
(BERT + LogicalFusioner cross-attention), and the Executor's path from
BEV features to control signals, with the consistency-loss feedback arrow.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(15, 6.4))
ax.set_xlim(0, 15)
ax.set_ylim(0, 6.4)
ax.axis("off")

IN, OP, OUT, ALT = "#1a6fa3", "#0b7a5c", "#9e5c00", "#6a55c2"

def box(x, y, w, h, text, fc, ec, fs=9, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       fc=fc, ec=ec, alpha=alpha, linewidth=1.2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, wrap=True, linespacing=1.35)
    return (x + w / 2, y + h / 2, x, y, w, h)

def arrow(x1, y1, x2, y2, color="#6e737b", ls="-", width=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 color=color, linestyle=ls, linewidth=width,
                                 mutation_scale=12, shrinkA=0, shrinkB=2))

# ---- column 1: inputs ----
box(0.2, 4.3, 2.2, 1.2, "Current view\n(multiview image)", "#e2eff5", IN)
box(0.2, 2.9, 2.2, 1.2, "Previous 3 key frames\n$F_{t-3..t}$", "#e2eff5", IN)
box(0.2, 1.5, 2.2, 1.2, "Status: speed, steer\n+ text prompt $T_p$", "#e2eff5", IN)
ax.text(1.3, 5.75, "inputs", fontsize=11, weight="bold", ha="center")

# ---- column 2: Decision-Maker ----
box(3.1, 2.2, 2.6, 2.6, "Decision-Maker\nLVLM (LLaVA + LLaMA2)\nimitation learning\non HBD dialogues", "#ebe7f8", ALT)
ax.text(4.4, 5.15, "Decision-Maker", fontsize=11, weight="bold", ha="center")
arrow(2.4, 4.9, 3.1, 4.4)
arrow(2.4, 3.5, 3.1, 3.5)
arrow(2.4, 2.1, 3.1, 2.6)

# ---- column 3: four text outputs ----
outs = [
    (6.4, 4.5, "Scene description\n\"Here is an intersection.\""),
    (6.4, 3.4, "Gaze\n\"I focus on the Traffic Light.\""),
    (6.4, 2.3, "Decision\n\"I will stay and wait.\""),
    (6.4, 1.2, "Driver logic\n\"The driver should stay\nand wait.\""),
]
for x, y, t in outs:
    box(x, y, 2.5, 0.95, t, "#f8ecd2", OUT, fs=8.5)
    arrow(5.7, 3.5, x, y + 0.475, color=OUT)
ax.text(7.65, 5.75, "natural-language interface", fontsize=11, weight="bold", ha="center")

# ---- column 4: Driver Logic Encoder ----
box(9.6, 2.6, 2.4, 1.8,
    "Driver Logic Encoder\n$E_{bert}$ text encoder\nLogicalFusioner:\n$B' = \\mathrm{MHA}(Q{=}B,K{=}T,V{=}T){+}B$",
    "#dcf2e8", OP, fs=8.5)
ax.text(10.8, 4.75, "text prior → BEV", fontsize=11, weight="bold", ha="center")
for _, y, _ in outs:
    arrow(8.9, y + 0.475, 9.6, 3.5, color=OP)

# ---- column 5: Executor ----
box(12.5, 2.6, 1.4, 1.8, "Executor\n(UniAD-based)\nTrack/Map\nMotion/Occ\n→ Planner", "#e2eff5", IN, fs=8)
arrow(12.0, 3.5, 12.5, 3.5, color=IN)

# control signal output
box(12.55, 0.4, 1.3, 0.9, "Brake: 100%\nSteer: 0°", "#f9e3d6", "#b34a06", fs=9)
arrow(13.2, 2.6, 13.2, 1.3, color="#b34a06")

# consistency loss feedback arrow (training time): from Planner back to the
# Decision text box — penalize control signals contradicting the decision
arrow(12.5, 2.9, 9.0, 2.5, color="#b34a06", ls=(0, (4, 3)))
ax.text(10.75, 2.0, "consistency loss (train):\npenalize control signals\nthat contradict the decision",
        fontsize=8, color="#b34a06", ha="center",
        bbox=dict(fc="white", ec="#b34a06", alpha=0.95, boxstyle="round,pad=0.25"))

ax.text(7, 0.15,
        "Walk-through of one DME-Driver decision cycle (paper Fig.3 intersection example, red light).",
        fontsize=10, style="italic", ha="center", color="#3d3d3a")
fig.tight_layout()
fig.savefig("fig-dme-cycle.png", dpi=150, bbox_inches="tight")
print("saved fig-dme-cycle.png")
