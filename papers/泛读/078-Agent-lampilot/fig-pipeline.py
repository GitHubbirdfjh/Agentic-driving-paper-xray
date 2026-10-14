"""LaMPilot pipeline: [R*, P] = l([A, I, C]) -> code -> exec -> simulator -> evaluator."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 9

IN, OUT, OP, ALT, WIRE = "#1a6fa3", "#9e5c00", "#0b7a5c", "#6a55c2", "#6e737b"

fig, ax = plt.subplots(figsize=(12, 4.6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4.6)
ax.axis("off")

def box(x, y, w, h, label, sub, fc, ec, fs=9):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       fc=fc, ec=ec, lw=1.4)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2 + 0.12, label, ha="center", va="center",
            fontsize=fs, weight="bold", color=ec)
    ax.text(x + w / 2, y + h / 2 - 0.28, sub, ha="center", va="center",
            fontsize=7.5, color=WIRE)
    return b

def arrow(x1, y1, x2, y2, label=None, ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                        lw=1.4, color=WIRE, linestyle=ls)
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.14, label, ha="center",
                fontsize=7.5, color=WIRE, style="italic")

# three inputs
box(0.15, 2.5, 2.3, 1.5, "API Docs (A)", "23 primitives\nsignatures + usage", "#e2eff5", IN)
box(0.15, 0.6, 2.3, 1.5, "Instruction (I)", '"overtake the\ncar ahead"', "#e2eff5", IN)
box(2.85, 1.55, 2.3, 1.5, "Driving Context (C)", "structured narrative\nfrom vector state", "#e2eff5", IN)

# LLM
box(5.5, 1.15, 2.4, 2.3, "LLM  l", "CoT reasoning R*\nthen code P", "#dcf2e8", OP)
ax.text(6.7, 0.95, "[R*, P] = l([A, I, C])", ha="center", fontsize=8.5, color=OP,
        style="italic")

arrow(2.45, 3.25, 5.5, 2.9)
arrow(2.45, 1.35, 5.5, 1.7)
arrow(5.15, 2.3, 5.5, 2.3)

# exec
box(8.3, 1.55, 1.7, 1.5, "exec(code)", "Python exec()\nAPI scope", "#ebe7f8", ALT)
arrow(7.9, 2.3, 8.3, 2.3, "policy fn")

# simulator loop
box(10.35, 1.55, 1.5, 1.5, "Simulator", "HighwayEnv\nstep loop", "#f8ecd2", OUT)
arrow(10.0, 2.3, 10.35, 2.3)

# evaluator below
box(8.3, 0.15, 3.55, 1.0, "Evaluator", "Score = 0.5 TTC + 0.3 SV + 0.2 TE   |   Driving Score", "#f8ecd2", OUT, fs=8)
arrow(10.0, 1.55, 10.0, 1.15, ls="--")

ax.text(6.0, 4.15, "LaMPilot: LLM writes a policy program; classical control executes it",
        fontsize=11, weight="bold", color="#141413")
ax.text(6.0, 3.82, "the LLM never outputs steering/throttle directly — it only sets strategic parameters in code",
        fontsize=8.5, color=WIRE, ha="center")

plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/078-Agent-lampilot/fig-pipeline.png",
            dpi=150, bbox_inches="tight")
print("saved fig-pipeline.png")
