"""Declarative vs aligned interpretability, redrawn from Fig.1 with the failure mode shown."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 8.5

IN, OUT, LOSS, WIRE = "#1a6fa3", "#9e5c00", "#b34a06", "#6e737b"

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

for ax, title in zip(axes, ["(a) declarative: language justifies behavior",
                            "(b) aligned: language reads the pipeline's tokens"]):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title(title, fontsize=10, weight="bold")

def box(ax, x, y, w, h, label, sub, fc, ec, fs=9):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", fc=fc, ec=ec, lw=1.3)
    ax.add_patch(b)
    ax.text(x+w/2, y+h/2+0.15, label, ha="center", va="center", fontsize=fs,
            weight="bold", color=ec)
    ax.text(x+w/2, y+h/2-0.3, sub, ha="center", va="center", fontsize=7.5, color=WIRE)

def arrow(ax, x1, y1, x2, y2, ls="-", c=WIRE, label=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.3, color=c, linestyle=ls))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2+0.2, label, ha="center", fontsize=7.5,
                color=c, style="italic")

# (a) declarative
ax = axes[0]
box(ax, 0.5, 4.6, 3.0, 1.3, "AD model", "black box", "#e2eff5", IN)
box(ax, 6.0, 4.6, 3.2, 1.3, "LLM", "sees only\nimages / BEV", "#f8ecd2", OUT)
arrow(ax, 3.5, 5.25, 6.0, 5.25, ls="--", label="no token flow")
ax.text(5.0, 3.4, '"The car slows down\nto let the pedestrian pass."',
        ha="center", fontsize=9, style="italic", color="#3d3d3a",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f9e3d6", ec=LOSS))
ax.text(5.0, 1.9, "sounds plausible even when the AD model\nsaw no pedestrian -> hallucinated justification",
        ha="center", fontsize=8, color=LOSS)
ax.text(5.0, 0.9, "criterion: is the text grounded in the AD model's\nintermediate outputs?  here: no",
        ha="center", fontsize=8, color=WIRE, style="italic")

# (b) aligned
ax = axes[1]
box(ax, 0.4, 4.7, 2.4, 1.2, "perception", "track tokens", "#e2eff5", IN)
box(ax, 3.6, 4.7, 2.4, 1.2, "prediction", "motion tokens", "#e2eff5", IN)
box(ax, 6.8, 4.7, 2.4, 1.2, "planning", "plan tokens", "#e2eff5", IN)
box(ax, 3.3, 2.9, 3.0, 1.1, "token mixer", "adapt + fuse", "#dcf2e8", "#0b7a5c")
box(ax, 3.3, 1.1, 3.0, 1.2, "LLM", "+ alignment task", "#f8ecd2", OUT)
arrow(ax, 1.6, 4.7, 4.0, 4.0)
arrow(ax, 4.8, 4.7, 4.7, 4.0)
arrow(ax, 8.0, 4.7, 5.4, 4.0)
arrow(ax, 4.8, 2.9, 4.8, 2.3)
ax.text(5.0, 0.35, '"Ego yields: track token #7 is a pedestrian\nat 8.2 m, 12 o\'clock; plan holds position."',
        ha="center", fontsize=8.5, style="italic", color="#3d3d3a",
        bbox=dict(boxstyle="round,pad=0.3", fc="#dcf2e8", ec="#0b7a5c"))

plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/077-Agent-hint-ad/fig-declarative-aligned.png",
            dpi=150, bbox_inches="tight")
print("saved fig-declarative-aligned.png")
