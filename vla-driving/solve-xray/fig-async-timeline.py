import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "cm", "axes.unicode_minus": False,
})
IN, OUT, OP, WIRE = "#1a6fa3", "#9e5c00", "#0b7a5c", "#6e737b"
IN_F, OUT_F, OP_F = "#e2eff5", "#f8ecd2", "#dcf2e8"

fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")

def box(x, y, w, h, face, edge, label, fs=10, tc=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                       facecolor=face, edgecolor=edge, linewidth=1.5)
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=fs,
            color=tc or edge, weight="bold")

def arrow(x1, y1, x2, y2, c=WIRE, ls="-", wd=1.3):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 color=c, linewidth=wd, linestyle=ls, shrinkA=0, shrinkB=2))

xs = [2.0, 4.0, 6.0, 8.0]
ax.text(0.15, 0.45, "time", fontsize=11, color=WIRE, style="italic")
for x, t in zip(xs, ["t-2", "t-1", "t", "t+1"]):
    ax.text(x, 0.65, t, fontsize=11, color=WIRE, ha="center")
    ax.plot([x, x], [0.85, 4.0], color=WIRE, lw=0.8, ls=":", alpha=0.55)

# VLM row
ax.text(0.12, 3.45, "VLM\n(slow)", fontsize=11, color=OUT, weight="bold", va="center")
box(2.0, 3.1, 4.0, 0.7, OUT_F, OUT, "VLM predicts long-horizon traj  $T_{vlm}(t_0)$", fs=9.5)
ax.annotate("", xy=(2.0, 2.95), xytext=(6.0, 2.95),
            arrowprops=dict(arrowstyle="<->", color=OUT, lw=1.2, linestyle="--"))
ax.text(4.0, 2.68, "horizon > VLM latency + E2E horizon", fontsize=8.5, color=OUT, ha="center")
box(6.35, 3.1, 1.15, 0.7, OUT_F, OUT, "memory", fs=10)
arrow(6.0, 3.45, 6.35, 3.45)

# E2E row
ax.text(0.12, 1.7, "E2E\n(fast)", fontsize=11, color=OP, weight="bold", va="center")
for x in xs:
    box(x-0.5, 1.35, 1.0, 0.7, OP_F, OP, "plan", fs=10)
ax.text(5.0, 1.05, "every frame: fetch latest delayed VLM traj as init prior", fontsize=9,
        color=WIRE, ha="center", style="italic")

# delayed prior into current frame's planner
arrow(6.9, 3.1, 6.9, 2.35, ls="--")
arrow(6.9, 2.35, 6.45, 2.06, ls="--")
ax.text(7.35, 2.6, "delayed prior", fontsize=8.5, color=WIRE, style="italic")

# output of current frame
box(8.75, 1.35, 1.55, 0.7, IN_F, IN, "final traj", fs=10)
ax.plot([6.5, 9.52], [2.18, 2.18], color="#6e737b", lw=1.3)
arrow(9.52, 2.18, 9.52, 2.06)
ax.text(9.52, 1.05, "queries x collector queries -> MLP", fontsize=8.5, color=WIRE, ha="center", style="italic")

plt.tight_layout(pad=0.6)
plt.savefig("fig-async-timeline.png", dpi=150, facecolor="white", bbox_inches="tight")
print("ok")
