"""Self-drawn figure for DriveDreamer4D xray:
NTGM in ego-start coordinates: bird's-eye view of the offset search
(Algorithm 1) with a concrete small example. Bright theme."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

IN = "#1a6fa3"; OUT = "#9e5c00"; OP = "#0b7a5c"; LOSS = "#b34a06"
ALT = "#6a55c2"; WIRE = "#6e737b"
IN_FILL = "#e2eff5"; OUT_FILL = "#f8ecd2"; OP_FILL = "#dcf2e8"; ALT_FILL = "#ebe7f8"
LOSS_FILL = "#f9e3d6"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6),
                               gridspec_kw={"width_ratios": [1.25, 1]})

# ---------- left: bird's-eye view ----------
ax1.set_xlim(-1, 14); ax1.set_ylim(-2.5, 8); ax1.set_aspect("equal")
ax1.set_xlabel("x: vehicle heading → (m)", fontsize=10)
ax1.set_ylabel("y: left ← (m)", fontsize=10)
ax1.set_title("NTGM proposal in EgoStart coordinates (lane change example)", fontsize=11, weight="bold")

# drivable area
ax1.add_patch(mpatches.Rectangle((0, 0), 13.5, 7.0, fc="#f2f0e8", ec=WIRE, lw=1.2, ls="--"))
ax1.text(12.8, 6.6, "Broad\n(drivable)", ha="right", fontsize=9, color=WIRE, style="italic")

# lane divider lines
for y in (1.75, 3.5, 5.25):
    ax1.plot([0, 13.5], [y, y], color=WIRE, lw=1.0, ls=(0, (5, 4)), alpha=0.7)

# original trajectory: straight along x at y=0.875 (lane center)
xs = np.array([0, 2.5, 5, 7.5, 10])
yo = np.full_like(xs, 0.875)
ax1.plot(xs, yo, color=IN, lw=2.5, marker="o", ms=6, label="original T_ori (y const)")
for i, (x, y) in enumerate(zip(xs, yo)):
    ax1.annotate(f"p{i}", (x, y), textcoords="offset points", xytext=(4, 8), fontsize=8, color=IN)

# offset accumulation: new trajectory shifts left (+y) lane change
offsets = [0.0, 0.08, 0.20, 0.38, 0.62]
yn = yo + np.array(offsets)
ax1.plot(xs, yn, color=OUT, lw=2.5, marker="s", ms=6, label="novel T_novel (y grows)")
for i, (x, y, o) in enumerate(zip(xs, yn, offsets)):
    ax1.annotate(f"Δ={o:.2f}", (x, y), textcoords="offset points", xytext=(4, -14), fontsize=8, color=OUT)

# rejected candidate: would collide with another vehicle
ax1.plot([9], [0.875], color=LOSS, marker="X", ms=13, mew=2)
ax1.annotate("candidate p2′ (Δ too big)\nrejected: ‖p−o_j‖ < d_min",
             (9, 0.875), xytext=(-190, -52), textcoords="offset points",
             fontsize=8.5, color=LOSS, weight="bold",
             arrowprops=dict(arrowstyle="->", color=LOSS, lw=1.5))
# other vehicle
car = mpatches.Rectangle((8.3, 0.05), 1.5, 0.8, fc=LOSS_FILL, ec=LOSS, lw=1.6)
ax1.add_patch(car)
ax1.text(9.05, 0.45, "o_j", ha="center", va="center", fontsize=9, color=LOSS, weight="bold")
# dmin double arrow
ax1.annotate("", xy=(9, 0.05), xytext=(9, 0.875),
             arrowprops=dict(arrowstyle="<|-|>", color=LOSS, lw=1.4))
ax1.text(9.35, 0.5, "d_min", fontsize=9, color=LOSS)
ax1.legend(fontsize=9, loc="upper left")
ax1.grid(alpha=0.25)

# ---------- right: Algorithm 1 walkthrough ----------
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis("off")
ax2.set_title("Algorithm 1, one frame step (worked example)", fontsize=11, weight="bold")
steps = [
    ("1. take p_world[i], convert to EgoStart:", "p = (7.50, 0.875)", IN),
    ("2. propose offset (uniform in [0, MaxOffset]):", "MaxOffset=0.10 → Δ=0.13", OP),
    ("   candidate p′ = p + [0, Δ, 0]", "p′ = (7.50, 1.005)", OP),
    ("3a. inside drivable area?", "yes: p′ ∈ B_road", OP),
    ("3b. distance to every agent o_j?", "‖p′−o_j‖ = 0.42 ≥ d_min=0.40 ✓", OP),
    ("→ accept: Offset=0.13, append p′", "", OP),
    ("", "", None),
    ("next frame: candidate Δ too big,", "‖p′−o_j‖ = 0.31 < 0.40 ✗", LOSS),
    ("→ reject, MaxOffset /= 2 (0.10→0.05),", "retry smaller Δ", LOSS),
    ("   repeat until a safe Δ passes", "", LOSS),
]
y = 9.2
for title, val, color in steps:
    if title == "" and val == "":
        y -= 0.35
        continue
    ax2.text(0.4, y, title, fontsize=9.5, color=color if color else "black")
    if val:
        ax2.text(0.9, y - 0.45, val, fontsize=9.5, color=color, weight="bold",
                 bbox=dict(fc="white", ec=color, lw=0.9, boxstyle="round,pad=0.25"))
        y -= 1.05
    else:
        y -= 0.55
ax2.text(0.4, y - 0.3, "Why this matters: the random walk is bounded by 0.10/frame\n"
        "so maneuvers stay smooth; safety check rejects the one Δ\n"
        "that would put the ego car within d_min of o_j.",
        fontsize=9, color=WIRE, va="top")

plt.tight_layout(pad=0.7)
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/055-WM-drivedreamer4d/fig-ntgm.png",
            dpi=150, bbox_inches="tight")
print("saved fig-ntgm.png")
