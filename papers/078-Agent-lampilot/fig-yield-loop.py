"""The handoff loop: LLM policy sets strategic params; classical controllers act every step."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 8.5

IN, OUT, OP, ALT, LOSS, WIRE = "#1a6fa3", "#9e5c00", "#0b7a5c", "#6a55c2", "#b34a06", "#6e737b"

fig, ax = plt.subplots(figsize=(12, 5.2))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5.2)
ax.axis("off")

def box(x, y, w, h, label, sub, fc, ec, fs=9):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1.4)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2 + 0.14, label, ha="center", va="center",
            fontsize=fs, weight="bold", color=ec)
    ax.text(x + w / 2, y + h / 2 - 0.3, sub, ha="center", va="center",
            fontsize=7.5, color=WIRE)
    return b

def arrow(x1, y1, x2, y2, label=None, ls="-", c=WIRE):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                        lw=1.3, color=c, linestyle=ls)
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.13, label, ha="center",
                fontsize=7.5, color=c, style="italic")

# LLM one-shot at top
box(4.4, 3.9, 3.2, 1.0, "LLM (once per task)", "seconds of latency, CoT + code", "#dcf2e8", OP, fs=9)
ax.text(6.0, 4.85, "two clocks: LLM thinks in seconds, the car acts in milliseconds",
        fontsize=11, weight="bold", color="#141413")

# policy generator
box(4.4, 2.55, 3.2, 1.05, "policy generator", "yield autopilot()  between\nstrategic decisions", "#ebe7f8", ALT, fs=9)
arrow(6.0, 3.9, 6.0, 3.6, "exec code")

# classical control at bottom
box(0.4, 0.5, 3.2, 1.55, "steering control", "_steering_control(\ntarget_lane)", "#f8ecd2", OUT, fs=8.5)
box(4.4, 0.5, 3.2, 1.55, "IDM car-following", "_intelligent_driver_model(\ntarget_speed)", "#f8ecd2", OUT, fs=8.5)
box(8.4, 0.5, 3.2, 1.55, "safety in APIs", "is_safe_enter(lane)\nchecks before lane change", "#f9e3d6", LOSS, fs=8.5)
ax.text(6.0, 0.18, "every sim step: autopilot() -> [acceleration, steering]  (10-20 Hz)",
        ha="center", fontsize=8.5, color=WIRE, style="italic")

# arrows down: strategic params
arrow(5.2, 2.55, 2.0, 2.05, "set_target_lane")
arrow(6.8, 2.55, 6.0, 2.05, "set_target_speed")

# exception fallback
arrow(7.6, 2.8, 9.6, 1.3, "exception / StopIteration", ls="--", c=LOSS)
ax.text(9.9, 2.15, "fall back to\nautopilot()", fontsize=7.5, color=LOSS, ha="center")

# example code snippet
ax.text(0.4, 3.35,
        'def overtake_using_right_lane():\n'
        '    ...\n'
        '    set_target_lane(right_lane)   # strategy\n'
        '    set_target_speed(1.5*v_lead)  # strategy\n'
        '    yield autopilot()             # <- classical control runs here',
        fontsize=7.5, family="monospace", color="#3d3d3a",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f4ed", ec=WIRE))

plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/078-Agent-lampilot/fig-yield-loop.png",
            dpi=150, bbox_inches="tight")
print("saved fig-yield-loop.png")
