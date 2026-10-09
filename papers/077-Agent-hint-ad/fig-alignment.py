"""The four online alignment tasks as concrete Q&A on one frame."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 8

IN, OP, ALT, OUT, WIRE = "#1a6fa3", "#0b7a5c", "#6a55c2", "#9e5c00", "#6e737b"

fig, ax = plt.subplots(figsize=(12.5, 5.2))
ax.set_xlim(0, 12.5)
ax.set_ylim(0, 5.2)
ax.axis("off")

ax.text(6.25, 4.85, "Alignment task: force the language decoder to read the AD model's own tokens",
        fontsize=11.5, weight="bold", ha="center")
ax.text(6.25, 4.5, "Q&A pairs are generated online during training; answers come from the AD model's predictions, not human labels",
        fontsize=8.5, color=WIRE, ha="center", style="italic")

cards = [
    ("<align_count>  Counting", IN, "#e2eff5",
     "Q: How many cars, trucks, buses,\nbicycles, pedestrians are detected\nby the model in this frame?",
     "A: 4 cars, 1 truck, 2 pedestrians."),
    ("<align_position>  Position", OP, "#dcf2e8",
     "Q: Tell the distance and clock\ndirection of the 2nd car.",
     "A: 12.50 m at 1 o'clock direction."),
    ("<align_motion>  Motion", ALT, "#ebe7f8",
     "Q: Tell the future velocity and\nmoving direction of the 2nd car.",
     "A: moving at 8.32 m/s to its\n3 o'clock direction."),
    ("<align_planning>  Planning", OUT, "#f8ecd2",
     "Q: Tell the future trajectory of\nthe ego car (0.5 s steps).",
     "A: (1.2, 0.1), (2.5, 0.2),\n(3.9, 0.2), (5.4, 0.1)."),
]

for k, (title, ec, fc, q, a) in enumerate(cards):
    x = 0.25 + k * 3.05
    b = FancyBboxPatch((x, 0.35), 2.85, 3.85, boxstyle="round,pad=0.06",
                       fc=fc, ec=ec, lw=1.4)
    ax.add_patch(b)
    ax.text(x + 1.425, 3.85, title, ha="center", fontsize=9, weight="bold", color=ec)
    ax.text(x + 1.425, 3.0, q, ha="center", va="top", fontsize=7.8, color="#3d3d3a")
    ax.text(x + 1.425, 1.55, a, ha="center", va="top", fontsize=7.8, color=ec,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=ec, alpha=0.85))

ax.text(6.25, 0.08, "effect: the decoder can no longer ignore the tokens — Tab.9 position disalignment 10.01 -> 1.22",
        fontsize=8.5, color=WIRE, ha="center", style="italic")

plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/077-Agent-hint-ad/fig-alignment.png",
            dpi=150, bbox_inches="tight")
print("saved fig-alignment.png")
