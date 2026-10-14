"""TTC geometry: a concrete 2-car example computed by hand."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 9

IN, OUT, WIRE = "#1a6fa3", "#9e5c00", "#6e737b"

fig, ax = plt.subplots(figsize=(11, 3.6))
ax.set_xlim(-2, 52)
ax.set_ylim(-3.2, 3.2)
ax.axis("off")

# road
ax.add_patch(Rectangle((-2, -1.6), 54, 3.2, fc="#f5f4ed", ec=WIRE, lw=1))
ax.plot([-2, 52], [0, 0], ls="--", color=WIRE, lw=1)

# ego car at x=0, v=25
ax.add_patch(Rectangle((-2, -0.7), 4, 1.4, fc="#e2eff5", ec=IN, lw=1.6))
ax.text(0, -1.15, "ego", ha="center", fontsize=9, weight="bold", color=IN)
ax.add_patch(FancyArrowPatch((2.4, 0), (9, 0), arrowstyle="-|>", mutation_scale=16,
                             lw=2, color=IN))
ax.text(5.7, 0.35, "v0 = 25 m/s", fontsize=8.5, color=IN)
ax.text(0, 1.15, "p0 = (0, 0)", ha="center", fontsize=8.5, color=IN)

# lead car at x=10, v=15
ax.add_patch(Rectangle((8, -0.7), 4, 1.4, fc="#f8ecd2", ec=OUT, lw=1.6))
ax.text(10, -1.15, "lead", ha="center", fontsize=9, weight="bold", color=OUT)
ax.add_patch(FancyArrowPatch((12.4, 0), (17, 0), arrowstyle="-|>", mutation_scale=16,
                             lw=2, color=OUT))
ax.text(14.7, 0.35, "v1 = 15 m/s", fontsize=8.5, color=OUT)
ax.text(10, 1.15, "p1 = (10, 0)", ha="center", fontsize=8.5, color=OUT)

# gap bracket
ax.annotate("", xy=(0, -2.2), xytext=(10, -2.2),
            arrowprops=dict(arrowstyle="<->", color=WIRE, lw=1.2))
ax.text(5, -2.55, "gap 10 m, closing at 10 m/s", ha="center", fontsize=8.5, color=WIRE)

ax.text(30, 1.9,
        r"$\tau = -\frac{(p_0-p_1)\cdot(v_0-v_1)}{\|v_0-v_1\|^2}$",
        fontsize=13, color="#141413")
ax.text(30, 0.9, "= -((-10, 0) . (10, 0)) / 100", fontsize=10, color="#3d3d3a")
ax.text(30, 0.2, "= 100 / 100 = 1.0 s", fontsize=10, color="#3d3d3a")
ax.text(30, -0.7, "-> TTC score = 100 - 1/1.0 = 99", fontsize=10, weight="bold", color=OUT)
ax.text(30, -1.6, "constant-velocity projection: how long until the gap closes",
        fontsize=8.5, color=WIRE, style="italic")

ax.set_title("Time-to-collision is just the gap divided by the closing speed, in vector form",
             fontsize=11, weight="bold", pad=12)
plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/078-Agent-lampilot/fig-ttc-geometry.png",
            dpi=150, bbox_inches="tight")
print("saved fig-ttc-geometry.png")
