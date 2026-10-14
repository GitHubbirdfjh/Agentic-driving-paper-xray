"""Self-drawn figure for OmniDrive xray: trajectory-centered scene representation.
Shows an ego car, one candidate (simulated) trajectory, the 10m/3s close-object
band, close objects highlighted, drivable-area boundary, and a second trajectory
that leaves the drivable area (ruled 'out of drivable area' by the checklist).
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "serif"
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)
ax.set_facecolor("#faf9f5")
fig.patch.set_facecolor("#faf9f5")

# --- drivable area boundary (grey dashed) ---
theta = np.linspace(-0.35, 1.15, 200)
bound_x = 34 * np.cos(theta) - 4
bound_y = 34 * np.sin(theta) - 8
ax.plot(bound_x, bound_y, color="#6e737b", lw=1.4, ls=(0, (6, 4)))
ax.text(27.5, 8.0, "drivable-area\nboundary", color="#6e737b", fontsize=11,
        ha="left", va="center")

# --- ego car at origin (blue input role) ---
ego = plt.Rectangle((-1.4, -0.7), 2.8, 1.4, fc="#e2eff5", ec="#1a6fa3", lw=1.8)
ax.add_patch(ego)
ax.text(0, 0, "ego", color="#1a6fa3", fontsize=13, ha="center", va="center", weight="bold")
ax.annotate("", xy=(0, 0), xytext=(-4.5, 0),
            arrowprops=dict(arrowstyle="-|>", color="#1a6fa3", lw=1.6))
ax.text(-5.6, 0.4, "heading", color="#1a6fa3", fontsize=11)

# --- expert trajectory: slow straight (green, safe) ---
tx = np.linspace(0, 14, 60)
ty = 0.15 * tx
ax.plot(tx, ty, color="#0b7a5c", lw=2.6)
for s in [4, 8, 12]:
    ax.plot(s, 0.15 * s, marker="o", color="#0b7a5c", ms=5)
ax.text(14.8, 0.6, "expert: slow, lane keeping\n(rule: safe)", color="#0b7a5c",
        fontsize=11, ha="left", va="center")

# --- simulated trajectory: moderate-speed left turn (orange -> out of area) ---
sx = np.linspace(0, 20, 80)
sy = 0.02 * sx + 0.045 * sx ** 1.6
ax.plot(sx, sy, color="#9e5c00", lw=2.6)
ax.plot(sx[-1], sy[-1], marker="x", color="#9e5c00", ms=10, mew=2.5)
ax.text(13.5, 17.5, "simulated: moderate speed, left turn\n(rule: out of drivable area)",
        color="#9e5c00", fontsize=11, ha="left", va="center")

# --- close-object band: 10m / 3s around expert trajectory (light orange fill) ---
band_x = np.linspace(0, 14, 60)
band_y = 0.15 * band_x
ax.fill_between(band_x, band_y - 10, band_y + 10, color="#f8ecd2", alpha=0.55, zorder=0)
ax.text(0.6, -13.5, "10 m / 3 s band\naround trajectory", color="#9e5c00", fontsize=11,
        ha="left", va="center")

# --- close objects (highlighted, orange role) ---
cone = Circle((8.2, 2.4), 0.9, fc="#f0cf8a", ec="#9e5c00", lw=2.2)
ax.add_patch(cone)
ax.text(8.2, 2.4, "cone", color="#9e5c00", fontsize=10, ha="center", va="center", weight="bold")
ax.text(9.8, 4.6, "(+8.2, +2.4)", color="#9e5c00", fontsize=10, ha="left", va="center")

ped = Circle((4.4, -7.2), 0.9, fc="#f0cf8a", ec="#9e5c00", lw=2.2)
ax.add_patch(ped)
ax.text(4.4, -7.2, "ped", color="#9e5c00", fontsize=10, ha="center", va="center", weight="bold")
ax.text(4.4, -9.3, "(+4.4, -7.2)", color="#9e5c00", fontsize=10, ha="center", va="center")

# --- far object (greyed out: not fed to GPT-4) ---
far = Circle((24, -14), 0.9, fc="#e8e8e8", ec="#9aa0a8", lw=1.2, ls=(0, (4, 3)))
ax.add_patch(far)
ax.text(24, -14, "car", color="#9aa0a8", fontsize=10, ha="center", va="center")
ax.text(22.2, -16.4, "outside band: dropped", color="#9aa0a8", fontsize=10)

ax.set_xlim(-8, 32)
ax.set_ylim(-20, 26)
ax.set_aspect("equal")
ax.set_xlabel("x (m, ego frame)", fontsize=12, color="#3d3d3a")
ax.set_ylabel("y (m, ego frame)", fontsize=12, color="#3d3d3a")
ax.tick_params(colors="#6e737b")
for spine in ax.spines.values():
    spine.set_color("#deddd8")
ax.set_title("Trajectory-centered scene representation: only what a candidate\n"
             "trajectory may hit is listed for the VLM; the rest is dropped",
             fontsize=13, color="#141413", pad=12, loc="left")
fig.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-40/vla-driving/omnidrive-xray/fig-counterfactual.png"
fig.savefig(out, bbox_inches="tight")
print("saved:", out)
