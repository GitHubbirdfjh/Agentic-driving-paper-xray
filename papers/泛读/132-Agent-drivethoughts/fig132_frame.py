# -*- coding: utf-8 -*-
"""Drive the Thoughts: ego-frame vs lane-relative evidence (Fig.5 of paper, redrawn)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# curved lane centerline (route reference)
s = np.linspace(0, 60, 200)
lane_x = 8 * np.sin(s / 25)          # gentle left curve
traj_x = lane_x + 1.5 + 0.3 * np.sin(s / 8)   # trajectory hugging lane, small offset
traj_y = s

for ax, title in [(ax1, "(a) Raw ego-frame reference"), (ax2, "(b) Lane-relative reference")]:
    ax.plot(lane_x, traj_y, color="#1a6fa3", linewidth=2.5, label="lane center (route)")
    ax.plot(traj_x, traj_y, color="#9e5c00", linewidth=2.5, label="planned trajectory")
    ax.set_aspect("equal"); ax.set_xlim(-12, 12); ax.set_ylim(0, 60)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.legend(fontsize=8, loc="upper left")

# (a) ego-frame: straight ray fixed to initial heading -> apparent lateral departure
ax1.plot([traj_x[0], traj_x[0]], [0, 60], color="#b34a06", linestyle="--", linewidth=1.5,
         label="ego initial-heading ray")
ax1.annotate("", xy=(traj_x[0], 45), xytext=(traj_x[-1], 45),
             arrowprops=dict(arrowstyle="<->", color="#b34a06", linewidth=1.5))
ax1.text((traj_x[0]+traj_x[-1])/2, 47, "apparent\net = +35.2 m", color="#b34a06",
         fontsize=10, weight="bold", ha="center",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#b34a06"))
ax1.text(0.5, -0.18, "Curved road + straight reference ray:\nordinary curve-following looks like a lane change.",
         transform=ax1.transAxes, fontsize=8.5, ha="center", color="#333333")

# (b) lane-relative: offset from lane center stays small
for yy in [15, 30, 45]:
    ix = np.argmin(np.abs(traj_y - yy))
    ax2.annotate("", xy=(lane_x[ix], yy), xytext=(traj_x[ix], yy),
                 arrowprops=dict(arrowstyle="<->", color="#0b7a5c", linewidth=1.5))
ax2.text(6.5, 32, "lane-relative\noffset within\n+3.7 m", color="#0b7a5c", fontsize=10, weight="bold", ha="left",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#0b7a5c"))
ax2.text(0.5, -0.18, "Same trajectory vs routed lane center:\ncorrectly read as lane-keeping.",
         transform=ax2.transAxes, fontsize=8.5, ha="center", color="#333333")

fig.suptitle('Frame-of-reference effect (paper Fig.5, redrawn): CoT = "keep lane, lane clear ahead"',
             fontsize=12, weight="bold")
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/132-Agent-drivethoughts/fig-frame.png", dpi=150)
print("saved2")
