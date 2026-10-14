"""Self-drawn figure for Drive-OccWorld xray:
Auto-regressive continuous forecasting-and-planning loop (Eq. 3-4),
with a worked cost example taken from the paper's Figure 1.
Bright theme."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

IN = "#1a6fa3"; OUT = "#9e5c00"; OP = "#0b7a5c"; NORM = "#7a6b00"
LOSS = "#b34a06"; ALT = "#6a55c2"; WIRE = "#6e737b"
IN_FILL = "#e2eff5"; OUT_FILL = "#f8ecd2"; OP_FILL = "#dcf2e8"
ALT_FILL = "#ebe7f8"; LOSS_FILL = "#f9e3d6"

fig, ax = plt.subplots(figsize=(13, 6.8))
ax.set_xlim(0, 13); ax.set_ylim(0, 8); ax.axis("off")

def box(x, y, w, h, text, fc, ec, fs_=9.5):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                         fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs_, linespacing=1.6)

def arrow(x1, y1, x2, y2, c=WIRE, rad=0.0, w=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=w,
                                connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=4))

ax.text(6.5, 7.6, "Continuous rollout: world model W forecasts, planner P picks, pick becomes next action",
        ha="center", fontsize=12, weight="bold")

# step t
box(0.4, 5.2, 2.6, 1.4, "history\n{o_-h..o_0}, {s_1..s_t}\nactions {a_-h..a_t}", IN_FILL, IN)
box(3.7, 5.2, 2.8, 1.4, "world model W\n(Eq. 3)\nforecasts s_{t+1}", OP_FILL, OP)
arrow(3.0, 5.9, 3.7, 5.9, IN)
box(7.2, 5.2, 2.8, 1.4, "future occupancy S_{t+1}\n+ flow F_{t+1}\n(200x200x16)", ALT_FILL, ALT)
arrow(6.5, 5.9, 7.2, 5.9, OP)

# planner
box(7.2, 3.0, 2.8, 1.4, "planner P (Eq. 4)\nsample tau* in R^{N_tau x 2}\nargmin occupancy cost", LOSS_FILL, LOSS)
arrow(8.6, 5.2, 8.6, 4.4, ALT)

box(10.7, 3.0, 1.9, 1.4, "tau_{t+1}\noptimal\ntrajectory", OUT_FILL, OUT)
arrow(10.0, 3.7, 10.7, 3.7, LOSS)

# feedback loop: tau becomes action condition
arrow(11.65, 4.4, 11.65, 6.9, OUT, rad=0.0, w=2.2)
arrow(11.65, 6.9, 5.1, 6.9, OUT, rad=0.0, w=2.2)
arrow(5.1, 6.9, 5.1, 6.6, OUT, w=2.2)
ax.text(11.9, 5.9, "tau_{t+1} fed back\nas action a_{t+1}\nfor next step", fontsize=9, color=OUT, weight="bold",
        bbox=dict(fc="white", ec=OUT, lw=1.0, boxstyle="round,pad=0.25"))
ax.text(8.3, 7.15, "no GT ego leakage: train AND test use predicted trajectories", fontsize=9,
        color=OUT, style="italic")

# cost breakdown
ax.text(0.4, 2.7, "Occupancy-based cost f_o (three factors, summed):", fontsize=10.5, weight="bold", color=LOSS)
costs = [
    ("Agent-safety", "penalize candidates overlapping\nagent-occupied grids (+ near-miss margin)", LOSS),
    ("Road-safety", "penalize candidates outside\ndrivable area (from occupancy)", LOSS),
    ("Learned-volume", "learnable head on F_bev_{+t}\n-> 2D cost map (ST-P3 style)", LOSS),
]
for i, (t, d, c) in enumerate(costs):
    x = 0.4 + i * 2.15
    box(x, 1.35, 2.0, 1.0, f"{t}\n{d}", LOSS_FILL, c, 8.5)

# worked example from Figure 1
ax.text(7.0, 2.7, "Worked example (values from paper Fig. 1, t+1):", fontsize=10.5, weight="bold")
rows = [
    ("candidate (-1.2m, +1.1m)", "cost 0.9", "SELECTED", OP),
    ("candidate (-0.3m, +3.5m)", "cost 1.3", "safe but worse", WIRE),
    ("candidate (+1.6m, +1.8m)", "collision", "REJECTED", LOSS),
]
y = 2.15
for cand, cost, verdict, c in rows:
    ax.text(7.0, y, cand, fontsize=9.5, color=c)
    ax.text(10.0, y, cost, fontsize=9.5, color=c, weight="bold")
    ax.text(11.2, y, verdict, fontsize=9.5, color=c, weight="bold",
            bbox=dict(fc="white", ec=c, lw=0.9, boxstyle="round,pad=0.2") if c != WIRE else None)
    y -= 0.5
ax.text(7.0, 0.5, "t+2 repeats: candidates (-0.8m,+1.0m) off-road rejected; (+0.1m,+3.6m) cost 0.6 selected;\n"
        "(+1.1m,+1.4m) collision rejected. Each step re-forecasts from the picked trajectory.",
        fontsize=9, color=WIRE, va="top")

plt.tight_layout(pad=0.6)
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/054-WM-drive-occworld/fig-rollout.png",
            dpi=150, bbox_inches="tight")
print("saved fig-rollout.png")
