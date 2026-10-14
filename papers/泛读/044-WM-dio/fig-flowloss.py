"""DIO flow-consistency loss: the geometry of q' = q + f*dt and the BCE penalty.

A car moving at f = (4, 0, 0) m/s. Query point q sits on the car's front edge at t0.
If the predicted flow is right, the occupancy must reappear at q' = q + f*dt at t0+dt.
The loss punishes any disagreement between o(q) and o(q').
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "serif"
import matplotlib.pyplot as plt
import numpy as np

OUT = "#9e5c00"    # occupancy / predictions
OP = "#0b7a5c"     # flow vector
LOSS = "#b34a06"   # loss
WIRE = "#6e737b"
IN = "#1a6fa3"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("white")

# ---- left: geometry of the consistency constraint ----
ax = ax1
ax.set_facecolor("white")
# car at t0: x in [0,4]; at t0+dt: x in [2.4, 6.4]
car0 = plt.Rectangle((0, 0.6), 4, 1.4, facecolor="#dceefb", edgecolor=IN, lw=1.5)
car1 = plt.Rectangle((2.4, -1.2), 4, 1.4, facecolor="#f8ecd2", edgecolor=OUT, lw=1.5,
                     linestyle="--")
ax.add_patch(car0); ax.add_patch(car1)
ax.text(2, 1.35, "car at $t_0$", ha="center", color=IN, fontsize=12)
ax.text(4.4, -0.45, "car at $t_0{+}\\Delta t$", ha="center", color=OUT, fontsize=12)

# query point on the front edge at t0
q = np.array([4.0, 1.3])
qp = np.array([4.0 + 2.4, 1.3])
ax.plot([q[0]], [q[1]], "o", ms=11, color=OUT, zorder=5)
ax.text(q[0] - 0.15, q[1] + 0.35, "$q$", fontsize=14, color=OUT)
ax.plot([qp[0]], [qp[1] - 2.5], "o", ms=11, color=OUT, zorder=5)
ax.text(qp[0] + 0.1, qp[1] - 2.5 + 0.35, "$q'$", fontsize=14, color=OUT)
# flow arrow
ax.annotate("", xy=(qp[0] - 0.25, q[1]), xytext=(q[0] + 0.25, q[1]),
            arrowprops=dict(arrowstyle="->", color=OP, lw=2.5))
ax.text(5.0, 1.75, r"$f\Delta t = 2.4\,\mathrm{m}$", color=OP, fontsize=13, ha="center")
# vertical time arrow
ax.annotate("", xy=(7.6, -0.6), xytext=(7.6, 1.6),
            arrowprops=dict(arrowstyle="->", color=WIRE, lw=1.3))
ax.text(8.0, 0.5, r"$\Delta t=0.6$s", rotation=90, va="center", color=WIRE, fontsize=11)
ax.set_xlim(-0.5, 9); ax.set_ylim(-1.6, 2.6)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_color("#c9c6bb")
ax.set_title("Instance permanence: $o(q)$ and $o(q')$ must agree", fontsize=13)

# ---- right: the BCE penalty curve on a toy example ----
ax = ax2
ax.set_facecolor("white")
o_q = 0.9          # occupancy predicted at q (fixed)
o_qp = np.linspace(0.01, 0.99, 200)
bce = -(o_q * np.log(o_qp) + (1 - o_q) * np.log(1 - o_qp))
ax.plot(o_qp, bce, color=LOSS, lw=2.5)
ax.axvline(o_q, color=WIRE, ls="--", lw=1.2)
ax.text(0.9, bce.max() * 0.92, r"$o(q')=o(q)$", ha="right", color=WIRE, fontsize=12)
# mark two toy operating points
for val, tag in [(0.85, "consistent"), (0.10, "inconsistent")]:
    v = -(o_q * np.log(val) + (1 - o_q) * np.log(1 - val))
    ax.plot([val], [v], "o", ms=8, color=LOSS)
    ax.text(val, v + 0.25, f"{tag}\nL={v:.2f}", ha="center", fontsize=10.5, color=LOSS)
ax.set_xlabel(r"predicted $o(q', s)$", fontsize=13)
ax.set_ylabel(r"$-\left(o(q)\log o(q') + (1-o(q))\log(1-o(q'))\right)$", fontsize=12, color=LOSS)
ax.set_title(r"$L_{\mathrm{flow}}$ with $o(q,s)=0.9$: only agreement is cheap", fontsize=13)
ax.set_xlim(0, 1)
for sp in ax.spines.values():
    sp.set_color("#c9c6bb")

fig.suptitle("Flow consistency: move the query point with the predicted flow, then demand agreement",
             fontsize=14)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/044-WM-dio/fig-flowloss.png",
            dpi=150, facecolor="white")
print("saved fig-flowloss.png")
