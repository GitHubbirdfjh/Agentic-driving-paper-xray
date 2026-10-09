"""DrivingGPT: action tokenization + unbinning + SE(2) integration, worked out.

Top: binning of one action component (illustrative percentiles, labeled as such):
    clamp to [1st, 99th percentile] -> M=128 uniform bins -> floor formula.
Bottom: three unbinned relative actions are lifted to SE(2) matrices and
chain-multiplied to an absolute trajectory (numbers computed here, quoted in text).
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"

# ---------- illustrative numbers (labeled illustrative in the doc) ----------
M = 128
a1, a99 = -0.5, 5.0          # 1st / 99th percentile of dx per 0.5s frame, NAVSIM@2Hz
true_dx = 2.35
clamped = min(max(true_dx, a1), a99)
q = int(np.floor((clamped - a1) / (a99 - a1) * (M - 1)))
unbinned = a1 + (q + 0.5) / M * (a99 - a1)
bin_w = (a99 - a1) / M
print(f"bin width = {bin_w:.4f} m")
print(f"true {true_dx} -> clamped {clamped} -> bin q = {q}")
print(f"unbin: bin {q} -> {unbinned:.4f} m (quantization error {true_dx - unbinned:.4f} m)")

# ---------- SE(2) chain integration ----------
actions = [(5.0, 0.0, 0.0), (4.8, 0.2, 0.05), (4.9, 0.1, -0.02)]  # (dx, dy, dtheta)
def T_of(dx, dy, th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, dx], [s, c, dy], [0, 0, 1]])
poses = [np.eye(3)]
for a in actions:
    poses.append(poses[-1] @ T_of(*a))
for i, P in enumerate(poses[1:], 1):
    print(f"absolute pose T(t+{i}->t): x={P[0,2]:.3f} y={P[1,2]:.3f} theta={np.arctan2(P[1,0],P[0,0]):.4f} rad")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

ax = axes[0]
xs = np.linspace(-1, 6, 400)
ax.plot(xs, np.clip(xs, a1, a99), color="#1a6fa3", lw=2, label="clamp to [1st, 99th pct]")
edges = np.linspace(a1, a99, M + 1)
for e in edges[::16]:
    ax.axvline(e, color="#b5d8ec", lw=0.8)
ax.scatter([true_dx], [clamped], s=90, color="#9e5c00", zorder=5)
ax.annotate(f"true dx={true_dx} -> bin q={q}\nunbin={unbinned:.3f} m",
            xy=(true_dx, clamped), xytext=(2.6, 4.6),
            arrowprops=dict(arrowstyle="->", color="#9e5c00"), fontsize=10,
            color="#9e5c00", weight="bold")
ax.set_xlabel("raw frame-to-frame dx (m)  [illustrative percentiles]")
ax.set_ylabel("clamped value (m)")
ax.set_title("component-wise binning: q = floor((a-a1st)/(a99th-a1st) * 127)", fontsize=11)
ax.legend(fontsize=9)

ax = axes[1]
traj = np.array([[P[0, 2], P[1, 2]] for P in poses])
ax.plot(traj[:, 0], traj[:, 1], "o-", color="#0b7a5c", ms=8, lw=2)
for i, (x, y) in enumerate(traj):
    ax.text(x + 0.15, y + 0.15, f"t+{i}", fontsize=10)
ax.set_aspect("equal")
ax.set_xlabel("x (m, ego frame at t)")
ax.set_ylabel("y (m, ego frame at t)")
ax.set_title("unbin -> SE(2) matrices -> chain multiply = absolute trajectory", fontsize=11)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/041-WM-drivinggpt/fig-bin.png", dpi=150)
print("saved fig-bin.png")
