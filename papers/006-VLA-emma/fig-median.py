"""Schematic: Top-K sampling + medoid (median-trajectory) selection in EMMA.

24 sampled trajectories (light blue) scatter around the ground truth (dashed).
The medoid (orange) is the sample minimizing the average pairwise L2 distance.
Illustrative numbers only, not from the paper.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
T = 16  # 16 waypoints, e.g. 8s at 2Hz
t = np.linspace(0, 8, T)
# ground truth: gentle lane change / curve in BEV (x: forward, y: lateral)
gt = np.stack([t * 3.0, 0.6 * np.sin(t * 0.5)], axis=1)

K = 24
samples = []
for k in range(K):
    noise = rng.normal(0, 0.35, size=(T, 2))
    # growing noise with horizon, like autoregressive drift
    noise *= (0.4 + 0.6 * np.linspace(0, 1, T))[:, None]
    bias = rng.normal(0, 0.25, size=2)
    samples.append(gt + bias + noise)
samples = np.array(samples)  # (K, T, 2)

# pairwise L2 distance between full trajectories
d = np.sqrt(((samples[:, None] - samples[None, :]) ** 2).sum(-1)).sum(-1)  # (K, K)
avg_d = d.mean(axis=1)
medoid = int(np.argmin(avg_d))

fig, ax = plt.subplots(figsize=(7.2, 4.6))
for k in range(K):
    ax.plot(samples[k, :, 0], samples[k, :, 1], color="#7fb3d5", lw=1.1, alpha=0.55)
ax.plot(samples[medoid, :, 0], samples[medoid, :, 1], color="#d4661f", lw=2.6,
        label="medoid (min avg pairwise L2)")
ax.plot(gt[:, 0], gt[:, 1], color="#333333", lw=1.6, ls="--", label="ground truth")
ax.scatter([0], [0], s=60, color="#333333", zorder=5)
ax.text(0.3, -0.55, "ego start", fontsize=10, color="#333333")
ax.set_xlabel("x (m, forward)", fontsize=11)
ax.set_ylabel("y (m, lateral)", fontsize=11)
ax.set_title("Top-K sampling: pick the trajectory closest to all others", fontsize=12)
ax.legend(fontsize=10, loc="upper left")
ax.set_aspect("equal")
ax.grid(alpha=0.25)
fig.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-40/vla-driving/emma-xray/fig-median.png"
fig.savefig(out, dpi=150)
print("saved", out)
