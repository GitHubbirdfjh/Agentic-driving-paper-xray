"""Figure for diffusiondrive-xray: (a) truncated noise schedule curve,
(b) schematic of anchored Gaussian mixture vs pure Gaussian start points."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/hatch/workspace/your_files/paper-xray-40/e2e-mainstream/diffusiondrive-xray/fig-truncated-diffusion.png"

T = 1000
beta = np.linspace(1e-4, 0.02, T)
alpha_bar = np.cumprod(1 - beta)
sigma = np.sqrt(1 - alpha_bar)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

# ---- Panel A: noise schedule truncation ----
ax = axes[0]
ax.plot(np.arange(1, T + 1), sigma, color="#1a6fa3", lw=2)
ax.axvline(50, color="#b34a06", ls="--", lw=1.5)
ax.scatter([50], [sigma[49]], s=55, color="#b34a06", zorder=5)
ax.annotate("truncation point t=50\nnoise std ~ %.2f" % sigma[49],
            xy=(50, sigma[49]), xytext=(300, 0.45),
            fontsize=10, color="#b34a06",
            arrowprops=dict(arrowstyle="->", color="#b34a06", lw=1.2))
ax.set_xlabel("diffusion timestep t", fontsize=11)
ax.set_ylabel("noise std  sqrt(1 - alpha_bar_t)", fontsize=11)
ax.set_xlim(0, 1000)
ax.set_ylim(0, 1.05)
ax.set_title("(a) training only adds noise up to t = 50 / 1000", fontsize=12)
ax.grid(alpha=0.25)

# ---- Panel B: start distributions schematic ----
rng = np.random.default_rng(7)
ax2 = axes[1]
anchors = np.array([[0.0, 0.0], [3.0, 1.4], [-1.8, 3.2]])
for a in anchors:
    pts = a + 0.17 * rng.standard_normal((400, 2))
    ax2.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.28, color="#0b7a5c")
    ax2.scatter([a[0]], [a[1]], s=90, marker="x", color="#b34a06",
                linewidths=2.5, zorder=5)
pts = 2.2 * rng.standard_normal((1200, 2)) + np.array([6.5, 1.5])
ax2.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.10, color="#1a6fa3")
ax2.scatter([], [], s=40, color="#0b7a5c", alpha=0.6, label="anchored Gaussian (ours)")
ax2.scatter([], [], s=40, color="#1a6fa3", alpha=0.35, label="pure Gaussian (vanilla)")
ax2.scatter([], [], s=90, marker="x", color="#b34a06", label="prior anchors")
ax2.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax2.set_xlim(-4, 11)
ax2.set_ylim(-3, 6)
ax2.set_aspect("equal", adjustable="box")
ax2.set_xlabel("action dim 1 (schematic)", fontsize=11)
ax2.set_ylabel("action dim 2 (schematic)", fontsize=11)
ax2.set_title("(b) where denoising starts: near anchors vs from nowhere", fontsize=12)

fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("saved:", OUT)
