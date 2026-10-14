"""Self-drawn: GaussianWorld decomposes scene evolution into three factors
(Sec.3.2). Static scene: global affine transform by ego motion. Dynamic
objects: local position updates only for dynamic Gaussians. New areas:
random Gaussian completion.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax in axes:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

np.random.seed(7)

def scatter(ax, pts, c, s=90, ec="black", alpha=0.85, lw=1.0):
    for (x, y) in pts:
        ax.add_patch(Ellipse((x, y), 1.15, 0.95, fc=c, ec=ec, lw=lw, alpha=alpha))

def arr(ax, x1, y1, x2, y2, c="#0b7a5c"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=15,
                                 lw=2.0, color=c))

static = np.array([[2.2, 7.0], [3.4, 7.6], [2.8, 6.0], [4.2, 6.6], [1.6, 5.4], [3.0, 5.0]])
dynamic = np.array([[6.6, 6.4], [7.4, 7.0]])

# Panel 1: ego motion alignment
ax = axes[0]
ax.text(5, 9.3, "(1) Ego motion alignment of static scenes", ha="center", fontsize=10.5, weight="bold")
scatter(ax, static, "#1a6fa3"); scatter(ax, dynamic, "#9e5c00")
ax.text(2.6, 4.0, "ego", fontsize=9, ha="center"); ax.plot(2.6, 4.5, "k^", ms=9)
arr(ax, 2.6, 5.6, 2.6, 7.4, c="#333333")
ax.text(1.15, 6.4, "ego moves\nforward", fontsize=8.5, ha="center", color="#333333")
# shifted copies
scatter(ax, static + [0.9, -0.35], "#1a6fa3", alpha=0.45, ec="#1a6fa3")
scatter(ax, dynamic + [0.9, -0.35], "#9e5c00", alpha=0.45, ec="#9e5c00")
ax.text(5, 1.6, "$g^T_A = \\mathrm{Align}(g^{T-1}, M_{ego})$ (Eq.7)\n"
                "ALL Gaussians: global affine transform", ha="center", fontsize=8.8,
        bbox=dict(fc="#e2eff5", ec="#1a6fa3", boxstyle="round,pad=0.35"))
ax.text(5, 0.45, "Table 3: w/o ego align, mIoU -3.0", ha="center", fontsize=8.5, style="italic")

# Panel 2: dynamic movement
ax = axes[1]
ax.text(5, 9.3, "(2) Local movements of dynamic objects", ha="center", fontsize=10.5, weight="bold")
scatter(ax, static + [0.9, -0.35], "#1a6fa3")
scatter(ax, dynamic + [0.9, -0.35], "#9e5c00")
arr(ax, 7.5, 6.05, 8.5, 5.75, c="#9e5c00")
arr(ax, 8.3, 6.65, 8.9, 7.05, c="#9e5c00")
ax.text(5, 1.6, "$g^T_M = \\mathrm{Move}(g^T_A, x_T)$ (Eq.8)\n"
                "ONLY dynamic Gaussians move: soft-weighted by\n"
                "dynamic-category semantic probability", ha="center", fontsize=8.8,
        bbox=dict(fc="#f8ecd2", ec="#9e5c00", boxstyle="round,pad=0.35"))
ax.text(5, 0.45, "in code: Ref with soft semantic weights (suppl. A)", ha="center", fontsize=8.5, style="italic")

# Panel 3: completion
ax = axes[2]
ax.text(5, 9.3, "(3) Completion of newly-observed areas", ha="center", fontsize=10.5, weight="bold")
ax.plot([0.4, 0.4], [2.6, 8.6], ls="--", c="#333333", lw=1.4)
ax.text(0.45, 8.9, "old boundary", fontsize=8, color="#333333")
scatter(ax, static + [0.9, -0.35], "#1a6fa3")
rng = np.random.default_rng(3)
new_pts = np.c_[rng.uniform(0.6, 2.6, 8), rng.uniform(3.2, 8.0, 8)]
for (x, y) in new_pts:
    ax.plot(x, y, marker="x", color="#b34a06", ms=10, mew=1.8)
ax.text(1.6, 2.0, "random init\n$g^T_I$", fontsize=8.5, ha="center", color="#b34a06")
ax.text(5, 1.6, "$g^T_C = \\mathrm{Per}(g^T_I, x_T)$ (Eq.9)\n"
                "random Gaussians -> all attributes perceived\n"
                "from current RGB", ha="center", fontsize=8.8,
        bbox=dict(fc="#f9e3d6", ec="#b34a06", boxstyle="round,pad=0.35"))
ax.text(5, 0.45, "w/o completion: Gaussians drift out of range -> collapse", ha="center", fontsize=8.5, style="italic")

plt.suptitle("GaussianWorld: scene evolution = ego shift + dynamic motion + new-area completion",
             fontsize=12.5, weight="bold", y=1.02)
plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/043-WM-gaussianworld/fig-three-factors.png", dpi=150)
print("saved fig-three-factors.png")
