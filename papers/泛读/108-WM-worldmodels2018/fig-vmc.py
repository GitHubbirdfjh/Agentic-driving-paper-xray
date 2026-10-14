# V-M-C agent architecture of "World Models" (Ha & Schmidhuber, 2018).
# Left: real-environment loop. Right: dreaming loop (policy trained inside M).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
titles = ["Real environment loop", "Dream loop: train C inside M"]

for ax, title in zip(axes, titles):
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=10)

def box(ax, xy, w, h, text, sub="", fc="#e8f0f7", ec="#1a6fa3"):
    p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.06", fc=fc, ec=ec, lw=1.6)
    ax.add_patch(p)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(cx, cy + (0.25 if sub else 0), text, ha="center", va="center",
            fontsize=11.5, weight="bold", color="#111")
    if sub:
        ax.text(cx, cy - 0.5, sub, ha="center", va="center", fontsize=8.6, color="#333")

def arr(ax, a, b, label="", ls="-", c="#555"):
    p = FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.6,
                        linestyle=ls, color=c, shrinkA=2, shrinkB=4)
    ax.add_patch(p)
    if label:
        ax.text((a[0]+b[0])/2, (a[1]+b[1])/2 + 0.22, label, ha="center",
                fontsize=9, color="#333", style="italic")

# ---- Left panel: V, M, C ----
ax = axes[0]
box(ax, (0.4, 4.6), 2.6, 1.6, "V  (VAE)", "obs o  ->  z (32-dim)", fc="#e2eff5", ec="#1a6fa3")
box(ax, (4.0, 4.6), 2.6, 1.6, "M  (MDN-RNN)", "P(z'|z,a,h),  LSTM 256", fc="#dcf2e8", ec="#0b7a5c")
box(ax, (7.4, 4.6), 2.2, 1.6, "C  (linear)", "a = W[z;h]+b\n867 params", fc="#f8ecd2", ec="#9e5c00")
box(ax, (3.0, 1.0), 4.4, 1.5, "Environment", "CarRacing-v0 / VizDoom", fc="#f3f1ea", ec="#6e737b")
arr(ax, (3.0, 5.6), (4.0, 5.6), "z")
arr(ax, (6.6, 5.6), (7.4, 5.6), "[z; h]")
arr(ax, (8.5, 4.6), (8.5, 2.5), "a", c="#9e5c00")
arr(ax, (5.2, 2.5), (5.2, 4.6), "o, r", c="#1a6fa3")
ax.text(5.2, 0.55, "train: VAE 1 epoch + MDN-RNN 20 epochs (random-policy data) -> CMA-ES on C",
        ha="center", fontsize=8.8, color="#333")

# ---- Right panel: dream ----
ax = axes[1]
box(ax, (0.4, 4.6), 2.6, 1.6, "M  (MDN-RNN)", "sample z' ~ mix-of-Gaussians\ntemperature tau", fc="#dcf2e8", ec="#0b7a5c")
box(ax, (4.0, 4.6), 2.6, 1.6, "V decoder", "z' -> fake frame", fc="#e2eff5", ec="#1a6fa3")
box(ax, (7.4, 4.6), 2.2, 1.6, "C  (linear)", "trained by\nCMA-ES here", fc="#f8ecd2", ec="#9e5c00")
box(ax, (3.0, 1.0), 4.4, 1.5, "Virtual env (Gym wrapper over M)", "same interface as real env", fc="#f9e3d6", ec="#b34a06")
arr(ax, (3.0, 5.6), (4.0, 5.6), "z'")
arr(ax, (6.6, 5.6), (7.4, 5.6), "[z'; h]")
arr(ax, (8.5, 4.6), (8.5, 2.5), "a", c="#9e5c00")
arr(ax, (5.2, 2.5), (5.2, 4.6), "z', d (done)", c="#b34a06")
ax.text(5.2, 0.55, "take the C trained in the dream -> deploy into the real env (transfer)",
        ha="center", fontsize=8.8, color="#333")

fig.suptitle("World Models (Ha & Schmidhuber, 2018): the V-M-C split and the dream loop",
             fontsize=14, weight="bold", y=1.02)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/108-wm-worldmodels2018/fig-vmc.png", dpi=150, bbox_inches="tight")
print("ok")
