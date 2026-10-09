"""LAW: action-aware latent construction + latent world model, with shapes.

One figure, one idea: the "world model" here is not a video generator but a
single MSE in latent space. Current visual latents (L vectors of dim D) and
the predicted waypoint vector (2M numbers) are concatenated per-vector and
pushed through an MLP to become action-aware latents; transformer blocks
predict the future latent; training matches it to the FUTURE frame's encoder
output, which is DETACHED (stop-gradient, present in code, absent from the paper).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(13.5, 5.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 10)
ax.axis("off")

def box(x, y, w, h, label, sub, fc, ec="#3d3d3a", fs=10):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25",
                                         fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2 + 0.35, label, ha="center", va="center", fontsize=fs, weight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.75, sub, ha="center", va="center", fontsize=8.5, color="#3d3d3a")

def arrow(a, b, txt=None, c="#6e737b", style="-"):
    p = FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, color=c, lw=1.6, linestyle=style)
    ax.add_patch(p)
    if txt:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.55, txt, ha="center", fontsize=8.5, color=c)

# visual latents
box(1, 4, 13, 3, "visual latents", "Vt: L x D\n(e.g. 6 x 256)", "#e2eff5")
# waypoints
box(1, 0.6, 13, 2.2, "predicted waypoints", "Wt: M x 2 -> ew: R^2M\n(e.g. 6 -> 12)", "#f8ecd2")
# concat
box(18.5, 2.4, 15, 3.4, "concat per vector", "[v_i ; ew]: D+2M\n(e.g. 256+12=268)", "#f8f4cf")
# mlp
box(38, 2.4, 13, 3.4, "MLP", "action-aware latents\nAt: L x D", "#dcf2e8")
# wm
box(55.5, 2.4, 15, 3.4, "transformer blocks", "predicted latents\nVhat(t+1): L x D", "#dcf2e8")
# target
box(75, 2.4, 13, 3.4, "encoder @ t+1", "visual latents\nV(t+1): L x D", "#e2eff5")
# loss
box(55.5, 7.6, 32.5, 1.9, "MSE loss  L_latent = mean ||vhat - v||^2", "target is .detach() : stop-gradient", "#f9e3d6", fs=9.5)

arrow((14, 5.5), (18.5, 4.6))
arrow((14, 1.7), (18.5, 3.6))
arrow((33.5, 4.1), (38, 4.1), "a_i = MLP([v_i, ew])")
arrow((51, 4.1), (55.5, 4.1))
arrow((70.5, 4.1), (75, 4.1))
arrow((62, 5.8), (62, 7.6), c="#b34a06")
arrow((81, 5.8), (81, 7.6), c="#b34a06")
ax.text(90.5, 8.5, "no gradients into the target\n(BYOL-style collapse guard)", fontsize=9, color="#b34a06", ha="left")

ax.text(50, 0.4, "the world model never generates pixels: its whole existence is one MSE term that drags the encoder toward predictable futures",
        ha="center", fontsize=10, color="#3d3d3a")
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/051-WM-law/fig-latent.png", dpi=150)
print("saved fig-latent.png")
