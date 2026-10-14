"""COME ControlNet injection: where the control features enter the frozen world model.

Bottom path (green): frozen DiT world model. Every early block feeds a skip to its
mirrored late block. Top path (purple): ControlNet = trainable copy of the first half
of the blocks, fed by scene-condition latents z_i from the forecasting UNet; each block
emits a zero-initialized linear projection c^n_i, gated by the invisibility mask M_i,
and residually added into the corresponding skip before the late block.
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "serif"
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OP = "#0b7a5c"     # frozen world model
ALT = "#6a55c2"    # ControlNet / new parts
IN = "#1a6fa3"     # inputs
WIRE = "#6e737b"
NORM = "#7a6b00"

fig, ax = plt.subplots(figsize=(13, 6.2))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 13); ax.set_ylim(0, 6.2)
ax.axis("off")


def box(x, y, w, h, text, color, fs=12, lines=1):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                         facecolor=color + "22", edgecolor=color, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color, fontsize=fs,
            linespacing=1.4)


def arrow(x1, y1, x2, y2, color=WIRE, style="-", lw=1.3):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=style, mutation_scale=11))


# ---- bottom path: frozen world model ----
box(0.4, 0.6, 2.2, 1.0, "noisy latent\n$x_t$", IN)
box(3.2, 0.6, 2.2, 1.0, "blocks 1..N/2\n(frozen)", OP)
box(7.0, 0.6, 2.2, 1.0, "blocks N/2+1..N\n(frozen)", OP)
box(10.8, 0.6, 1.8, 1.0, r"$\hat{x}_{t+1:t+\tau}$", IN)
arrow(2.6, 1.1, 3.2, 1.1)
arrow(5.4, 1.1, 7.0, 1.1)
arrow(9.2, 1.1, 10.8, 1.1)
ax.text(4.3, 0.15, "skip connections (early -> late, concat + MLP)", ha="center",
        color=WIRE, fontsize=11)

# skip arrows up and over into late blocks
for x0, x1 in [(3.6, 7.6), (4.4, 8.4), (5.0, 9.0)]:
    ax.annotate("", xy=(x1, 1.6), xytext=(x0, 1.6),
                arrowprops=dict(arrowstyle="-", color=WIRE, lw=1.1, linestyle=(0, (3, 2))))
ax.plot([3.6, 3.6], [1.6, 2.35], color=WIRE, lw=1.1, ls=(0, (3, 2)))
ax.plot([9.0, 9.0], [1.6, 2.35], color=WIRE, lw=1.1, ls=(0, (3, 2)))
ax.plot([3.6, 9.0], [2.35, 2.35], color=WIRE, lw=1.1, ls=(0, (3, 2)))

# ---- top path: ControlNet ----
box(0.4, 3.6, 2.2, 1.0, "scene condition\n$z_i$", ALT)
box(3.2, 3.6, 2.2, 1.0, "trainable copy of\nblocks 1..N/2", ALT)
box(7.0, 3.6, 2.2, 1.0, "zero convs\n$c^n_i$", ALT)
box(10.0, 3.6, 1.6, 1.0, r"$\odot M_i$", NORM)
arrow(2.6, 4.1, 3.2, 4.1, color=ALT)
arrow(5.4, 4.1, 7.0, 4.1, color=ALT)
arrow(9.2, 4.1, 10.0, 4.1, color=ALT)
ax.text(1.5, 3.15, "from forecasting UNet\n(via Occ-VAE encoder)", ha="center",
        color=WIRE, fontsize=10.5)

# control injection: from zero convs down into the skips (residual add)
for x in [7.6, 8.4, 9.0]:
    ax.annotate("", xy=(x, 2.45), xytext=(x, 3.6),
                arrowprops=dict(arrowstyle="-|>", color=ALT, lw=1.6, mutation_scale=11))
ax.text(9.9, 2.9, "residual add\ninto skip", color=ALT, fontsize=10.5, ha="left")

ax.text(0.4, 5.15, "stage 3: only ControlNet is trained, world model frozen",
        color=ALT, fontsize=12)
ax.text(0.4, 5.55, "COME = forecasting-guided generation", fontsize=15, weight="bold")
ax.text(0.4, 0.0, "", fontsize=1)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/045-WM-come/fig-controlnet.png",
            dpi=150, facecolor="white")
print("saved fig-controlnet.png")
