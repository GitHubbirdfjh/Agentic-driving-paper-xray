"""Epona architecture + tensor shapes diagram (matplotlib, light theme)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "DejaVu Sans"

IN, OUT, OP, ALT = "#1a6fa3", "#9e5c00", "#0b7a5c", "#6a55c2"
IN_F, OUT_F, OP_F, ALT_F = "#e2eff5", "#f8ecd2", "#dcf2e8", "#ebe7f8"
WIRE = "#6e737b"

fig, ax = plt.subplots(figsize=(13.2, 7.6))
ax.set_xlim(0, 13.2); ax.set_ylim(0, 7.6); ax.axis("off")

def box(x, y, w, h, label, sub, lc, fc, fs=11):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                       facecolor=fc, edgecolor=lc, linewidth=1.6)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2 + 0.10, label, ha="center", va="center",
            fontsize=fs, color=lc, weight="bold")
    ax.text(x + w/2, y + h/2 - 0.24, sub, ha="center", va="center",
            fontsize=8.5, color=WIRE)

def arrow(x1, y1, x2, y2, label=None, c=WIRE):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=13, linewidth=1.4, color=c, shrinkA=2, shrinkB=4)
    ax.add_patch(a)
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.16, label, ha="center", va="bottom",
                fontsize=8.5, color=c)

ax.text(6.6, 7.15, "Epona: one shared latent F feeds two diffusion heads", ha="center",
        fontsize=14, weight="bold", color="#141413")
ax.text(6.6, 6.82, "shapes for B=1, 10 cond. frames, 512x1024 images, DCAE 32x, 5 fps",
        ha="center", fontsize=9.5, color=WIRE)

# ---- inputs ----
box(0.25, 4.35, 2.3, 1.15, "10 past frames", r"$O_{1..10}$: $10\times512\times1024\times3$", IN, IN_F)
box(0.25, 2.85, 2.3, 1.15, "10 past actions", r"$a_{t-1\to t}\in\mathbb{R}^3$", IN, IN_F)
arrow(2.55, 5.15, 3.15, 5.15); arrow(2.55, 3.65, 3.15, 3.65)
box(3.15, 4.35, 2.0, 1.15, "DCAE encoder", r"$16\times32\times32$ / frame", OP, OP_F, fs=10)
box(3.15, 2.85, 2.0, 1.15, "discretize + Fourier", "3 tokens / frame", OP, OP_F, fs=10)
arrow(5.15, 5.15, 5.75, 5.15); arrow(5.15, 3.65, 5.75, 3.65)

# ---- MST ----
box(5.75, 3.0, 2.5, 2.3, "MST", "12x time-space blocks\ncausal time + spatial attn", ALT, ALT_F)
ax.text(7.0, 2.72, r"$E\in\mathbb{R}^{10\times515\times2048}$", ha="center", fontsize=9.5, color=ALT)
arrow(8.25, 4.4, 8.85, 4.4)
box(8.85, 3.85, 1.9, 1.1, "shared latent F", r"$B\times515\times2048$", ALT, ALT_F)
ax.text(9.8, 3.62, "last frame only", ha="center", fontsize=8.5, color=WIRE)

# ---- heads ----
arrow(10.75, 4.55, 11.35, 5.35)
arrow(10.75, 4.25, 11.35, 3.45)
box(11.35, 4.95, 1.65, 1.15, "TrajDiT", "1+1 blocks, 50M", OUT, OUT_F, fs=10)
box(11.35, 3.05, 1.65, 1.15, "VisDiT", "6+6 blocks, 1.2B", OUT, OUT_F, fs=10)
ax.text(12.18, 6.35, r"$15\times3$ trajectory", ha="center", fontsize=9, color=OUT)
arrow(12.18, 6.1, 12.18, 5.55, c=OUT)
ax.text(12.18, 2.85, r"$16\times32\times32$ next latent", ha="center", fontsize=9, color=OUT)
arrow(12.18, 3.05, 12.18, 3.5, c=OUT)

# action conditioning branch: TrajDiT first step -> VisDiT
ax.annotate("", xy=(11.55, 4.2), xytext=(12.18, 4.95),
            arrowprops=dict(arrowstyle="-|>", mutation_scale=11, linewidth=1.2,
                            color=OUT, linestyle=(0, (4, 3)), shrinkB=4))
ax.text(12.75, 4.55, r"$a_{T\to T+1}\to y\in\mathbb{R}^{6144}$", fontsize=8.5,
        color=OUT, va="center")

# decode
box(5.75, 0.55, 2.5, 1.15, "temporal DCAE decoder", r"$\to 512\times1024$ image", OP, OP_F, fs=10)
ax.annotate("", xy=(8.25, 1.1), xytext=(12.18, 2.55),
            arrowprops=dict(arrowstyle="-|>", mutation_scale=12, linewidth=1.3,
                            color=WIRE, shrinkA=2, shrinkB=4))

# chain-of-forward loop: decoder/MST output feeds back as history into MST
ax.annotate("", xy=(5.9, 2.95), xytext=(5.9, 1.75),
            arrowprops=dict(arrowstyle="-|>", mutation_scale=12, linewidth=1.4,
                            color="#b34a06", linestyle=(0, (5, 3)), shrinkA=2, shrinkB=6))
ax.text(3.15, 2.05, "chain-of-forward (training): 1-step estimate\nof next frame feeds back as history\n(every 10 steps, 3 passes)",
        fontsize=8.5, color="#b34a06", ha="center", va="center",
        bbox=dict(facecolor="#f9e3d6", edgecolor="#f1bd9c", boxstyle="round,pad=0.25"))

# legend
ax.text(0.25, 1.15, "blue = input   green = transform   purple = shared latent (MST/F)   orange = outputs",
        fontsize=9, color=WIRE)
ax.text(0.25, 0.75, "MST 1.3B (12 layers)   VisDiT 1.2B (12 blocks)   TrajDiT 50M (2 blocks)   total 2.5B",
        fontsize=9, color=WIRE)

plt.tight_layout(pad=0.6)
out = "/home/hatch/workspace/your_files/paper-xray-40/world-model-driving/epona-xray/fig-arch.png"
plt.savefig(out, dpi=170, facecolor="#faf9f5")
print("saved", out)
