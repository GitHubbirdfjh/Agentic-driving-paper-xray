"""Tensor shape flow of the OccSora 4D scene tokenizer (B=2 example)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "cm"

IN, OP, OUT = "#1a6fa3", "#0b7a5c", "#9e5c00"
IN_F, OP_F, OUT_F = "#e2eff5", "#dcf2e8", "#f8ecd2"
WIRE = "#6e737b"

fig, ax = plt.subplots(figsize=(13, 4.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 4.2); ax.axis("off")

stages = [
    ("$R_{in}$\ninput occupancy", "B x 16 x 200 x 200 x 32\n(class labels)", IN, IN_F),
    ("class embedding\n+ reshape", "B x 128 x 32 x 200 x 200\n(D folded into channels)", OP, OP_F),
    ("3x downsample\n(3D conv, stride 2)", "B x 512 x 4 x 25 x 25\n(T,H,W each /8)", OP, OP_F),
    ("codebook\nquantize (N=512*)", "$R_{mi}$: B x 128 x 4 x 25 x 25\n2500 tokens x 128-d", OUT, OUT_F),
    ("decoder\n(upsample)", "$R_o$: B x 16 x 200 x 200 x 32\nreconstructed", IN, IN_F),
]
notes = ["", "c'=8 per class\n16x8=128 (code)", "cross-channel\nattention here", "*N from code,\nnot in paper", ""]
xs = [0.4, 2.9, 5.4, 7.9, 10.4]
w = 2.0
for x, (title, shape, col, fill), note in zip(xs, stages, notes):
    box = FancyBboxPatch((x, 1.1), w, 1.7, boxstyle="round,pad=0.08",
                         facecolor=fill, edgecolor=col, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, 2.25, title, ha="center", va="center", fontsize=9.5, color=col, weight="bold")
    ax.text(x + w/2, 1.55, shape, ha="center", va="center", fontsize=8.5, color="#141413")
    if note:
        ax.text(x + w/2, 0.75, note, ha="center", va="center", fontsize=7.5, color=WIRE, style="italic")
for x0, x1 in zip(xs[:-1], xs[1:]):
    ax.add_patch(FancyArrowPatch((x0 + w, 1.95), (x1, 1.95), arrowstyle="-|>",
                                 mutation_scale=12, color=WIRE, linewidth=1.3))
ax.text(6.5, 3.6, "OccSora tokenizer: 1,280,000 positions  ->  2,500 positions  (ratio 512 = 8x8x8)",
        ha="center", fontsize=11, weight="bold", color="#141413")
ax.text(6.5, 3.25, "B=2 example; D=16 height slices folded into channels; codebook size N=512 only in released code",
        ha="center", fontsize=8.5, color=WIRE)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-40/world-model-driving/occsora-xray/fig-shapes.png", dpi=150)
print("saved")
