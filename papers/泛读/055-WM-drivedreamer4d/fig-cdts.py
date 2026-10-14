"""Self-drawn figure for DriveDreamer4D xray:
CDTS (Cousin Data Training Strategy): temporal-aligned real/novel pair
stacked as one batch to train a single 4DGS model with three losses.
Bright theme, role colors per paper-xray spec."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

IN = "#1a6fa3"; OUT = "#9e5c00"; OP = "#0b7a5c"; NORM = "#7a6b00"
LOSS = "#b34a06"; ALT = "#6a55c2"; WIRE = "#6e737b"
IN_FILL = "#e2eff5"; OUT_FILL = "#f8ecd2"; OP_FILL = "#dcf2e8"
ALT_FILL = "#ebe7f8"; LOSS_FILL = "#f9e3d6"

fig, ax = plt.subplots(figsize=(13, 7.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 8); ax.axis("off")
fs = 10

def box(x, y, w, h, text, fc, ec, fs_=fs):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                         fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs_, linespacing=1.5)

def arrow(x1, y1, x2, y2, c=WIRE, w=1.8, ls="-", rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=w, ls=ls,
                                connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=4))

# ---------- top-left: original trajectory video ----------
ax.text(0.4, 7.5, "Original trajectory (real Waymo video)", fontsize=11, weight="bold", color=IN)
frames_o, frames_n = [], []
for t in range(5):
    x = 0.4 + t * 1.06
    box(x, 6.0, 0.95, 0.62, f"t={t*10}", IN_FILL, IN, 9)
    frames_o.append((x + 0.475, 6.0))
    box(x, 4.35, 0.95, 0.62, f"t={t*10}", ALT_FILL, ALT, 9)
    frames_n.append((x + 0.475, 4.97))
ax.text(0.4, 5.75, "I_ori,t : GT image (40 frames)", fontsize=fs, color=IN)
ax.text(0.4, 5.08, "Novel trajectory (world-model video)", fontsize=11, weight="bold", color=ALT)
ax.text(0.4, 4.12, "I_novel,t : generated image (same 40 steps)", fontsize=fs, color=ALT)

# vertical temporal-alignment braces between pairs
for (xo, yo), (xn, yn) in zip(frames_o, frames_n):
    ax.plot([xo, xn], [yo, yn], color=WIRE, lw=1.0, ls=":")
ax.text(6.35, 5.5, "temporal-aligned\ncousin pair", fontsize=9, color=WIRE, ha="left", va="center",
        style="italic")
ax.annotate("", xy=(6.05, 5.15), xytext=(6.05, 5.85),
            arrowprops=dict(arrowstyle="<|-|>", color=WIRE, lw=1.2))

# ---------- BatchStack ----------
box(6.9, 5.05, 2.0, 1.6, "BatchStack\n({I_ori,t}, {I_novel,t})", OP_FILL, OP)
arrow(5.6, 6.31, 6.9, 6.1, IN)
arrow(5.6, 4.66, 6.9, 5.5, ALT)

# ---------- 4DGS model ----------
box(9.6, 5.05, 2.7, 1.6, "one shared 4DGS model\nφ′ = φ + F(φ, t)", OUT_FILL, OUT)
arrow(8.9, 5.85, 9.6, 5.85, OP)

# ---------- renders ----------
box(6.9, 2.5, 2.0, 0.9, "rendered Î_ori,t", IN_FILL, IN)
box(9.6, 2.5, 2.0, 0.9, "rendered Î_novel,t", ALT_FILL, ALT)
arrow(10.5, 5.05, 10.5, 3.4, OUT)
arrow(10.5, 5.05, 7.9, 3.4, OUT, rad=0.15)

# ---------- losses ----------
box(2.2, 0.45, 2.6, 1.15, "L_ori = λ1‖Î−I‖1\n + λ2‖D̂−D‖1 + λ3 SSIM", LOSS_FILL, LOSS, 9)
box(5.2, 0.45, 3.0, 1.15, "L_novel = λ1‖Î−I‖1 + λ3 SSIM\n(no depth: LiDAR incomplete)", LOSS_FILL, LOSS, 9)
box(8.6, 0.45, 3.6, 1.15, "L_reg = ‖Fp(Î_ori) − Fp(Î_novel)‖1\nperceptual coherence", LOSS_FILL, LOSS, 9)
ax.text(6.05, 1.95, "L = L_ori + λ_novel L_novel + λ_reg L_reg     (λ_novel=1, λ_reg=1e−3 from ablation)",
        ha="center", fontsize=10.5, weight="bold", color=LOSS,
        bbox=dict(fc="white", ec=LOSS, lw=1.2, boxstyle="round,pad=0.3"))

arrow(7.9, 2.5, 4.8, 1.6, LOSS, rad=-0.1)
arrow(7.9, 2.5, 3.5, 1.6, LOSS, rad=0.15)
arrow(10.6, 2.5, 6.7, 1.6, LOSS, rad=0.1)
arrow(9.0, 2.5, 10.4, 1.6, LOSS, rad=-0.12)

ax.text(0.4, 0.3, "Key: real frames supervise RGB+depth+SSIM; synthetic frames supervise RGB+SSIM only;\n"
        "the regularizer pulls rendered pairs at the same timestep together in perception-feature space.",
        fontsize=9, color=WIRE, va="top")
plt.tight_layout(pad=0.6)
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/055-WM-drivedreamer4d/fig-cdts.png",
            dpi=150, bbox_inches="tight")
print("saved fig-cdts.png")
