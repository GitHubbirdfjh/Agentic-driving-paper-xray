"""DrivingGPT: interleaved driving-language token sequence.

One figure, one idea: z_t and q_t live in ONE token stream and are trained
by ONE cross-entropy, yet reading out image-token positions is the world
model and reading out action-token positions is the planner. The causal
mask decides what each position may see: image tokens see past actions
(action-conditioned prediction); action tokens also see the current image
(observation-grounded planning).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(2, 1, figsize=(13, 6.2), gridspec_kw={"height_ratios": [1.25, 1]})
ax = axes[0]
ax.set_xlim(0, 100)
ax.set_ylim(0, 10)
ax.axis("off")

IMG, ACT = "#1a6fa3", "#9e5c00"

# ---- top panel: the interleaved sequence (2 frames shown, "..." between) ----
def draw_frame(x0, w, label, ax):
    n_img, n_act = 12, 3
    gap = 0.35
    iw = (w - n_act * 2.2 - gap * (n_img + n_act + 1)) / n_img
    x = x0 + gap
    for i in range(n_img):
        ax.add_patch(mpatches.Rectangle((x, 4.4), iw, 3.2, fc=IMG, ec="white", lw=0.8))
        x += iw + gap
    x += gap * 2
    acts = []
    for i in range(n_act):
        r = mpatches.Rectangle((x, 4.4), 2.2, 3.2, fc=ACT, ec="white", lw=0.8)
        ax.add_patch(r)
        acts.append((x + 1.1, 6.0))
        x += 2.2 + gap
    ax.text(x0 + w / 2, 3.3, label, ha="center", va="top", fontsize=11, style="italic")
    return acts

draw_frame(1, 40, "z1 (576 image tokens)      q1 (3 action tokens)", ax)
ax.text(43.5, 6.0, "...", fontsize=18, va="center", ha="center", color="#6e737b")
draw_frame(48, 40, "z2 (576 image tokens)      q2 (3 action tokens)", ax)
ax.text(90.5, 6.0, "...", fontsize=18, va="center", ha="center", color="#6e737b")

ax.annotate("world modeling\np(z_t | z<t, q<t)", xy=(21, 8.6), xytext=(8, 10.0),
            fontsize=10, ha="center", color=IMG, weight="bold",
            arrowprops=dict(arrowstyle="->", color=IMG, lw=1.6))
ax.annotate("planning\np(q_t | z<=t, q<t)", xy=(36, 8.6), xytext=(50, 10.0),
            fontsize=10, ha="center", color=ACT, weight="bold",
            arrowprops=dict(arrowstyle="->", color=ACT, lw=1.6))
ax.text(50, 1.6, "one token stream, one cross-entropy  -log p(x_t | x<t); the two tasks differ only in which positions you read out",
        ha="center", fontsize=10, color="#3d3d3a")

# ---- bottom panel: unified vocabulary map ----
ax = axes[1]
ax.set_xlim(0, 100)
ax.set_ylim(0, 10)
ax.axis("off")
ax.text(2, 8.6, "unified vocabulary: 16384 + 3x128 = 16768", fontsize=12, weight="bold")

segs = [("image tokens (VQ-VAE codebook)", 0, 78, IMG),
        ("dx", 78, 84, ACT), ("dy", 84, 90, "#c97a00"), ("dtheta", 90, 96, "#7a4a00")]
for name, a, b, c in segs:
    ax.add_patch(mpatches.Rectangle((a, 4.4), b - a, 2.6, fc=c, ec="white", lw=1))
    ax.text((a + b) / 2, 5.7, name, ha="center", va="center", fontsize=9, color="white", weight="bold")
ax.text(39, 3.4, "token id 0 ... 16383", ha="center", fontsize=9)
ax.text(81, 3.4, "16384", ha="center", fontsize=9)
ax.text(87, 3.4, "16512", ha="center", fontsize=9)
ax.text(93, 3.4, "16640", ha="center", fontsize=9)
ax.text(97.5, 3.4, "16767", ha="center", fontsize=9)
ax.text(50, 1.2, "guided sampling at inference: logits of the foreign vocabulary slice are masked out",
        ha="center", fontsize=10, color="#3d3d3a")

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/041-WM-drivinggpt/fig-seq.png", dpi=150)
print("saved fig-seq.png")
