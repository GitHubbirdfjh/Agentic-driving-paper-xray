"""Self-drawn figure for 135-WM-occworld: two-stage pipeline.
Stage 1: VQ-VAE tokenizer (3D occupancy -> BEV -> discrete scene tokens).
Stage 2: GPT-like spatial-temporal transformer predicting next tokens + ego token."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.2),
                               gridspec_kw={"height_ratios": [1, 1.25]})
fig.patch.set_facecolor("#faf9f5")
for ax in (ax1, ax2):
    ax.set_facecolor("#faf9f5"); ax.axis("off")

def box(ax, x, y, w, h, label, sub, fc, ec, fs=9.5):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                         fc=fc, ec=ec, lw=1.5))
    ax.text(x + w / 2, y + h / 2 + 0.18, label, ha="center", va="center",
            fontsize=fs, weight="bold", color="#141413")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.28, sub, ha="center", va="center",
                fontsize=7.5, color="#3d3d3a")

def arrow(ax, x1, y1, x2, y2, label=None, color="#6e737b", lw=1.6):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.22, label, ha="center",
                fontsize=8, color=color, style="italic")

# ---------- Stage 1 ----------
ax1.set_xlim(0, 13); ax1.set_ylim(0, 3.4)
ax1.text(6.5, 3.05, "Stage 1: scene tokenizer (VQ-VAE)  --  turn 3D occupancy into discrete tokens",
         ha="center", fontsize=11, weight="bold", color="#141413")
box(ax1, 0.2, 0.9, 2.0, 1.5, "3D occupancy", "H x W x D voxels\nsemantic labels", "#e2eff5", "#1a6fa3")
box(ax1, 2.9, 0.9, 2.0, 1.5, "BEV projection", "class embeddings\nfold height into\nchannels (2D!)", "#dcf2e8", "#0b7a5c")
box(ax1, 5.6, 0.9, 2.0, 1.5, "2D CNN encoder", "downsample x4\n50 x 50 feature map", "#dcf2e8", "#0b7a5c")
box(ax1, 8.3, 0.9, 2.0, 1.5, "Nearest code", "codebook: 512 codes\n128-dim each", "#ebe7f8", "#6a55c2")
box(ax1, 10.9, 0.9, 1.9, 1.5, "Scene tokens", "discrete 0..511\nper BEV cell", "#f8ecd2", "#9e5c00")
for a, b in [(2.2, 2.9), (4.9, 5.6), (7.6, 8.3), (10.3, 10.9)]:
    arrow(ax1, a, 1.65, b, 1.65)
ax1.text(6.5, 0.35, 'each token ~= a "word" like "there is a car here"  |  decoder reconstructs occupancy (softmax + Lovasz loss)',
         ha="center", fontsize=8.5, color="#3d3d3a", style="italic")

# ---------- Stage 2 ----------
ax2.set_xlim(0, 13); ax2.set_ylim(0, 4.4)
ax2.text(6.5, 4.05, "Stage 2: GPT-like spatial-temporal transformer  --  continue the token sequence into the future",
         ha="center", fontsize=11, weight="bold", color="#141413")
# past frames
for i, t in enumerate(["t-2", "t-1", "t"]):
    x = 0.4 + i * 1.7
    box(ax2, x, 1.7, 1.45, 1.3, f"tokens {t}", "scene + ego token", "#f8ecd2", "#9e5c00", fs=8.5)
arrow(ax2, 5.6, 2.35, 6.3, 2.35)
box(ax2, 6.3, 1.55, 2.5, 1.6, "Spatial agg.", "self-attn in frame\n2x2 merge, multi-scale\n(3 scales)", "#dcf2e8", "#0b7a5c", fs=9)
arrow(ax2, 8.8, 2.35, 9.5, 2.35)
box(ax2, 9.5, 1.55, 2.6, 1.6, "Temporal causal attn", "per position, across\ntime; future masked", "#ebe7f8", "#6a55c2", fs=9)
arrow(ax2, 12.1, 2.35, 12.6, 2.35)
box(ax2, 11.35, 0.15, 1.5, 1.1, "not drawn", "", "#faf9f5", "#faf9f5", fs=1)
# outputs below
arrow(ax2, 10.8, 1.55, 10.8, 0.9)
box(ax2, 9.7, 0.05, 2.2, 0.85, "pred tokens t+1", "scene -> decode occ.\nego -> decode traj.", "#f9e3d6", "#b34a06", fs=8.5)
ax2.text(6.5, 0.45, "autoregressive: feed predicted t+1 back to predict t+2 ...  |  w/o ego token: forecasting itself degrades (Table 4)",
         ha="center", fontsize=8.5, color="#3d3d3a", style="italic")

plt.tight_layout()
out = "/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/135-WM-occworld/fig-occworld.png"
plt.savefig(out, dpi=150, facecolor="#faf9f5")
print("saved", out)
