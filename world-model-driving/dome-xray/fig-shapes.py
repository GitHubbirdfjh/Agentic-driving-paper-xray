# -*- coding: utf-8 -*-
"""Tensor shape flow through DOME: occupancy -> Occ-VAE -> diffusion -> occupancy."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(15, 5.2))
ax.set_xlim(0, 15)
ax.set_ylim(0, 5.2)
ax.axis("off")

IN, OP, OUT, WIRE = "#1a6fa3", "#0b7a5c", "#9e5c00", "#6e737b"
INFILL, OPFILL, OUTFILL = "#e2eff5", "#dcf2e8", "#f8ecd2"

def box(x, y, w, h, text, fc, ec, fs=8.5, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                facecolor=fc, edgecolor=ec, linewidth=1.4))
    ax.text(x + w / 2, y + h / 2 + (0.12 if sub else 0), text, ha="center",
            va="center", fontsize=fs, weight="bold", color=ec)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.32, sub, ha="center", va="center",
                fontsize=7.2, color="#3d3d3a", family="monospace")

def arrow(x1, y1, x2, y2, label=None, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 color=WIRE, linewidth=1.3, linestyle=ls,
                                 mutation_scale=12, shrinkA=2, shrinkB=4))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label, ha="center",
                va="bottom", fontsize=7.2, color=WIRE, style="italic")

# ---- Occ-VAE encoder row (top) ----
ax.text(0.15, 4.75, "Occ-VAE  encoder (frozen during diffusion training)", fontsize=10,
        weight="bold", color=OP)
box(0.15, 3.55, 2.5, 0.95, "occupancy frame", INFILL, IN, sub="200x200x16  int labels")
arrow(2.65, 4.02, 3.05, 4.02, "class embed E_cls")
box(3.05, 3.55, 2.5, 0.95, "BEV-style tensor", INFILL, IN, sub="200x200x128  (16x8)")
arrow(5.55, 4.02, 5.95, 4.02, "2D ResNet")
box(5.95, 3.55, 2.6, 0.95, "mu / sigma", OPFILL, OP, sub="25x25x64 each")
arrow(8.55, 4.02, 8.95, 4.02, "reparam.")
box(8.95, 3.55, 2.5, 0.95, "latent z", OPFILL, OP, sub="25x25x64   x0.18215")
arrow(11.45, 4.02, 11.85, 4.02)

# ---- diffusion row (middle) ----
ax.text(0.15, 3.05, "Spatial-temporal DiT  (depth 28 = 14 spatial + 14 temporal blocks)", fontsize=10,
        weight="bold", color=OP)
box(0.15, 1.85, 2.5, 0.95, "noised latents", OUTFILL, OUT, sub="b x 11 x 64 x 25 x 25")
arrow(2.65, 2.32, 3.05, 2.32, "patchify p=1")
box(3.05, 1.85, 2.9, 0.95, "tokens", OUTFILL, OUT, sub="b x 11 x 625 x 768")
arrow(5.95, 2.32, 6.35, 2.32)
box(6.35, 1.85, 2.9, 0.95, "spatial block", OPFILL, OP, sub="(b*11) x 625 x 768")
arrow(9.25, 2.32, 9.65, 2.32, "rearrange")
box(9.65, 1.85, 2.9, 0.95, "temporal block", OPFILL, OP, sub="(b*625) x 11 x 768")
arrow(12.55, 2.32, 12.95, 2.32)
box(12.95, 1.85, 1.9, 0.95, "pred noise", OUTFILL, OUT, sub="b x 11 x 128 x 25 x 25")

# ---- conditioning (bottom) ----
ax.text(0.15, 1.35, "conditioning (added to adaLN, not to tokens)", fontsize=10,
        weight="bold", color=IN)
box(0.15, 0.15, 2.5, 0.95, "context mask M", INFILL, IN, sub="replace 0..4 frames w/ clean z")
box(3.05, 0.15, 3.3, 0.95, "trajectory -> Fourier -> MLP", INFILL, IN,
    sub="11x2 rel (x,y) -> pad 77 -> 1x768 vector")
box(6.75, 0.15, 2.6, 0.95, "timestep embed", INFILL, IN, sub="diffusion step t -> 768")
arrow(3.9, 1.1, 3.9, 1.85, ls=(0, (3, 2)))
arrow(8.05, 1.1, 8.05, 1.85, ls=(0, (3, 2)))
ax.text(4.35, 1.55, "c = t_emb + pose_emb  ->  every block's adaLN", fontsize=8, color=WIRE, style="italic")

# ---- decoder row (right top) ----
box(12.35, 3.55, 2.5, 0.95, "3D decoder", OUTFILL, OUT, sub="latent -> logits")
arrow(11.45, 4.02, 12.35, 4.02)
ax.text(12.35, 3.25, "logits: b x 11 x 200 x 200 x 16 x 18", fontsize=7.5, color=WIRE, family="monospace")
ax.text(12.35, 2.95, "argmax over 18 classes -> predicted occupancy", fontsize=7.5, color=WIRE)

ax.text(0.15, 0.02, "", fontsize=1)
plt.tight_layout(pad=0.6)
plt.savefig("fig-shapes.png", dpi=170, bbox_inches="tight")
print("saved fig-shapes.png")
