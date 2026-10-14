# -*- coding: utf-8 -*-
"""RenderWorld: Img2Occ -> AM-VAE -> autoregressive world model."""
import sys; sys.path.insert(0, "/home/hatch/workspace/your_files/agentic-driving-idea-report/xray")
from figlib import *

fig, ax = new_fig(13, 7.5)
title(ax, "RenderWorld: 2D labels -> 3D occupancy -> 4D forecast + planning")

# stage 1: Img2Occ
box(ax, 2, 60, 14, 16, "multi-view\n2D labels", "in", sub="depth + semantic")
arrow(ax, 16, 68, 20, 68)
box(ax, 20, 60, 16, 16, "anchor\nGaussians", "op", sub="per-voxel center")
arrow(ax, 36, 68, 40, 68)
box(ax, 40, 60, 16, 16, "gaussian\nsplatting", "op", sub="render D, S (eq.4-5)")
arrow(ax, 56, 68, 60, 68)
box(ax, 60, 60, 16, 16, "3D occupancy\nlabels", "out", sub="L = Lsem + Ldep")
ax.text(39, 58, "Img2Occ (self-supervised, 2D->3D)", ha="center", fontsize=8.5, color="#0b7a5c", weight="bold")

# stage 2: AM-VAE
arrow(ax, 68, 60, 68, 52)
box(ax, 60, 36, 16, 14, "AM-VAE", "norm", sub="air / non-air\n2 codebooks")
ax.text(39, 34, "separate encoding: sparse objects keep their own codebook", ha="center", fontsize=8, color="#7a6b00")

# stage 3: world model
arrow(ax, 68, 36, 68, 28)
box(ax, 60, 18, 16, 14, "autoregressive\nworld model", "op", sub="masked temporal\nattention (eq.18)")
arrow(ax, 76, 18, 76, 10)
box(ax, 66, 1, 12, 8, "4D occ\nforecast", "out", fontsize=8)
box(ax, 80, 1, 12, 8, "ego\ntrajectory", "out", fontsize=8)
arrow(ax, 72, 5, 80, 5); arrow(ax, 86, 9, 82, 9)

ax.text(68, 82, "pure vision: no LiDAR, no 3D GT", ha="center", fontsize=8.5, color="#1a6fa3", weight="bold")
caption(fig, "Fig. RW-1: Gaussian splatting turns 2D labels into 3D occupancy; AM-VAE tokenizes air/non-air separately; "
        "a GPT-style world model autoregressively predicts future tokens and decodes occupancy + ego displacement.")
fig.savefig("/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/139-WM-renderworld/fig-renderworld-arch.png",
            dpi=150, bbox_inches="tight")
print("saved")
