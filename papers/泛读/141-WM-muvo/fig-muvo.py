# -*- coding: utf-8 -*-
"""MUVO: transformer sensor fusion -> GRU transition -> 3 decoders."""
import sys; sys.path.insert(0, "/home/hatch/workspace/your_files/agentic-driving-idea-report/xray")
from figlib import *

fig, ax = new_fig(13, 7.5)
title(ax, "MUVO: camera + lidar fused by transformer, decoded into 3 spaces")

box(ax, 2, 60, 13, 14, "RGB\nimage", "in", sub="600x960")
box(ax, 2, 40, 13, 14, "lidar point\ncloud", "in", sub="range view")
arrow(ax, 15, 67, 20, 67); arrow(ax, 15, 47, 20, 47)
box(ax, 20, 60, 14, 14, "ResNet18", "op", fontsize=8)
box(ax, 20, 40, 14, 14, "ResNet18\n(range)", "op", fontsize=8)
arrow(ax, 34, 67, 38, 62); arrow(ax, 34, 47, 38, 52)
box(ax, 38, 48, 17, 18, "transformer\nfusion", "norm", sub="tokens + pos emb\n+ sensor emb")
arrow(ax, 55, 57, 60, 57)
box(ax, 60, 48, 15, 18, "GRU\ntransition", "op", sub="action-cond.\n2D latent (s,h)")
ax.text(49, 44, "RV-WOB-TR wins: no BEV bottleneck", ha="center", fontsize=8, color="#7a6b00")

# decoders fan out
arrow(ax, 75, 60, 80, 70)
arrow(ax, 75, 57, 80, 57)
arrow(ax, 75, 54, 80, 44)
box(ax, 80, 64, 16, 14, "image\ndecoder", "out", sub="L1")
box(ax, 80, 50, 16, 14, "lidar\ndecoder", "out", sub="L2 xyz + L1 r")
box(ax, 80, 36, 16, 14, "occupancy\ndecoder", "out", sub="SCAL loss\n192x192x64")

ax.text(88, 32, "occupancy is the\n'actionable' head:\ngeometry a planner\ncan read directly", ha="center", fontsize=7.5, color="#9e5c00",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f8ecd2", ec="#9e5c00"))

caption(fig, "Fig. MUVO-1: fused camera+lidar tokens drive an action-conditioned GRU world model; "
        "three decoders predict images, point clouds, and 3D occupancy. Self-supervised: no labels needed.")
fig.savefig("/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/141-WM-muvo/fig-muvo-arch.png",
            dpi=150, bbox_inches="tight")
print("saved")
