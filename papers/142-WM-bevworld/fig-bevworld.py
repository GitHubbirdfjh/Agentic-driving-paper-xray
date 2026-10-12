# -*- coding: utf-8 -*-
"""BEVWorld: multi-modal tokenizer + latent BEV sequence diffusion."""
import sys; sys.path.insert(0, "/home/hatch/workspace/your_files/agentic-driving-idea-report/xray")
from figlib import *

fig, ax = new_fig(13, 8)
title(ax, "BEVWorld: tokenizer compresses sensors; diffusion dreams the BEV future")

# tokenizer row (top)
box(ax, 2, 64, 12, 14, "multi-view\nimages", "in", fontsize=8)
box(ax, 2, 48, 12, 14, "lidar", "in", fontsize=8)
arrow(ax, 14, 71, 18, 71); arrow(ax, 14, 55, 18, 62)
box(ax, 18, 58, 15, 16, "BEV\nencoder", "op", sub="Swin + deformable\nfusion, C'=4")
arrow(ax, 33, 66, 37, 66)
box(ax, 37, 58, 14, 16, "BEV\nlatents", "norm", sub="x_t compact")
arrow(ax, 51, 66, 55, 66)
box(ax, 55, 58, 15, 16, "voxel lift\n+ ray render", "op", sub="NeRF-style\nN_r=150")
arrow(ax, 70, 66, 74, 66)
box(ax, 74, 58, 12, 16, "recon.\nimages+lidar", "out", sub="L1+perc+GAN\n+ depth L1", fontsize=8)
ax.text(46, 79, "multi-modal tokenizer: self-supervised, bidirectional", ha="center", fontsize=8.5,
        color="#0b7a5c", weight="bold")

# diffusion row (bottom)
arrow(ax, 44, 58, 44, 46)
box(ax, 30, 32, 16, 12, "past BEV\ntokens", "norm", sub="x_{t-P..t}")
box(ax, 48, 32, 16, 12, "noisy future\ntokens", "norm", sub="x^eps_{t+1..t+N}")
box(ax, 66, 32, 12, 12, "action\ntokens", "in", sub="ego motion", fontsize=8)
arrow(ax, 46, 38, 52, 38); arrow(ax, 64, 38, 72, 38)
box(ax, 80, 32, 16, 12, "spatio-temporal\ntransformer", "op", sub="causal temporal\nattn + AdaLN")
arrow(ax, 88, 32, 88, 24)
box(ax, 80, 12, 16, 12, "denoised\nfuture BEV", "out", sub="DDIM sampling\none shot")
arrow(ax, 80, 18, 62, 18)
ax.text(71, 14, "decode -> future\nimages + lidar", ha="center", fontsize=7.5, color="#9e5c00")

ax.text(52, 48, "L_diff = ||eps_hat - eps||_1", ha="center", fontsize=8, color="#b34a06",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f9e3d6", ec="none"))
ax.text(15, 38, "all N future frames\ndeniosed at once:\nno autoregressive\ndrift", ha="center", fontsize=7.5, color="#0b7a5c",
        bbox=dict(boxstyle="round,pad=0.4", fc="#dcf2e8", ec="#0b7a5c"))

caption(fig, "Fig. BW-1: the tokenizer learns a compact BEV latent via ray-casting reconstruction; "
        "a diffusion transformer predicts the whole future BEV sequence in one shot, conditioned on action tokens.")
fig.savefig("/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/142-WM-bevworld/fig-bevworld-arch.png",
            dpi=150, bbox_inches="tight")
print("saved")
