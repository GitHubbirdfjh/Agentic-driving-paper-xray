# -*- coding: utf-8 -*-
"""SEM2: RSSM + semantic filter, the two decode paths (clean layout)."""
import sys; sys.path.insert(0, "/home/hatch/workspace/your_files/agentic-driving-idea-report/xray")
from figlib import *

fig, ax = new_fig(13, 7.2)
title(ax, "SEM2: semantic filter splits one latent state into two decode paths")

# main chain
box(ax, 3, 40, 14, 16, "camera +\nlidar", "in", sub="o_t")
arrow(ax, 17, 48, 21, 48)
box(ax, 21, 40, 16, 16, "representation\nmodel (CNN)", "op", sub="q(z_t|h_t,o_t)")
arrow(ax, 37, 48, 41, 48)
box(ax, 41, 40, 13, 16, "RSSM\nlatent", "norm", sub="h_t , z_t")

# branch up: observation reconstruction
arrow(ax, 47.5, 56, 47.5, 68, "keep all info")
box(ax, 39, 68, 17, 14, "observation\npredictor", "alt", sub="p(o_t|h_t,z_t)")
arrow(ax, 56, 75, 60, 75)
box(ax, 60, 68, 17, 14, "image\nreconstruction", "out", sub="o^_t")

# branch down: semantic filter
arrow(ax, 47.5, 40, 47.5, 28, "filter")
box(ax, 39, 14, 17, 14, "semantic filter S", "op", sub="s^m_t = {h^m,z^m}")
arrow(ax, 56, 21, 60, 21)
box(ax, 60, 14, 17, 14, "mask predictor", "alt", sub="p(m_t|s^m_t)")
arrow(ax, 77, 21, 81, 21)
box(ax, 81, 14, 16, 14, "BEV semantic\nmask", "out", sub="m^_t", fontsize=8)
arrow(ax, 47.5, 14, 30, 14)
box(ax, 13, 14, 17, 14, "reward\npredictor", "loss", sub="p(r_t|s^m_t)")
arrow(ax, 47.5, 14, 47.5, 5)
box(ax, 39, -4, 17, 14, "actor / critic", "loss", sub="pi(a|s^m), v(s^m)")

ax.text(48, 86, "path A: reconstruct pixels (auxiliary, keeps the encoder honest)", ha="left", fontsize=8, color="#6a55c2")
ax.text(48, 10.5, "path B: driving-relevant only -> mask, reward, policy", ha="left", fontsize=8, color="#0b7a5c")

caption(fig, "Fig. SEM2-1: the original latent (h,z) reconstructs the image; the semantic filter output s^m reconstructs "
        "the BEV semantic mask and feeds the reward predictor and the actor/critic. Only path B touches driving decisions.")
fig.savefig("/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/138-WM-sem2/fig-sem2-arch.png",
            dpi=150, bbox_inches="tight")
print("saved")
