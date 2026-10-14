# -*- coding: utf-8 -*-
"""Raw2Drive: dual-stream MBRL with Guidance Mechanism."""
import sys; sys.path.insert(0, "/home/hatch/workspace/your_files/agentic-driving-idea-report/xray")
from figlib import *

fig, ax = new_fig(13, 8)
title(ax, "Raw2Drive: privileged stream teaches the raw-sensor stream")

# privileged stream (top)
box(ax, 2, 68, 12, 14, "privileged\ninput o_t", "in", sub="BEV masks")
arrow(ax, 14, 75, 18, 75)
box(ax, 18, 68, 14, 14, "privileged\nworld model", "op", sub="DreamerV3-like")
arrow(ax, 32, 75, 36, 75)
box(ax, 36, 68, 14, 14, "privileged\npolicy", "loss", sub="actor-critic\nin imagination")
ax.text(26, 85, "stage I: easy problem first (compact, structured input)", ha="center", fontsize=8.5,
        color="#0b7a5c", weight="bold")

# raw stream (bottom)
box(ax, 2, 40, 12, 14, "raw sensor\ninput o^_t", "in", sub="multi-view video")
arrow(ax, 14, 47, 18, 47)
box(ax, 18, 40, 14, 14, "raw world\nmodel", "alt", sub="BEVFormer enc;\nRSSM shared init")
arrow(ax, 32, 47, 36, 47)
box(ax, 36, 40, 14, 14, "raw sensor\npolicy", "loss", sub="fine-tuned\nin imagination")
ax.text(26, 57, "stage II: hard problem, supervised by the teacher", ha="center", fontsize=8.5,
        color="#6a55c2", weight="bold")

# guidance mechanism (right)
box(ax, 58, 58, 20, 24, "Rollout Guidance", "norm", fontsize=10,
    sub="MSE(e,e^)  spatial-temporal\nMSE(h,h^)  deterministic\nKL(s,s^)  stochastic")
box(ax, 82, 58, 16, 24, "Head Guidance", "norm", fontsize=10,
    sub="reward r_t and\ncontinue c_t\nfrom privileged\nheads")
arrow(ax, 50, 75, 58, 70, "align")
arrow(ax, 50, 47, 58, 63, "align")

ax.text(89, 47, "single sampling:\nsample s^ once,\nfeed into BOTH\nstreams", ha="center", fontsize=7.5, color="#b34a06",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f9e3d6", ec="#b34a06"))
ax.text(89, 33, "-> kills cumulative\nsampling noise", ha="center", fontsize=7.5, color="#b34a06")

arrow(ax, 25, 68, 25, 54, "teaches", style="--", color="#0b7a5c")

caption(fig, "Fig. R2D-1: the privileged stream solves CARLA v2 first; the raw stream aligns its encoder/latent states "
        "to it (rollout guidance) and borrows its reward/continue heads (head guidance). Raw stream keeps only a decoder head.")
fig.savefig("/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/140-WM-raw2drive/fig-raw2drive-arch.png",
            dpi=150, bbox_inches="tight")
print("saved")
