# -*- coding: utf-8 -*-
"""079-WKER Fig1: instruction-guided interactor pipeline with token counts."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13, 6.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis("off")

C_IN, C_OP, C_OUT, C_WIRE = "#1a6fa3", "#0b7a5c", "#9e5c00", "#6e737b"

def box(x, y, w, h, text, sub="", fc="#e2eff5", ec=C_IN, fs=10):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1.6)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2 + (0.18 if sub else 0), text, ha="center", va="center",
            fontsize=fs, fontweight="bold", color="#141413")
    if sub:
        ax.text(x + w/2, y + h/2 - 0.38, sub, ha="center", va="center",
                fontsize=8.5, color="#3d3d3a")

def arrow(x1, y1, x2, y2, label="", color=C_WIRE, style="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                        lw=1.5, color=color, linestyle=style)
    ax.add_patch(a)
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.22, label, ha="center", fontsize=8.5, color=color)

box(0.3, 3.9, 2.2, 1.5, "Multi-view\nvideo features",
    "F_mv: N tokens per view\nEVA-02-L encoder", fc="#e2eff5")
box(0.3, 2.0, 2.2, 1.5, "BEV features",
    "F_bev: N_bev tokens\nretrained SparseBEV", fc="#e2eff5")
box(0.3, 0.3, 2.2, 1.5, "Instruction tokens",
    "L_inst: N_inst tokens\nLLaMA3-8B embeddings", fc="#f8ecd2", ec=C_OUT)

box(3.4, 2.9, 2.6, 1.8, "Cosine-similarity\ntop-k selection",
    "S_ij = <f_i, l_j>, max over j\nk=90 (image) / 300 (BEV)", fc="#dcf2e8", ec=C_OP)
arrow(2.5, 4.9, 3.4, 4.2); arrow(2.5, 2.75, 3.4, 3.4)
arrow(2.5, 1.05, 3.4, 3.2, label="instruction embedding", style="--", color=C_OUT)

box(6.9, 2.9, 2.4, 1.8, "Selected tokens",
    "image: 6x90 = 540\nBEV: 300 -> LLM input", fc="#ebe7f8", ec="#6a55c2")
arrow(6.0, 3.8, 6.9, 3.8, label="top-k")

box(3.4, 0.4, 2.6, 1.6, "Cross-Attention\n(2 layers)",
    "Q = selected tokens\nK,V = ALL original features", fc="#dcf2e8", ec=C_OP)
arrow(2.5, 4.4, 3.4, 1.2, style=":", color=C_WIRE)
arrow(6.0, 1.2, 6.9, 3.0, label="enriched tokens")
arrow(9.3, 3.8, 10.2, 3.8)

box(10.2, 2.9, 2.5, 1.8, "LLaMA3-8B (LoRA)",
    "input length: ~10k+\nreduced to ~1k tokens", fc="#f8ecd2", ec=C_OUT)
ax.text(11.45, 2.35, "output: QA / risk assess / trajectory",
        ha="center", fontsize=8.5, color="#3d3d3a")

ax.text(0.3, 5.85, "WKER's instruction-guided interactor: coarse select by instruction, then refill lost detail via attention",
        fontsize=12, fontweight="bold", color="#141413")
ax.text(3.4, 2.35, "all original features also feed K,V", fontsize=8, color=C_WIRE, ha="center",
        bbox=dict(fc="white", ec=C_WIRE, lw=0.6, boxstyle="round,pad=0.2"))

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/079-Agent-wker/fig-interactor.png",
            dpi=160, bbox_inches="tight", facecolor="white")
print("saved fig-interactor.png")
