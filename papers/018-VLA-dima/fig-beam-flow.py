# -*- coding: utf-8 -*-
"""DiMA Fig B: block diagram of the joint-training architecture.

Left branch: vision planner (scene encoder -> planning transformer).
Right branch: MLLM (BEAM tokens -> per-component Q-formers -> LLM).
Links: distillation (KL between penultimate-layer ego features),
surrogate decoder heads (masked recon / future pred / scene editing),
VQA + LLM planning heads. Inference: MLLM branch is discarded.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(12, 7.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis("off")

IN, OP, ALT, LOSS, OUT = "#e2eff5", "#dcf2e8", "#ebe7f8", "#f9e3d6", "#f8ecd2"

def box(x, y, w, h, text, color, fontsize=9.5, style="round,pad=0.35"):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle=style,
                 facecolor=color, edgecolor="black", linewidth=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

def arrow(x1, y1, x2, y2, color="#6e737b", ls="-", lw=1.4):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=ls))

# input
box(0.3, 8.6, 2.6, 0.9, "multi-view\nimage sequence", IN, 9.5)
# scene encoder
box(3.6, 8.6, 2.6, 0.9, "scene encoder\n(VAD / UniAD backbone)", OP, 9.5)
arrow(2.9, 9.05, 3.6, 9.05)
# BEAM tokens
box(6.9, 8.6, 4.6, 0.9, "BEAM tokens:  B  E  A  M\n(BEV, ego, agent, map embeddings)", ALT, 10)
arrow(6.2, 9.05, 6.9, 9.05)

ax.text(1.6, 7.9, "vision branch (used at inference)", fontsize=10, color="#0b7a5c", ha="center")
ax.text(9.2, 7.9, "MLLM branch (discarded at inference)", fontsize=10, color="#6a55c2", ha="center")

# vision branch
box(0.3, 6.0, 2.6, 1.0, "planning\ntransformer", OP, 9.5)
arrow(1.6, 8.6, 1.6, 7.0, color="#0b7a5c")
box(0.3, 4.3, 2.6, 0.9, "waypoint prediction\n(L_planning)", OUT, 9.5)
arrow(1.6, 6.0, 1.6, 5.2, color="#0b7a5c")

# MLLM branch
box(7.3, 6.4, 3.8, 0.8, "per-component Q-formers\n(Map/BEV/Ego/Agent)", ALT, 9.5)
arrow(9.2, 8.6, 9.2, 7.2)
box(7.3, 5.0, 3.8, 0.9, "LLM (LLaVA-v1.5-7B,\nLoRA in stage 2)", ALT, 9.5)
arrow(9.2, 6.4, 9.2, 5.9)

# decoder heads off the LLM penultimate layer
box(6.9, 3.0, 2.0, 0.9, "ego/agent\nprediction\nheads", OUT, 9)
box(9.4, 3.0, 2.0, 0.9, "VQA\nlanguage\nhead", OUT, 9)
arrow(8.0, 5.0, 7.9, 3.9)
arrow(10.4, 5.0, 10.4, 3.9)

box(4.0, 1.1, 5.9, 1.0, "surrogate heads (3-layer MLPs on penultimate LLM features)\n"
                        "masked BEV recon (L_recon) | future BEV pred (L_future) | scene editing", "#f8f4cf", 9)
arrow(8.2, 5.0, 6.2, 2.1)

# distillation link between branches
box(3.0, 6.3, 2.4, 1.0, "distillation\nL_distill = D_KL(P_llm || P_vis)", LOSS, 9)
arrow(5.4, 6.8, 7.3, 6.0, color="#b34a06", ls="--")
arrow(3.0, 6.8, 2.9, 6.5, color="#b34a06", ls="--")
ax.text(5.9, 6.35, "penultimate-layer\nego features", fontsize=8.5, color="#b34a06", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9))

ax.text(6, 0.25, "Stage 1 (60 epochs): vision branch alone.   Stage 2 (30 epochs): joint, losses L_planning + L_LLM + L_recon + L_future + L_distill",
        fontsize=9.5, ha="center", color="#3d3d3a",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#6e737b"))

ax.set_title("DiMA: one shared scene encoder, two training branches, one inference branch", fontsize=12, pad=10)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/018-VLA-dima/fig-beam-flow.png", dpi=150)
print("saved fig-beam-flow.png")
