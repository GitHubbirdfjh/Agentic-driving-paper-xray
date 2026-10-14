# -*- coding: utf-8 -*-
"""fig-cascaded-rl.py: the three-stage progressive training strategy.
Stage 1 DM-SFT: cold-start both modes.
Stage 2 FCM-RL: MP-GRPO forces G/2 text + G/2 tool responses per query, pools
all G for reward normalization -> intra- + inter-mode contrast.
Stage 3 AMS-RL: native GRPO, agent picks its own mode token, contrastive tool
reward teaches WHEN to use tools.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

IN, OUT, OP, ALT, LOSS, WIRE = "#1a6fa3", "#9e5c00", "#0b7a5c", "#6a55c2", "#b34a06", "#6e737b"
IN_F, OUT_F, OP_F, ALT_F, LOSS_F = "#e2eff5", "#f8ecd2", "#dcf2e8", "#ebe7f8", "#f9e3d6"

fig, ax = plt.subplots(figsize=(14, 7.6))
ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")

def box(x, y, w, h, face, edge, txt, fs=9, lw=1.6, ls="-"):
    r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                facecolor=face, edgecolor=edge, linewidth=lw, linestyle=ls)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs, zorder=5)

def arrow(x1, y1, x2, y2, c=WIRE, w=1.7):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=c, lw=w))

ax.text(7, 7.6, "DriveAgent-R1: foundation building → mode strengthening → intelligent selection",
        fontsize=12.5, weight="bold", ha="center")

# ---------- Stage 1 ----------
ax.text(2.1, 6.9, "Stage 1 · DM-SFT (cold start)", fontsize=10.5, weight="bold", color=IN, ha="center")
box(0.2, 4.6, 3.8, 1.9, IN_F, IN,
    "4K CoT annotations\nD_text 2K  +  D_tool 2K\n"
    "(72B teacher, inverse reasoning\nfrom GT meta-actions)\n\nLR 2e-5 · 5 epochs · vision encoder frozen", fs=8.5)
arrow(4.0, 5.55, 4.7, 5.55)

# ---------- Stage 2 ----------
ax.text(7.2, 6.9, "Stage 2 · FCM-RL (mode strengthening, MP-GRPO)", fontsize=10.5, weight="bold", color=OP, ha="center")
box(4.7, 4.6, 5.0, 1.9, OP_F, OP,
    "per query q: FORCE G/2 <think_text> + G/2 <think_tool>\n"
    "all G pooled → one reward normalization\n\n"
    "intra-mode contrast: which text path is best\n"
    "inter-mode contrast: which mode suits this scene", fs=8.5)
# reward box stage 2
box(5.15, 3.0, 4.1, 1.3, LOSS_F, LOSS, "R = R_acc + R_fmt\nR_acc: 0.7·speed + 0.3·traj\n(weighted edit distance)\nR_fmt: format consistency", fs=8.5, lw=1.3)
arrow(7.2, 4.6, 7.2, 4.3, c=LOSS)
arrow(9.7, 5.55, 10.4, 5.55)

# ---------- Stage 3 ----------
ax.text(12.2, 6.9, "Stage 3 · AMS-RL (intelligent selection)", fontsize=10.5, weight="bold", color=ALT, ha="center")
box(10.4, 4.6, 3.4, 1.9, ALT_F, ALT,
    "native GRPO, no forcing:\nagent emits its own\n<think_text> / <think_tool>\n\n"
    "learns WHEN tools pay off", fs=8.5)
box(10.55, 3.0, 3.1, 1.3, LOSS_F, LOSS, "R = R_acc + R_fmt\n+ 1(mode=M_tool)·R_tool\nR_tool: contrastive,\nonly if tool path beats\ntext-only baseline", fs=8.5, lw=1.3)
arrow(12.1, 4.6, 12.1, 4.3, c=LOSS)

# ---------- bottom: GRPO core (shared) ----------
box(0.2, 0.5, 13.6, 2.1, "#ffffff", WIRE,
    "GRPO core (all RL stages):  J(θ) = E[ (1/G) Σ min( wi·Ai , clip(wi, 1±ε)·Ai ) − β·DKL(πθ‖πref) ]\n"
    "wi = πθ(oi|q)/πθold(oi|q)   ·   Ai = (ri − mean(r))/std(r) over the pooled group (no critic)\n"
    "Stage 2/3: T=1.0, G=4, max 3 tool calls, LR 1e-6 constant, 1 epoch each, 8×H20 (7 train + 1 vLLM server)", fs=9, lw=1.2)

# init note
box(0.2, 2.9, 3.8, 0.9, "#f5f4ed", WIRE,
    "init: DriveAlign-3B\n(530K VQA domain alignment\non Qwen2.5-VL-3B)", fs=8.5, lw=1.2)
arrow(2.1, 3.8, 2.1, 4.6, c=IN)

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/071-Agent-driveagent-r1/fig-cascaded-rl.png",
            dpi=150, bbox_inches="tight")
print("saved")
