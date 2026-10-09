# -*- coding: utf-8 -*-
"""NavGPT-2 每步前向的 token 流向与形状变化（N=3 个候选视角的具体例子）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "cm"

BG = "#faf9f5"
C_IN = "#1a6fa3"; C_IN_F = "#e2eff5"
C_OP = "#0b7a5c"; C_OP_F = "#dcf2e8"
C_OUT = "#9e5c00"; C_OUT_F = "#f8ecd2"
C_ALT = "#6a55c2"; C_ALT_F = "#ebe7f8"
INK = "#141413"; WIRE = "#6e737b"

fig, ax = plt.subplots(figsize=(13.2, 4.6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 1320); ax.set_ylim(0, 460)
ax.axis("off")

boxes = [
    # (x, w, label, shape, color, fill)
    (20, 150, "ViT-g/14\n(frozen)", "257 x 1408", C_IN, C_IN_F),
    (215, 150, "Q-former\n(32 queries)", "32 x 768", C_OP, C_OP_F),
    (410, 150, "linear W", "32 x 2048", C_OP, C_OP_F),
    (605, 190, "FlanT5 encoder\n(frozen, decoder deleted)", "96 x 2048", C_ALT, C_ALT_F),
    (840, 150, "merge 32->1\n+ proj", "1 x 768", C_OP, C_OP_F),
    (1035, 150, "policy net\nDUET global", "node scores", C_OUT, C_OUT_F),
]
for x, w, label, shape, c, f in boxes:
    r = patches.FancyBboxPatch((x, 150), w, 150, boxstyle="round,pad=8,rounding_size=10",
                               facecolor=f, edgecolor=c, linewidth=1.8)
    ax.add_patch(r)
    ax.text(x + w / 2, 235, label, ha="center", va="center", fontsize=11.5, color=INK, weight="bold")
    ax.text(x + w / 2, 172, shape, ha="center", va="center", fontsize=10.5, color=WIRE,
            family="monospace")
    if x > 20:
        ax.annotate("", xy=(x, 225), xytext=(x - 45, 225),
                    arrowprops=dict(arrowstyle="-|>", color=WIRE, lw=1.4, shrinkA=0, shrinkB=2))

ax.text(95, 330, "view image\n224x224", ha="center", fontsize=10, color=C_IN)
ax.text(290, 330, "cross-attend\n257 patch tokens", ha="center", fontsize=10, color=C_OP)
ax.text(485, 330, "substitute\n<image> slots", ha="center", fontsize=10, color=C_OP)
ax.text(700, 330, "extract latents\nbetween [INST]..[/INST]\nfor instruction", ha="center", fontsize=10, color=C_ALT)
ax.text(915, 330, "Conv1d(32->1)\nLinear 2048->768", ha="center", fontsize=10, color=C_OP)
ax.text(1110, 330, "argmax over\ngraph nodes", ha="center", fontsize=10, color=C_OUT)

ax.text(20, 60, "N = 3 candidate views  ->  3 x 32 = 96 image token slots in the prompt; "
                "instruction latents: L x 2048 (L = instruction length)",
        fontsize=10, color=WIRE, family="monospace")
ax.text(20, 30, "frozen parts never receive gradients; only Q-former (+W) in stage 1 and the policy net in stage 2 are trained",
        fontsize=10, color=WIRE)

plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-40/vln/navgpt2-xray/fig-token-flow.png", dpi=150,
            facecolor=BG, bbox_inches="tight")
print("saved fig-token-flow.png")
