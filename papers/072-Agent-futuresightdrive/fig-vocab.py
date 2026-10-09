# -*- coding: utf-8 -*-
"""fig-vocab.py: how FSDrive turns an MLLM into a visual generator.
(1) MoVQGAN quantizes a 128x192 future frame (8x downsample) into 16x24 = 384
discrete tokens from a 16384-entry codebook.
(2) The LLM vocabulary is expanded with <|0|>...<|16383|> special tokens.
(3) Training/inference is plain next-token prediction over a mixed sequence:
text prompt -> 384 visual tokens -> waypoint text.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

WIRE = "#6e737b"
IN, IN_F = "#1a6fa3", "#e2eff5"
OP, OP_F = "#0b7a5c", "#dcf2e8"
ALT, ALT_F = "#6a55c2", "#ebe7f8"
OUT, OUT_F = "#9e5c00", "#f8ecd2"

fig = plt.figure(figsize=(13.5, 7.0))
ax = fig.add_subplot(111)
ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.2); ax.axis("off")

ax.text(6.75, 6.85, "One vocabulary, two modalities: next-token prediction over text + visual tokens",
        fontsize=12.5, weight="bold", ha="center")

# ---- left: quantization ----
ax.text(2.2, 6.2, "128×192 future frame", fontsize=10, weight="bold", color=IN, ha="center")
fr = mpatches.FancyBboxPatch((1.0, 4.35), 2.4, 1.6, boxstyle="round,pad=0.04",
                             facecolor="#dfe9f2", edgecolor=IN, linewidth=1.8)
ax.add_patch(fr)
road = plt.Polygon([[1.5, 4.35], [2.9, 4.35], [2.6, 5.95], [1.8, 5.95]], closed=True,
                   facecolor="#9aa7b5", edgecolor="none")
ax.add_patch(road)
ax.annotate("", xy=(3.9, 5.15), xytext=(3.4, 5.15),
            arrowprops=dict(arrowstyle="->", color=WIRE, lw=1.8))
ax.text(4.55, 5.55, "MoVQGAN\n8× down", ha="center", fontsize=8.5, color=WIRE)

# token grid 16x24 schematic (draw 8x12 cells to suggest it)
gx, gy, cw, chh = 5.2, 4.15, 0.16, 0.16
rng = np.random.default_rng(7)
for r in range(12):
    for c in range(16):
        col = plt.cm.viridis(rng.random() * 0.9)
        sq = mpatches.Rectangle((gx + c*cw, gy + (11-r)*chh), cw*0.94, chh*0.94,
                                facecolor=col, edgecolor="none")
        ax.add_patch(sq)
ax.text(6.5, 6.2, "16×24 = 384 tokens, raster order", fontsize=10, weight="bold", color=ALT, ha="center")
ax.text(6.5, 3.85, "each cell = one codebook index ∈ [0, 16384)", fontsize=8.5, color=WIRE, ha="center")

# ---- right: vocab ----
ax.text(10.9, 6.2, "expanded vocabulary", fontsize=10, weight="bold", color=OP, ha="center")
b1 = mpatches.FancyBboxPatch((9.4, 5.15), 3.0, 0.8, boxstyle="round,pad=0.05",
                             facecolor=IN_F, edgecolor=IN, linewidth=1.5)
ax.add_patch(b1)
ax.text(10.9, 5.55, "text tokens (Qwen2-VL)", ha="center", fontsize=9, color=IN)
b2 = mpatches.FancyBboxPatch((9.4, 4.15), 3.0, 0.8, boxstyle="round,pad=0.05",
                             facecolor=ALT_F, edgecolor=ALT, linewidth=1.5)
ax.add_patch(b2)
ax.text(10.9, 4.55, "<|0|> … <|16383|>  (visual)", ha="center", fontsize=9, color=ALT, weight="bold")
ax.text(10.9, 3.85, "added as special tokens;\nvision tower & projector frozen,\nLLM fully fine-tuned", fontsize=8.5, color=WIRE, ha="center")

# ---- bottom: mixed sequence ----
ax.text(6.75, 3.2, "a training sequence is just tokens", fontsize=11, weight="bold", ha="center", color=WIRE)
segs = [("generate future frame:", IN_F, IN, 1.6), ("<|342|> <|17|> … (384 visual)", ALT_F, ALT, 4.2),
        ("waypoints: [(1.2,0.3), …]", OUT_F, OUT, 3.2)]
x = 0.7
for txt, face, edge, w in segs:
    r = mpatches.FancyBboxPatch((x, 1.7), w, 1.0, boxstyle="round,pad=0.05",
                                facecolor=face, edgecolor=edge, linewidth=1.6)
    ax.add_patch(r)
    ax.text(x + w/2, 2.2, txt, ha="center", va="center", fontsize=9, color=edge, weight="bold")
    x += w + 0.35
ax.text(6.75, 1.15, "L = −Σ log P(qi | q<i): the same LM loss predicts the next word and the next image patch",
        fontsize=9.5, color=WIRE, ha="center", style="italic")
ax.text(6.75, 0.55, "detokenizer turns the 384 predicted indices back into pixels — that image IS the visual CoT",
        fontsize=9.5, color=WIRE, ha="center")

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/072-Agent-futuresightdrive/fig-vocab.png",
            dpi=150, bbox_inches="tight")
print("saved")
