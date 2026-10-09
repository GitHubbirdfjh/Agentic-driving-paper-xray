# -*- coding: utf-8 -*-
"""Decomposed 4D attention schematic for Panacea xray.
Panel 1: the joint latent layout: H x (W*6) panoramic strip, V=6 views, T frames.
Panel 2: intra-view attention (within one view chunk).
Panel 3: cross-view attention (each chunk looks at its two neighbors, wraparound).
Panel 4: cross-frame attention (same spatial position across T frames).
Bright palette per skill 5.3.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

INK = "#141413"; WIRE = "#6e737b"
IN = "#1a6fa3"; IN_FILL = "#e2eff5"
OUT = "#9e5c00"; OUT_FILL = "#f8ecd2"
OP = "#0b7a5c"; OP_FILL = "#dcf2e8"
ALT = "#6a55c2"; ALT_FILL = "#ebe7f8"
LOSS = "#b34a06"

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
fig.patch.set_facecolor("#faf9f5")

# ---------- Panel 1: joint latent layout ----------
ax = axes[0]
ax.set_xlim(0, 15); ax.set_ylim(0, 5.2)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("joint latent  z:  H x (WxV) x T x C", fontsize=11, color=INK, pad=10)
V, H, W = 6, 2.0, 2.0
for v in range(V):
    x0 = v * W
    face = IN_FILL if v % 2 == 0 else "#ffffff"
    ax.add_patch(Rectangle((x0, 1.6), W, H, facecolor=face, edgecolor=IN, lw=1.2))
    ax.text(x0 + W / 2, 1.6 + H / 2, f"v{v}", ha="center", va="center",
            fontsize=10, color=IN, weight="bold")
ax.add_patch(Rectangle((0, 1.55), 12, 2.1, fill=False, edgecolor=INK, lw=2))
ax.text(6, 0.9, "6 views spliced along width (panoramic)", ha="center", fontsize=9.5, color=WIRE)
ax.text(6, 4.35, "one time step t", ha="center", fontsize=9.5, color=WIRE)
# time stack: three small squares stacked vertically, right of the strip
tx0 = 12.7
for i, lab in enumerate(["t", "t+1", "t+2"]):
    yy = 3.0 - i * 0.75
    ax.add_patch(Rectangle((tx0, yy), 0.5, 0.5, facecolor=OUT_FILL,
                           edgecolor=OUT, lw=1.2))
    ax.text(tx0 + 0.62, yy + 0.25, lab, ha="left", va="center",
            fontsize=9, color=OUT)
ax.annotate("", xy=(tx0 + 0.25, 1.35), xytext=(tx0 + 0.25, 3.7),
            arrowprops=dict(arrowstyle="->", color=OUT, lw=1.5))
ax.text(tx0 + 0.25, 3.95, "T frames", ha="center", fontsize=9.5, color=OUT)

# ---------- Panel 2: intra-view ----------
ax = axes[1]
ax.set_xlim(0, 12); ax.set_ylim(0, 5.2)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("intra-view attention", fontsize=11, color=INK, pad=10)
for v in range(V):
    x0 = v * W
    face = OP_FILL if v == 2 else "#ffffff"
    ax.add_patch(Rectangle((x0, 1.6), W, H, facecolor=face, edgecolor=WIRE, lw=1))
    ax.text(x0 + W / 2, 1.35, f"v{v}", ha="center", fontsize=8.5, color=WIRE)
# draw attention arrows inside chunk v2
cx, cy = 2 * W + W / 2, 1.6 + H / 2
pts = [(cx - 0.55, cy + 0.55), (cx + 0.55, cy + 0.55), (cx - 0.55, cy - 0.55),
       (cx + 0.55, cy - 0.55), (cx, cy)]
for (x1, y1) in pts[:4]:
    ax.annotate("", xy=(cx, cy), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=OP, lw=1.4,
                                connectionstyle="arc3,rad=0.25"))
ax.plot(cx, cy, "o", color=OP, ms=7)
ax.text(6, 0.75, "query in v2 looks only inside v2", ha="center", fontsize=9.5, color=OP)
ax.text(6, 4.35, "same as SD spatial self-attention", ha="center", fontsize=9.5, color=WIRE)

# ---------- Panel 3: cross-view ----------
ax = axes[2]
ax.set_xlim(0, 12); ax.set_ylim(0, 5.2)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("cross-view attention (adjacent views)", fontsize=11, color=INK, pad=10)
for v in range(V):
    x0 = v * W
    if v == 2:
        face = ALT_FILL; ec = ALT; lw = 1.6
    elif v in (1, 3):
        face = "#f3eefc"; ec = ALT; lw = 1.1
    else:
        face = "#ffffff"; ec = WIRE; lw = 1
    ax.add_patch(Rectangle((x0, 1.6), W, H, facecolor=face, edgecolor=ec, lw=lw))
    ax.text(x0 + W / 2, 1.35, f"v{v}", ha="center", fontsize=8.5, color=WIRE)
# arrows from v2 to v1 and v3
for vq in (1, 3):
    ax.annotate("", xy=(vq * W + W / 2, 2.6), xytext=(2 * W + W / 2, 2.6),
                arrowprops=dict(arrowstyle="->", color=ALT, lw=1.6,
                                connectionstyle="arc3,rad=0.3"))
# wraparound v0 <-> v5
ax.annotate("", xy=(0.15, 3.9), xytext=(11.85, 3.9),
            arrowprops=dict(arrowstyle="<->", color=ALT, lw=1.2, ls=(0, (4, 3)),
                            connectionstyle="arc3,rad=-0.35"))
ax.text(6, 4.35, "wraparound: v0 <-> v5", ha="center", fontsize=9.5, color=ALT)
ax.text(6, 0.75, "query in v2 looks at v1 and v3 only", ha="center", fontsize=9.5, color=ALT)

# ---------- Panel 4: cross-frame ----------
ax = axes[3]
ax.set_xlim(0, 4.6); ax.set_ylim(0, 5.2)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("cross-frame attention", fontsize=11, color=INK, pad=10)
# draw 4 frame strips, each a small 6-chunk panorama
T = 4
for t in range(T):
    y0 = 3.9 - t * 1.05
    for v in range(V):
        x0 = 0.3 + v * 0.66
        face = LOSS if False else "#ffffff"
        hl = (v == 2)
        ax.add_patch(Rectangle((x0, y0), 0.66, 0.8,
                               facecolor="#f9e3d6" if hl else "#ffffff",
                               edgecolor=LOSS if hl else WIRE, lw=1.4 if hl else 0.8))
    ax.text(4.42, y0 + 0.4, f"t+{t}", va="center", fontsize=8.5, color=WIRE)
# vertical arrows through the highlighted spatial position
for t in range(T - 1):
    y_top = 3.9 - t * 1.05 + 0.4
    y_bot = 3.9 - (t + 1) * 1.05 + 0.4
    ax.annotate("", xy=(0.3 + 2 * 0.66 + 0.33, y_bot + 0.32),
                xytext=(0.3 + 2 * 0.66 + 0.33, y_top - 0.32),
                arrowprops=dict(arrowstyle="<->", color=OUT, lw=1.5))
ax.text(2.3, 0.35, "same (h, w) position across T frames", ha="center",
        fontsize=9.5, color=OUT)
ax.text(2.3, 4.75, "like VLDM temporal attention", ha="center", fontsize=9.5, color=WIRE)

plt.tight_layout(pad=1.2)
out = "/home/hatch/workspace/your_files/paper-xray-40/world-model-driving/panacea-xray/fig-4d-attn.png"
plt.savefig(out, dpi=170, facecolor="#faf9f5", bbox_inches="tight")
print("saved", out)
