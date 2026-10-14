"""Self-drawn figure for Drive-OccWorld xray:
Unified action-conditioning interface: heterogeneous actions
(velocity / steering->curvature / trajectory / command) ->
Fourier embeddings -> concat + learned projection ->
conditional cross-attention in the world decoder.
Includes a small worked numeric example. Bright theme."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

IN = "#1a6fa3"; OUT = "#9e5c00"; OP = "#0b7a5c"; NORM = "#7a6b00"
LOSS = "#b34a06"; ALT = "#6a55c2"; WIRE = "#6e737b"
IN_FILL = "#e2eff5"; OUT_FILL = "#f8ecd2"; OP_FILL = "#dcf2e8"
ALT_FILL = "#ebe7f8"; LOSS_FILL = "#f9e3d6"; NORM_FILL = "#f8f4cf"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2),
                               gridspec_kw={"width_ratios": [1.35, 1]})
for ax in (ax1, ax2):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

def box(ax, x, y, w, h, text, fc, ec, fs_=9):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                         fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs_, linespacing=1.55)

def arrow(ax, x1, y1, x2, y2, c=WIRE, w=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=w, shrinkA=2, shrinkB=4))

# ---------- left: interface diagram ----------
ax1.text(5, 9.5, "Unified conditioning interface", fontsize=11.5, weight="bold", ha="center")
acts = [
    ("velocity\n(vx, vy) m/s", IN_FILL, IN),
    ("steering -> curvature\n1/radius, m^-1", OUT_FILL, OUT),
    ("trajectory\n(Dx, Dy) m", ALT_FILL, ALT),
    ("command\ngo fwd / turn L / R", OP_FILL, OP),
]
for i, (t, fc, ec) in enumerate(acts):
    box(ax1, 0.2 + i * 2.45, 7.0, 2.3, 1.3, text=t, fc=fc, ec=ec, fs_=8.5)
    arrow(ax1, 1.35 + i * 2.45, 7.0, 1.35 + i * 2.45, 6.1, ec)

box(ax1, 2.2, 4.3, 5.6, 1.6, text="Fourier embeddings (Tancik et al. 2020)\nlow-dim action -> high-freq features", fc=NORM_FILL, ec=NORM)
ax1.text(5, 3.78, "why Fourier: an MLP on raw (vx,vy) can't fit sharp dependence on speed;\n"
         "sin/cos features at multiple frequencies give it high-freq basis for free", ha="center",
         fontsize=8.5, color=WIRE, style="italic")
box(ax1, 2.2, 2.2, 5.6, 1.4, text="concat + learned projection\n-> action embedding\n(matches cross-attn dim)", fc=IN_FILL, ec=IN)
arrow(ax1, 5, 4.3, 5, 3.98, NORM)
box(ax1, 2.2, 0.3, 5.6, 1.5, text="conditional cross-attention\nQ: BEV queries, K/V: action embedding\n(action steers the forecast)", fc=OUT_FILL, ec=OUT)
arrow(ax1, 5, 2.2, 5, 1.7, IN)
ax1.text(5, 0.0, "Table 7: cross-attn + Fourier beats additive embeddings", ha="center",
         fontsize=8.5, color=WIRE, style="italic")

# ---------- right: worked Fourier example ----------
ax2.text(5, 9.5, "Fourier embedding, worked example", fontsize=11.5, weight="bold", ha="center")
ax2.text(0.4, 8.85, "Take velocity (vx, vy) = (8.0, 0.3) m/s.", fontsize=9.5)
ax2.text(0.4, 8.35, "Fourier features with bands B = {1, 2, 4}:", fontsize=9.5)
ax2.text(0.4, 7.55, "for each band b:  [ sin(2 pi b vx), cos(2 pi b vx),\n"
         "                        sin(2 pi b vy), cos(2 pi b vy) ]", fontsize=9,
         bbox=dict(fc="white", ec=WIRE, lw=1.0, boxstyle="round,pad=0.3"))
ax2.text(0.4, 6.35, "b=1: sin(2 pi x 8.0)=sin(16 pi)=0.000,\n"
         "       cos(16 pi)=1.000\n"
         "       sin(2 pi x 0.3)=sin(0.6 pi)=0.951,\n"
         "       cos(0.6 pi)=-0.309", fontsize=9, color=IN,
         bbox=dict(fc=IN_FILL, ec=IN, lw=1.0, boxstyle="round,pad=0.3"))
ax2.text(0.4, 4.7, "b=2: sin(32 pi)=0.000, cos(32 pi)=1.000\n"
         "       sin(1.2 pi)=-0.588, cos(1.2 pi)=-0.809\n"
         "b=4: sin(64 pi)=0.000, cos(64 pi)=1.000\n"
         "       sin(2.4 pi)= 0.951, cos(2.4 pi)= 0.309", fontsize=9, color=IN,
         bbox=dict(fc=IN_FILL, ec=IN, lw=1.0, boxstyle="round,pad=0.3"))
ax2.text(0.4, 2.95, "Note: vx=8.0 is an integer -> sin terms vanish\n"
         "at every integer band; vy=0.3 keeps varying.\n"
         "The embedding is 12-dim here (paper's dim\n"
         "unspecified); a learned linear layer then\n"
         "projects it to the cross-attention width.", fontsize=9, color=WIRE)
ax2.text(0.4, 1.15, "Geometric reading: Fourier embedding turns\n"
         "a smooth scalar (speed) into a comb of\n"
         "oscillations; the network picks which teeth\n"
         "of the comb matter for this forecast.", fontsize=9.5, color=OUT, weight="bold",
         bbox=dict(fc=OUT_FILL, ec=OUT, lw=1.2, boxstyle="round,pad=0.3"))

plt.tight_layout(pad=0.7)
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/054-WM-drive-occworld/fig-action-cond.png",
            dpi=150, bbox_inches="tight")
print("saved fig-action-cond.png")
