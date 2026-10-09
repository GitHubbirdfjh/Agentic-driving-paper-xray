"""Self-drawn figure for Drive-OccWorld xray:
Semantic- and motion-conditional normalization (Eq. 5).
F_bev -> LayerNorm (no affine) -> modulate by (gamma*, beta*) derived from
three sources: semantic labels, ego pose, backward centripetal flow.
Bright theme."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

IN = "#1a6fa3"; OUT = "#9e5c00"; OP = "#0b7a5c"; NORM = "#7a6b00"
LOSS = "#b34a06"; ALT = "#6a55c2"; WIRE = "#6e737b"
IN_FILL = "#e2eff5"; OUT_FILL = "#f8ecd2"; OP_FILL = "#dcf2e8"
ALT_FILL = "#ebe7f8"; NORM_FILL = "#f8f4cf"

fig = plt.figure(figsize=(13, 7.0))
ax = fig.add_subplot(111)
ax.set_xlim(0, 13); ax.set_ylim(0, 8.2); ax.axis("off")

def box(x, y, w, h, text, fc, ec, fs_=9.5):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                         fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs_, linespacing=1.55)

def arrow(x1, y1, x2, y2, c=WIRE, rad=0.0, w=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=w,
                                connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=4))

ax.text(6.5, 7.8, "Semantic- and motion-conditional normalization  (Eq. 5:  F~ = gamma* . LN(F) + beta*)",
        ha="center", fontsize=12, weight="bold")

# input BEV embedding
box(0.3, 4.9, 2.4, 1.5, "F_bev in R^{h x w x c}\nh=w=200, c channels", IN_FILL, IN)
ax.text(1.5, 4.6, "raw BEV: ray-shaped patterns\n(same ray -> same pixel -> same feature)", ha="center",
        fontsize=8.5, color=WIRE, style="italic")

# LayerNorm without affine
box(3.5, 4.9, 2.4, 1.5, "LayerNorm (no affine)\nzero mean, unit var\nper location", NORM_FILL, NORM)
arrow(2.7, 5.65, 3.5, 5.65, IN)

# modulation
box(6.7, 4.9, 2.6, 1.5, "x gamma*  + beta*\n(adaptive affine,\nper-location, per-channel)", OUT_FILL, OUT)
arrow(5.9, 5.65, 6.7, 5.65, NORM)
ax.text(6.3, 6.0, "scale", ha="center", fontsize=9, color=OUT, weight="bold")
ax.text(6.3, 5.32, "shift", ha="center", fontsize=9, color=OUT, weight="bold")

box(10.1, 4.9, 2.6, 1.5, "F~_bev: semantic &\nmotion-aware BEV\n-> memory queue", OP_FILL, OP)
arrow(9.3, 5.65, 10.1, 5.65, OUT)

# three sources of (gamma*, beta*)
ax.text(1.5, 3.9, "Where do (gamma*, beta*) come from? Three independent paths:", fontsize=10.5, weight="bold")

# path 1: semantic
box(0.3, 1.7, 3.7, 1.7, "(gamma_s, beta_s): semantic\nvoxel labels S in R^{h x w x d x 1}\nvia argmax head -> one-hot -> conv", ALT_FILL, ALT)
arrow(2.15, 4.9, 2.15, 3.4, ALT, rad=0.25)
arrow(2.15, 3.4, 7.4, 5.15, ALT, rad=0.1)
ax.text(0.5, 1.15, "tells each BEV cell WHICH class it is\n-> amplifies instance responses", fontsize=8.5, color=ALT, style="italic")

# path 2: ego motion
box(4.6, 1.7, 3.7, 1.7, "(gamma_e, beta_e): ego motion\npose matrix E^{+t}_{-t}=[R,T]\nflatten -> MLP", IN_FILL, IN)
arrow(6.45, 4.9, 6.45, 3.4, IN)
ax.text(6.45, 1.15, "compensates EGO movement between\nframes (aligns BEV across time)", fontsize=8.5, color=IN, style="italic")

# path 3: agent motion
box(8.9, 1.7, 3.7, 1.7, "(gamma_f, beta_f): agent motion\n3D backward centripetal flow\nF in R^{h x w x d x 3} -> encode", OUT_FILL, OUT)
arrow(10.75, 4.9, 10.75, 3.4, OUT, rad=-0.25)
arrow(10.75, 3.4, 8.6, 5.15, OUT, rad=-0.1)
ax.text(10.75, 1.15, "compensates OTHER AGENTS'\nmovement (voxel -> instance center)", fontsize=8.5, color=OUT, style="italic")

# why box at bottom
ax.text(0.3, 0.55, "Why this works: raw BEV features are smeared along camera rays (appendix Fig. 7: 'before norm' shows ray streaks).\n"
        "Semantic labels tell the affine transform which cells belong to vehicles/pedestrians, so those cells get amplified;\n"
        "ego/agent motion tell it how the world moved between frames, so history aligns instead of smearing. Table 6: ego-motion alone +1.9 mIoUf.",
        fontsize=9, color=WIRE, va="top",
        bbox=dict(fc="white", ec=WIRE, lw=1.0, boxstyle="round,pad=0.35"))

plt.tight_layout(pad=0.6)
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/054-WM-drive-occworld/fig-condnorm.png",
            dpi=150, bbox_inches="tight")
print("saved fig-condnorm.png")
