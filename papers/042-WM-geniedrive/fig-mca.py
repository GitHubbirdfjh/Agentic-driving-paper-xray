"""Self-drawn: Mutual Control Attention (MCA) in GenieDrive, Eq.(5)-(7).
Two parallel streams (occupancy latent Z and control latent c) exchange
information through cross-attention at every layer, unlike a one-way
occupancy <-- control conditioning.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.8)
ax.axis("off")

OCC = "#1a6fa3"   # occupancy latent Z
CTL = "#9e5c00"   # control latent c
OP  = "#0b7a5c"   # attention op

def box(x, y, w, h, fc, ec, txt, fs=10.5, tc="black", ls="-"):
    r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec=ec, lw=1.8, linestyle=ls)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs, color=tc)

def arrow(x1, y1, x2, y2, c=OP, lw=1.8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                                 mutation_scale=16, lw=lw, color=c, zorder=5))

title = ax.text(5, 6.35, "Mutual Control Attention (GenieDrive Eq.5-7): occupancy and control refine each other",
                ha="center", fontsize=12.5, weight="bold")

# left column: Z stream
box(0.6, 4.9, 2.8, 0.75, "#e2eff5", OCC, "$Z_l$  occupancy latent")
box(0.6, 3.55, 2.8, 0.75, "#e2eff5", OCC, "$Z'_l$  after cross-attn")
box(0.6, 2.2, 2.8, 0.75, "#e2eff5", OCC, "$Z_{l+1}$  after self-attn")
# right column: c stream
box(6.6, 4.9, 2.8, 0.75, "#f8ecd2", CTL, "$c_l$  control latent")
box(6.6, 3.55, 2.8, 0.75, "#f8ecd2", CTL, "$c_{l+1}$  after cross-attn")

# center op boxes
box(4.0, 4.35, 2.0, 0.6, "#dcf2e8", OP, "Attn($Q_{Z_l}, K_{c_l}, V_{c_l}$)", fs=9.2)
box(4.0, 2.95, 2.0, 0.6, "#dcf2e8", OP, "Attn($Q_{Z'_l}, K_{Z'_l}, V_{Z'_l}$)", fs=9.2)
box(4.0, 1.45, 2.0, 0.6, "#dcf2e8", OP, "Attn($Q_{c_l}, K_{Z_{l+1}}, V_{Z_{l+1}}$)", fs=9.2)

# arrows: Eq5: Z_l -> Z'_l via cross attn with c_l as K,V
arrow(3.4, 5.28, 4.0, 4.65); arrow(6.6, 5.28, 6.0, 4.65)
arrow(4.6, 4.35, 3.4, 3.95)
ax.text(5.0, 5.05, "Eq.(5): control informs occupancy", fontsize=9.5, color=OP, ha="center")

# Eq6: Z'_l -> Z_{l+1} via self attn
arrow(2.0, 3.55, 2.0, 2.95)
ax.text(2.55, 3.25, "Eq.(6)", fontsize=9.5, color=OP)

# Eq7: c_l -> c_{l+1} via cross attn with Z_{l+1} as K,V
arrow(6.6, 4.9, 6.0, 1.75)
arrow(3.4, 2.2, 4.0, 1.75)
arrow(6.0, 1.45, 6.6, 3.55)
ax.text(5.0, 2.45, "Eq.(7): occupancy refines control", fontsize=9.5, color=OP, ha="center")

ax.text(5, 0.55, "Naive alternative (ablation 'w/o MCA'): Z and c each do self-attention in isolation, no cross-talk.\n"
                 "Table 3: removing MCA drops 3s mIoU 35.83 -> 30.48 (-5.35), 3s IoU 47.08 -> 41.81 (-5.27).",
        ha="center", fontsize=9.8, style="italic", color="#333333",
        bbox=dict(fc="#f5f4ed", ec="#deddd8", boxstyle="round,pad=0.5"))

plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/042-WM-geniedrive/fig-mca.png", dpi=150)
print("saved fig-mca.png")
