"""Self-drawn: tri-plane VAE of GenieDrive (Eq.1-3).
A downsampled voxel feature volume S is projected onto three planes;
each plane is compressed by a transformer with a learnable [CLS]-like
token; decoding multiplies the three planes (Hadamard product) and adds
a positional encoding before upsampling 3D conv.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(11.5, 5.2))
ax.set_xlim(0, 11.5)
ax.set_ylim(0, 5.4)
ax.axis("off")

def tbox(x, y, w, h, txt, fs=9.5, fc="white", ec="#6e737b"):
    r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                fc=fc, ec=ec, lw=1.6)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs)

def arr(x1, y1, x2, y2, c="#6e737b"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                                 mutation_scale=14, lw=1.6, color=c))

ax.text(5.75, 5.0, "Tri-plane VAE (GenieDrive Sec.3.1): compress a 3D volume into 3 planes",
        ha="center", fontsize=12.5, weight="bold")

# input volume
tbox(0.25, 2.35, 1.5, 1.5, "$O$\nocc input\n$H\\times W\\times D$", fs=10, fc="#e2eff5", ec="#1a6fa3")
arr(1.75, 3.1, 2.35, 3.1)
ax.text(2.05, 3.32, "3D conv\n$g_\\phi$", fontsize=8.5, ha="center", color="#0b7a5c")
tbox(2.35, 2.35, 1.5, 1.5, "$S$\n$h\\times w\\times d\\times C$", fs=10, fc="#e2eff5", ec="#1a6fa3")

# three planes
planets = [("xy", "z"), ("xz", "y"), ("yz", "x")]
px = [4.15, 5.75, 7.35]
for (pn, rd), x in zip(planets, px):
    arr(3.85, 3.35 if pn=="xy" else 3.1 if pn=="xz" else 2.85, x, 4.05)
    tbox(x, 3.5, 1.3, 1.1, f"$Z_{pn}$\nproj on\n{pn} (Eq.1)", fs=9.2, fc="#f8ecd2", ec="#9e5c00")
    ax.text(x+0.65, 3.28, f"[token] + transformer $F_{pn}$", fontsize=7.8, ha="center", color="#9e5c00")

ax.text(4.6, 2.5, "Hadamard product  $Z_{xy}\\odot Z_{yz}\\odot Z_{xz}$  +  PE($x,y,z$)", fontsize=9.5,
        ha="center", color="#0b7a5c")
for x in px:
    arr(x+0.65, 3.5, 4.6, 2.62)
arr(4.6, 2.35, 4.6, 2.0)

tbox(3.9, 1.0, 1.4, 0.95, "$\\hat{V}$\n$h\\times w\\times d\\times C$", fs=9.5, fc="#dcf2e8", ec="#0b7a5c")
arr(5.3, 1.48, 6.0, 1.48)
ax.text(5.65, 1.7, "3D conv\n$f_\\psi$", fontsize=8.5, ha="center", color="#0b7a5c")
tbox(6.0, 1.0, 1.4, 0.95, "$\\hat{O}$\nreconstructed\n$H\\times W\\times D\\times C$", fs=9.5, fc="#dcf2e8", ec="#0b7a5c")

ax.text(8.35, 2.6, "$L_{VAE}=L_{CE}+L_{Lov}(O,\\hat{O})+L_{KL}(Z,\\mathcal{N}(0,I))$  (Eq.3)", fontsize=10.2,
        color="#b34a06", bbox=dict(fc="#f9e3d6", ec="#f1bd9c", boxstyle="round,pad=0.4"))
ax.text(8.35, 1.6, "Latent size is only 58% of prior methods (DOME/COME)\n"
                   "while reconstruction mIoU 86.15 > 83.08 (Table 1).", fontsize=9.3, color="#333333")

plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/042-WM-geniedrive/fig-triplane.png", dpi=150)
print("saved fig-triplane.png")
