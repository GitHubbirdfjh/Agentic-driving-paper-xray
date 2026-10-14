"""Self-drawn: conventional temporal fusion vs GaussianWorld's world-model
paradigm (GaussianWorld Sec.3.1, Eq.1-4).
Conventional: perceive each frame independently, align, fuse.
GaussianWorld: keep last frame's Gaussians, evolve them with current input.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 7.2))
for ax in (ax1, ax2):
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

def box(ax, x, y, w, h, txt, fs=9.2, fc="white", ec="#6e737b"):
    r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=1.6)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs)

def arr(ax, x1, y1, x2, y2, c="#6e737b", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=14,
                                 lw=1.6, color=c, linestyle=ls))

# --- top: conventional pipeline
ax1.text(5.75, 4.15, "Conventional: perception -> transformation -> fusion (Eq.2)", ha="center",
         fontsize=12, weight="bold")
for i, t in enumerate(["$x_{T-2}$", "$x_{T-1}$", "$x_T$"]):
    x = 0.7 + i * 1.7
    box(ax1, x, 2.35, 1.3, 0.8, t, fc="#e2eff5", ec="#1a6fa3")
    box(ax1, x, 1.35, 1.3, 0.8, "$z_n=\\mathrm{Per}(x_n)$", fs=8.6, fc="#f5f4ed")
    arr(ax1, x + 0.65, 2.35, x + 0.65, 2.15)
box(ax1, 6.1, 1.35, 1.7, 1.8, "$a_n=\\mathrm{Trans}(z_n,p_n)$\nalign by ego", fs=9, fc="#dcf2e8", ec="#0b7a5c")
for x in (0.7, 2.4, 4.1):
    arr(ax1, x + 1.35, 1.75, 6.1, 2.2)
box(ax1, 8.4, 1.35, 2.4, 1.8, "$y_T=\\mathrm{Fuse}(a_T,...,a_{T-t})$\nBEV/voxel fusion", fs=9, fc="#f8ecd2", ec="#9e5c00")
arr(ax1, 7.8, 2.25, 8.4, 2.25)
ax1.text(5.75, 0.55, "Each frame re-perceives the full scene: fusion ignores that adjacent frames differ\n"
                     "only by ego motion + dynamic objects (Table 2: 3-frame fusion, 379ms / 9993MB, mIoU 20.24).",
         ha="center", fontsize=9.3, style="italic", color="#333333")

# --- bottom: world model
ax2.text(5.75, 4.15, "GaussianWorld: forecast $z_T$ from $z_{T-1}$ given $x_T$ (Eq.3-4)", ha="center",
         fontsize=12, weight="bold", color="#0b7a5c")
box(ax2, 0.5, 1.9, 2.0, 1.2, "$g_{T-1}$\n3D Gaussians\n$\\{p,s,r,c,f\\}$", fc="#e2eff5", ec="#1a6fa3")
box(ax2, 3.1, 1.9, 2.0, 1.2, "$x_T$\ncurrent RGB\nimages", fc="#e2eff5", ec="#1a6fa3")
box(ax2, 5.7, 1.9, 2.2, 1.2, "$g_T=w(g_{T-1},x_T)$\nworld model\nEq.(6)", fc="#dcf2e8", ec="#0b7a5c")
arr(ax2, 2.5, 2.5, 3.1, 2.5)
arr(ax2, 5.1, 2.5, 5.7, 2.5)
box(ax2, 8.5, 1.9, 2.2, 1.2, "$y_T=h(g_T)$\nGS-to-Occ head\nEq.(4)", fc="#f8ecd2", ec="#9e5c00")
arr(ax2, 7.9, 2.5, 8.5, 2.5)
arr(ax2, 6.8, 1.9, 6.8, 0.9, c="#0b7a5c")
box(ax2, 5.9, 0.05, 1.8, 0.75, "$g_T$ carried to next frame", fs=8.6, fc="#dcf2e8", ec="#0b7a5c")
arr(ax2, 6.8, 0.05, 6.8, -0.5, c="#0b7a5c")
ax2.text(5.75, -1.0, "Nothing is re-perceived: the prior frame's scene IS the scene; the model only predicts\n"
         "how it evolves (Table 2: 228ms / 7030MB vs 225ms / 6958MB single-frame, mIoU 21.87).",
         ha="center", fontsize=9.3, style="italic", color="#333333")

plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/043-WM-gaussianworld/fig-streaming-paradigm.png",
            dpi=150, bbox_inches="tight")
print("saved fig-streaming-paradigm.png")
