"""Action MoE decoder-layer schematic (bright theme, skill 5.3 palette)."""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.patheffects as pe

IN, IN_F = "#1a6fa3", "#e2eff5"
OP, OP_F = "#0b7a5c", "#dcf2e8"
ALT, ALT_F, ALT_HI = "#6a55c2", "#ebe7f8", "#cdc2f0"
OUT, OUT_F = "#9e5c00", "#f8ecd2"
NORM, NORM_F = "#7a6b00", "#f8f4cf"
PARAM = "#a8457c"
WIRE = "#6e737b"
BG = "#faf9f5"

fig, ax = plt.subplots(figsize=(13, 6.6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 13)
ax.set_ylim(0, 7)
ax.axis("off")

def box(x, y, w, h, label, edge, fill, fs=11, lw=1.5, sub=None):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                       ec=edge, fc=fill, lw=lw)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2 + (0.08 if sub else 0), label, ha="center",
            va="center", fontsize=fs, color=edge, style="italic")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.22, sub, ha="center", va="center",
                fontsize=9.5, color=edge)
    return b

def arrow(x1, y1, x2, y2, color=WIRE, lw=1.3, label=None, lx=0, ly=0.12):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=10, shrinkA=0, shrinkB=1))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=10,
                color=color, ha="center", va="bottom",
                bbox=dict(fc=BG, ec="none", pad=1))

# input
box(0.4, 3.05, 1.7, 0.9, r"$h^{(\ell-1)}$", IN, IN_F, fs=14)
ax.text(1.25, 2.75, r"$\in \mathbb{R}^d$", ha="center", fontsize=10, color=WIRE)

# router
box(3.1, 5.5, 1.8, 0.85, "Router", OP, OP_F, fs=11)
ax.text(4.0, 5.22, r"$R_{\mathrm{action}}$", ha="center", fontsize=11,
        color=PARAM, style="italic")
arrow(1.9, 3.9, 3.35, 5.55, color=OP)

# skill experts
skills = ["E1 Merging", "E2 Overtaking", "E3 Emergency Brake",
          "E4 Give Way", "E5 Traffic Sign", "E6 ParkingExit"]
ys = [5.75, 4.85, 3.95, 3.05, 2.15, 1.25]
sel = {0: "0.55", 2: "0.20", 1: "0.08"}  # selected idx -> r
for i, (s, y) in enumerate(zip(skills, ys)):
    hi = i in sel
    box(6.0, y - 0.32, 2.1, 0.64, s, ALT, ALT_HI if hi else ALT_F,
        fs=10, lw=2.6 if hi else 1.5)
    arrow(2.1, 3.5, 6.0, y, color=WIRE, lw=1.0)
    if hi:
        off = {0: (0.62, 0.16), 1: (0.75, -0.28), 2: (0.62, 0.16)}[i]
        arrow(4.0, 5.5, 6.35, y + 0.28, color=ALT, lw=1.6,
              label=r"$r=%s$" % sel[i], lx=off[0], ly=off[1])

# shared expert
box(3.1, 1.05, 1.8, 0.85, r"$E_{\mathrm{share}}$", OP, OP_F, fs=11)
arrow(1.9, 3.1, 3.35, 1.75, color=OP)

# weighted sum of skills
c1 = Circle((9.35, 4.6), 0.32, ec=NORM, fc=NORM_F, lw=1.8)
ax.add_patch(c1)
ax.text(9.35, 4.6, "+", ha="center", va="center", fontsize=16, color=NORM)
for i in sel:
    arrow(8.1, ys[i], 9.03, 4.6, color=ALT, lw=1.6)
ax.text(8.75, 6.15, "top-3 weighted sum", fontsize=10, color=ALT, ha="center",
        style="italic")

# add shared
c2 = Circle((10.7, 3.1), 0.32, ec=NORM, fc=NORM_F, lw=1.8)
ax.add_patch(c2)
ax.text(10.7, 3.1, "+", ha="center", va="center", fontsize=16, color=NORM)
arrow(9.35, 4.28, 10.55, 3.38, color=WIRE)
arrow(4.9, 1.47, 10.38, 3.1, color=OP, lw=1.6)

# output
box(11.35, 2.65, 1.35, 0.9, r"$h^{(\ell)}$", OUT, OUT_F, fs=14)
arrow(11.02, 3.1, 11.35, 3.1, color=WIRE)

ax.text(6.5, 0.35,
        r"$h^{(\ell)} = \sum_{k \in \mathrm{top3}} r_k\, y_k^{(\ell)} + y_{\mathrm{share}}^{(\ell)}$",
        fontsize=13, color="#141413", ha="left", va="center",
        bbox=dict(fc="white", ec="#deddd8", pad=6, boxstyle="round,pad=0.3"))

plt.tight_layout(pad=0.6)
out = ("D:/workspace/goals/cross-domain-frontier-paper-reading-vla-world-models-vln-multi-agent-driving"
       "/files/paper-xray-40/vla-driving/drivemoe-xray/fig-actionmoe.png")
fig.savefig("/home/hatch/workspace/goals/cross-domain-frontier-paper-reading-vla-world-models-vln-multi-agent-driving"
            "/files/paper-xray-40/vla-driving/drivemoe-xray/fig-actionmoe.png",
            dpi=150)
print("saved")
