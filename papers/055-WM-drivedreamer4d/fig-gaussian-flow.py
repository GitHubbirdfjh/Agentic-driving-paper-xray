"""Self-drawn figure for DriveDreamer4D xray:
4D Gaussian parameter flow with per-Gaussian shapes (Eq. 1-3),
geometric reading of each step. Bright theme."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

IN = "#1a6fa3"; OUT = "#9e5c00"; OP = "#0b7a5c"; NORM = "#7a6b00"
LOSS = "#b34a06"; ALT = "#6a55c2"; WIRE = "#6e737b"
IN_FILL = "#e2eff5"; OUT_FILL = "#f8ecd2"; OP_FILL = "#dcf2e8"
ALT_FILL = "#ebe7f8"; LOSS_FILL = "#f9e3d6"; NORM_FILL = "#f8f4cf"

fig, ax = plt.subplots(figsize=(13, 6.4))
ax.set_xlim(0, 13); ax.set_ylim(0, 8); ax.axis("off")

def box(x, y, w, h, text, fc, ec, fs_=9.5):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                         fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs_, linespacing=1.55)

def arrow(x1, y1, x2, y2, c=WIRE, rad=0.0, w=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=w,
                                connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=4))

ax.text(6.5, 7.55, "One 4D Gaussian: from canonical parameters to a rendered pixel", fontsize=12,
        weight="bold", ha="center")

# row 1: canonical params
box(0.3, 5.6, 2.6, 1.5, "canonical φ\nx∈R³, γ∈R, s∈R³,\nr∈R⁴, c: SH coeffs", IN_FILL, IN)
ax.text(1.6, 5.35, "one 3D Gaussian: center, opacity,\nshape (scale+quat), color", ha="center",
        fontsize=8.5, color=WIRE, style="italic")

box(3.6, 5.6, 2.6, 1.5, "temporal field F(φ, t_gs)\nδφ = {δx, δγ, δs, δr, δc}", OP_FILL, OP)
arrow(2.9, 6.35, 3.6, 6.35, WIRE)
ax.text(3.25, 6.7, "time t", ha="center", fontsize=9, color=OP, weight="bold")

box(6.9, 5.6, 2.6, 1.5, "φ′ = φ + δφ\n(Eq. 2: deform in\ncanonical space)", OUT_FILL, OUT)
arrow(6.2, 6.35, 6.9, 6.35, OP)

box(10.2, 5.6, 2.5, 1.5, "Σ = R S Sᵀ Rᵀ\n(Eq. 1: shape matrix\nfrom s, r)", NORM_FILL, NORM)
arrow(9.5, 6.35, 10.2, 6.35, OUT)

# row 2: projection + blending
box(0.3, 2.6, 3.1, 1.5, "camera projection\nΣ′ = J V Σ Vᵀ Jᵀ\n(world→cam→2D)", ALT_FILL, ALT)
arrow(1.6, 5.6, 1.6, 4.1, NORM, rad=0.12)
ax.text(1.6, 4.75, "3D covariance → 2D splat", ha="center", fontsize=9, color=ALT, weight="bold")

box(4.1, 2.6, 4.2, 1.5, "α-blending over N ordered splats\nC = Σ_i T_i c′_i α_i,  T_i = Π_{j<i}(1−α_j)\n(Eq. 3: front-to-back compositing)", OUT_FILL, OUT)
arrow(3.4, 3.35, 4.1, 3.35, ALT)

box(9.0, 2.6, 3.7, 1.5, "pixel color C∈R³\n(+ rendered depth D̂\nvs LiDAR D for L_ori)", IN_FILL, IN)
arrow(8.3, 3.35, 9.0, 3.35, OUT)

# geometric reading strip at bottom
ax.text(0.3, 1.75, "Geometric reading (what each step does in space):", fontsize=10.5, weight="bold")
geo = [
    "φ: the scene is a bag of movable ellipsoids; each remembers center, size, orientation, color.",
    "F(φ,t): how each ellipsoid moves over time (translate δx, rotate δr, fade δγ) — 4D = 3D ellipsoids + a time-deformation field.",
    "Σ=RS SᵀRᵀ: keeps covariance always positive-definite — scale s sets axis lengths, quaternion r sets orientation, training cannot collapse it.",
    "Σ′=JVΣVᵀJᵀ: move the ellipsoid into camera coordinates (V), then squash it with the local linear approximation of perspective (J) into a 2D splat.",
    "α-blending: composite front-to-back along the ray; each splat contributes its color weighted by opacity — T_i is the fraction not yet occluded by what is in front.",
]
y = 1.35
for g in geo:
    ax.text(0.5, y, "•  " + g, fontsize=9, color=WIRE, va="top")
    y -= 0.32

plt.tight_layout(pad=0.6)
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/055-WM-drivedreamer4d/fig-gaussian-flow.png",
            dpi=150, bbox_inches="tight")
print("saved fig-gaussian-flow.png")
