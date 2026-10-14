"""Self-drawn: k-disc trajectory tokenization in NoRD (schematic).

Top:    a 4 s trajectory is cut into eight 0.5 s segments; each segment is a
        sequence of 6 vehicle-contour frames (4 corners x xy), matched to the
        nearest of 2048 k-means centroids -> one TRAJ_XXXX token.
Bottom: decode chains the segments back: each token's local contour is
        rotated/translated by the previous segment's end pose, giving
        40 waypoints at 10 Hz.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, axes = plt.subplots(2, 1, figsize=(11, 6.4))
fig.patch.set_facecolor("white")

# ---------------- top: encode ----------------
ax = axes[0]
ax.set_xlim(0, 11); ax.set_ylim(0, 3.6); ax.axis("off")
ax.set_title("Encode: 4 s trajectory -> eight 0.5 s contour segments -> 8 discrete tokens",
             fontsize=12, pad=8)

t = np.linspace(0, 1, 41)
xs = 0.6 + 9.0 * t
ys = 1.7 + 0.9 * np.sin(2.2 * np.pi * t) * t
ax.plot(xs, ys, color="#1a6fa3", lw=2.5)

for k in range(9):
    i = k * 5
    ax.plot([xs[i], xs[i]], [ys[i] - 0.28, ys[i] + 0.28], color="#6e737b", lw=1.2, ls="--")
for i in [2, 12, 22, 32]:
    ang = np.arctan2(ys[i+1]-ys[i-1], xs[i+1]-xs[i-1])
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    corners = np.array([[2.4, 1.0], [2.4, -1.0], [-2.4, -1.0], [-2.4, 1.0]]) * 0.28
    box = (R @ corners.T).T + np.array([xs[i], ys[i]])
    ax.add_patch(patches.Polygon(box, closed=True, fc="#e2eff5", ec="#1a6fa3", lw=1.2))
ax.text(5.1, 0.55, "cut into 8 segments of 0.5 s (5 steps at 10 Hz + boundary frame)",
        fontsize=10, style="italic", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f4ed", ec="none"))

ax.annotate("", xy=(5.1, 2.05), xytext=(5.1, 1.75),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
ax.add_patch(patches.FancyBboxPatch((3.3, 2.05), 3.6, 0.75, boxstyle="round,pad=0.05",
                                    fc="#ebe7f8", ec="#6a55c2", lw=1.4))
ax.text(5.1, 2.52, "codebook: 2048 centroids", ha="center", fontsize=10,
        fontweight="bold", color="#3a2f7a")
ax.text(5.1, 2.22, "each = 6 frames x 4 corners x (x, y); nearest by contour distance",
        ha="center", fontsize=9, color="#3a2f7a")
ax.annotate("", xy=(8.6, 2.4), xytext=(6.9, 2.4),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
ax.text(9.5, 2.4, "TRAJ_0242  TRAJ_0150\nTRAJ_1172  ...  (8 tokens)",
        fontsize=10, family="monospace", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#dcf2e8", ec="#0b7a5c", lw=1.2))

# ---------------- bottom: decode ----------------
ax = axes[1]
ax.set_xlim(0, 11); ax.set_ylim(0, 3.0); ax.axis("off")
ax.set_title("Decode: each token's local contour is posed by the previous segment's end pose, then chained",
             fontsize=12, pad=8)

segs = [
    dict(x0=0.7, y0=1.5, ang=8,  col="#1a6fa3", tok="TRAJ_0242"),
    dict(x0=3.6, y0=1.75, ang=-14, col="#0b7a5c", tok="TRAJ_0150"),
    dict(x0=6.4, y0=1.35, ang=20, col="#9e5c00", tok="TRAJ_1172"),
]
for s in segs:
    ang = np.deg2rad(s["ang"]); c, sn = np.cos(ang), np.sin(ang)
    R = np.array([[c, -sn], [sn, c]])
    lt = np.linspace(0, 1, 6)
    local = np.stack([2.6 * lt, 0.5 * np.sin(2 * lt)], axis=1)
    glob = (R @ local.T).T + np.array([s["x0"], s["y0"]])
    ax.plot(glob[:, 0], glob[:, 1], color=s["col"], lw=3)
    ax.scatter(glob[:, 0], glob[:, 1], color=s["col"], s=26, zorder=3)
    corners = np.array([[2.4, 1.0], [2.4, -1.0], [-2.4, -1.0], [-2.4, 1.0]]) * 0.22
    box = (R @ corners.T).T + glob[-1]
    ax.add_patch(patches.Polygon(box, closed=True, fc="white", ec=s["col"], lw=1.6))
    ax.text(s["x0"] + 0.35, s["y0"] - 0.62, s["tok"], fontsize=9, family="monospace",
            color=s["col"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=s["col"], lw=1))

ax.annotate("", xy=(3.55, 1.78), xytext=(3.05, 1.72),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5, ls="--"))
ax.text(3.3, 2.25, "end pose\nbecomes\nnext frame", fontsize=8, ha="center", style="italic",
        bbox=dict(boxstyle="round,pad=0.25", fc="#f5f4ed", ec="none"))
ax.annotate("", xy=(6.35, 1.38), xytext=(5.9, 1.32),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5, ls="--"))

ax.text(9.3, 1.5, "8 tokens x 5 steps\n= 40 waypoints\n[x, y, heading] at 10 Hz",
        fontsize=10, va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f8ecd2", ec="#9e5c00", lw=1.2))
ax.text(5.5, 0.25, "boundary frames are written once (no averaging); heading comes from contour geometry",
        fontsize=9, style="italic", ha="center", color="#6e737b")

plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/030-VLA-nord/fig-kdisc.png",
            dpi=150, bbox_inches="tight")
print("saved fig-kdisc.png")
