"""COME: ego-centric vs scene-centric forecasting — where the "small gap" comes from.

Toy top-down scene: static road (gray) + one moving car (blue) + ego vehicle (triangle).
Left column: ego-centric frames — ego moves between t-1 and t+1, so the WHOLE static
background shifts in ego coordinates. Right column: scene-centric frames — every frame
is warped into the current ego frame, so static voxels do not move at all and only the
car changes. The forecasting network then only has to learn the small residual.
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "serif"
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

IN = "#1a6fa3"     # car / dynamic
OUT = "#9e5c00"    # paper numbers
WIRE = "#6e737b"
OP = "#0b7a5c"
STATIC = "#e3e0d6" # static background

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
fig.patch.set_facecolor("white")

rng = np.random.default_rng(7)
# static road pattern: fixed world grid of pillars
world = rng.integers(0, 2, size=(30, 30)).astype(float)

def draw_panel(ax, shift, car_xy, title, subtitle, ego_xy):
    ax.set_facecolor("white")
    # static world shifted by -shift (ego motion appears as background motion)
    big = np.zeros((40, 40))
    big[5:35, 5:35] = world
    sx, sy = 5 - shift[0], 5 - shift[1]
    ax.imshow(big[sy:sy + 30, sx:sx + 30], cmap=matplotlib.colors.ListedColormap(["white", STATIC]),
              origin="lower", extent=(0, 30, 0, 30), interpolation="nearest")
    # moving car (moves in world coordinates too)
    car = mpatches.Rectangle((car_xy[0], car_xy[1]), 5, 2.6, facecolor=IN, edgecolor=IN)
    ax.add_patch(car)
    # ego triangle
    ex, ey = ego_xy
    ax.add_patch(mpatches.RegularPolygon((ex, ey), 3, radius=1.6, orientation=np.pi / 2,
                                         facecolor=OUT, edgecolor=OUT))
    ax.text(ex, ey - 2.6, "ego", ha="center", color=OUT, fontsize=10)
    ax.set_xlim(0, 30); ax.set_ylim(0, 30)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#c9c6bb")
    ax.set_title(title, fontsize=13)
    ax.text(15, -2.2, subtitle, ha="center", fontsize=10.5, color=WIRE)

# ego-centric: ego drives right 6 cells between t-1 and t+1; background shifts left
draw_panel(axes[0, 0], shift=(0, 0), car_xy=(8, 14), title="$t-1$, ego frame",
           subtitle="ego at x=8", ego_xy=(8, 6))
draw_panel(axes[0, 1], shift=(6, 0), car_xy=(10 - 6 + 2, 14), title="$t+1$, ego frame",
           subtitle="ego at x=14: background shifted 6 cells", ego_xy=(8, 6))
draw_panel(axes[0, 2], shift=(0, 0), car_xy=(8, 14), title="what the network must learn",
           subtitle="static world moves 6 cells + car moves", ego_xy=(8, 6))
axes[0, 2].annotate("", xy=(13, 15), xytext=(7, 15),
                    arrowprops=dict(arrowstyle="->", color=LOSS if False else "#b34a06", lw=2))
axes[0, 2].text(10, 17.5, "everything moved", color="#b34a06", fontsize=11, ha="center")

# scene-centric: everything warped to the current (t) ego frame; static world identical
draw_panel(axes[1, 0], shift=(3, 0), car_xy=(8, 14), title="$t-1$ warped to frame $t$",
           subtitle="ego at x=11 in both panels", ego_xy=(11, 6))
draw_panel(axes[1, 1], shift=(3, 0), car_xy=(11, 14), title="$t+1$ warped to frame $t$",
           subtitle="static background: zero change", ego_xy=(11, 6))
ax = axes[1, 2]
ax.set_facecolor("white")
ax.axis("off")
ax.text(0.5, 0.85, "In Occ3D-nuScenes, 92.7% of occupied", ha="center", fontsize=13,
        transform=ax.transAxes)
ax.text(0.5, 0.74, "voxels are static.", ha="center", fontsize=13, transform=ax.transAxes)
ax.text(0.5, 0.58, "Scene-centric: a vanilla U-Net scores", ha="center", fontsize=12.5,
        color=WIRE, transform=ax.transAxes)
ax.text(0.5, 0.47, "39.12 mIoU vs DOME's 27.10", ha="center", fontsize=14, color=OUT,
        transform=ax.transAxes)
ax.text(0.5, 0.36, "(+44%) on 3-s prediction", ha="center", fontsize=12.5, color=WIRE,
        transform=ax.transAxes)
ax.text(0.5, 0.20, "The network only learns the residual:", ha="center", fontsize=12.5,
        color=OP, transform=ax.transAxes)
ax.text(0.5, 0.10, "moving agents, not the whole world.", ha="center", fontsize=12.5,
        color=OP, transform=ax.transAxes)

axes[0, 0].set_ylabel("ego-centric\n(frames in their own ego frames)", fontsize=12, color="#b34a06")
axes[1, 0].set_ylabel("scene-centric\n(all frames in the current ego frame)", fontsize=12, color=OP)
fig.suptitle("Decouple the two motions: ego motion is geometry, scene evolution is learning",
             fontsize=15)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/045-WM-come/fig-scenecentric.png",
            dpi=150, facecolor="white")
print("saved fig-scenecentric.png")
