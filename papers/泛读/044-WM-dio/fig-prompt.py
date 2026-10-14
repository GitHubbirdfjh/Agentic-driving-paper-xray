"""DIO: what one prompt changes — same network, three different occupancy outputs.

Top row: source point placement (bird's-eye view toy scene: one car, one tree).
Bottom row: occupancy output o(q, s) at t0 for a row of query points across the scene.
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "serif"
import matplotlib.pyplot as plt
import numpy as np

IN = "#1a6fa3"     # scene / query
OUT = "#9e5c00"   # occupancy output
ALT = "#6a55c2"   # source point prompt
WIRE = "#6e737b"

fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
fig.patch.set_facecolor("white")

# Toy scene layout on a 1-D slice at fixed y,z: 24 cells
N = 24
car = (slice(6, 12),)      # cells 6..11 occupied by car
tree = (slice(17, 20),)    # cells 17..19 occupied by tree

scenes = [
    ("s = empty", None),
    ("s in car", 8),
    ("s in free space", 14),
]
outputs = {
    "s = empty": np.array([1 if (6 <= i < 12 or 17 <= i < 20) else 0 for i in range(N)]),
    "s in car": np.array([1 if 6 <= i < 12 else 0 for i in range(N)]),
    "s in free space": np.zeros(N),
}

for col, (label, s_pos) in enumerate(scenes):
    # --- top: scene slice with source point ---
    ax = axes[0, col]
    ax.set_facecolor("white")
    for i in range(N):
        face = "#dceefb" if 6 <= i < 12 else ("#ddefe3" if 17 <= i < 20 else "white")
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=face, edgecolor="#c9c6bb"))
    ax.text(8.5, 0.5, "car", ha="center", va="center", fontsize=12, color=IN)
    ax.text(18, 0.5, "tree", ha="center", va="center", fontsize=12, color="#0b7a5c")
    if s_pos is not None:
        ax.plot(s_pos + 0.5, 1.25, marker="*", ms=18, color=ALT)
        ax.text(s_pos + 0.5, 1.45, "s", ha="center", fontsize=14, color=ALT)
    ax.set_xlim(0, N); ax.set_ylim(-0.2, 1.7)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#c9c6bb")
    ax.set_title(label, fontsize=14)

    # --- bottom: occupancy output bar ---
    ax = axes[1, col]
    ax.set_facecolor("white")
    o = outputs[label]
    for i in range(N):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1,
                                   facecolor=OUT if o[i] else "white",
                                   edgecolor="#c9c6bb"))
    ax.set_xlim(0, N); ax.set_ylim(-0.15, 1.35)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#c9c6bb")
    ax.set_title(r"$o(q,s)$ at $t_0$", fontsize=14, color=OUT)

axes[0, 0].set_ylabel("scene slice\n(source prompt)", fontsize=12, color=WIRE)
axes[1, 0].set_ylabel("occupancy output", fontsize=12, color=OUT)
fig.suptitle("One network $f_\\theta(X,q,s)$: the source point decides which occupancy is predicted",
             fontsize=15)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/044-WM-dio/fig-prompt.png",
            dpi=150, facecolor="white")
print("saved fig-prompt.png")
