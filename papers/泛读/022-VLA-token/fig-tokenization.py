# TOKEN: object-centric vs scene-centric tokenization.
# Left: dense ViT-style grid tokens -- one object's information is spread across
# many tokens ("entangled"). Right: TOKEN's object-centric tokens -- one object
# gets one token (track token + motion token concatenated, through an adapter).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "figure.facecolor": "#faf9f5",
    "axes.facecolor": "#faf9f5",
    "font.size": 12,
    "mathtext.fontset": "cm",
})

IN, OUT, OP, ALT, LOSS, WIRE = "#1a6fa3", "#9e5c00", "#0b7a5c", "#6a55c2", "#b34a06", "#6e737b"

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
fig.suptitle("What the LLM sees: scene-centric (dense) vs object-centric (TOKEN) tokenization",
             fontsize=13, y=0.98)

# ---------- Left: scene-centric grid ----------
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
ax.axis("off")
# road
ax.add_patch(mpatches.Rectangle((0, 3), 10, 4, facecolor="#e8e6df", edgecolor=WIRE, lw=1))
ax.plot([0, 10], [5, 5], color=WIRE, lw=1, linestyle=(0, (4, 3)))
# objects: car, pedestrian, cone
ax.add_patch(mpatches.Rectangle((5.6, 5.6), 2.2, 1.1, facecolor=IN, alpha=0.85))
ax.text(6.7, 6.15, "car", ha="center", va="center", fontsize=11, color="white")
ax.add_patch(mpatches.Circle((2.2, 6.3), 0.32, facecolor=OUT, alpha=0.9))
ax.text(2.2, 6.3, "p", ha="center", va="center", fontsize=10, color="white")
ax.add_patch(mpatches.Polygon([[8.3, 5.5], [8.7, 5.5], [8.5, 6.1]], closed=True,
                              facecolor=LOSS, alpha=0.9))
# 8x8 token grid
rng = np.random.default_rng(3)
for i in range(8):
    for j in range(8):
        x0, y0 = i * 1.25, j * 1.25
        # cells overlapping the car get highlighted: its info is spread across them
        overlap_car = (x0 < 7.8 + 0.01) and (x0 + 1.25 > 5.6) and \
                      (y0 < 6.7 + 0.01) and (y0 + 1.25 > 5.6)
        if overlap_car:
            ax.add_patch(mpatches.Rectangle((x0, y0), 1.25, 1.25,
                                            facecolor=OUT, alpha=0.45,
                                            edgecolor=OUT, lw=1.4))
        else:
            ax.add_patch(mpatches.Rectangle((x0, y0), 1.25, 1.25,
                                            fill=False, edgecolor=WIRE, lw=0.5, alpha=0.6))
ax.text(5, 0.4, "one car's bbox + class + motion is smeared over ~12 grid tokens",
        ha="center", fontsize=11, color=OUT)
ax.set_title("Scene-centric (Video-LLaMA / VILA-1.5 style)\n"
             "dense grid tokens: unstructured, entangled", fontsize=12, pad=10)

# ---------- Right: object-centric ----------
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

token_specs = [
    ("car", r"$z^{track}$(256) + $z^{motion}$(256)", IN),
    ("pedestrian", r"$z^{track}$(256) + $z^{motion}$(256)", OUT),
    ("traffic cone", r"$z^{map}$(256)", OP),
]
for k, (name, dims, color) in enumerate(token_specs):
    y = 7.6 - k * 2.2
    ax.add_patch(mpatches.FancyBboxPatch((1.2, y - 0.75), 4.6, 1.5,
                                         boxstyle="round,pad=0.08,rounding_size=0.35",
                                         facecolor=color, alpha=0.16,
                                         edgecolor=color, lw=1.8))
    ax.text(3.5, y + 0.28, name, ha="center", va="center", fontsize=12,
            color=color, weight="bold")
    ax.text(3.5, y - 0.32, dims, ha="center", va="center", fontsize=10.5, color="#141413")
    # arrow through adapter to LLM
    ax.annotate("", xy=(8.6, y), xytext=(6.1, y),
                arrowprops=dict(arrowstyle="-|>", color=WIRE, lw=1.4,
                                shrinkA=2, shrinkB=2))
    ax.text(7.35, y + 0.32, "MLP adapter", ha="center", fontsize=10, color=WIRE)

ax.add_patch(mpatches.FancyBboxPatch((8.35, 1.1), 1.1, 7.4,
                                     boxstyle="round,pad=0.05,rounding_size=0.3",
                                     facecolor="#e2eff5", edgecolor=IN, lw=1.8))
ax.text(8.9, 4.8, "LLaMA\ntokens", ha="center", va="center", fontsize=11, color=IN)
ax.text(5, 0.4, "1 object = 1 token: boundaries of tokens match units of reasoning",
        ha="center", fontsize=11, color=ALT)
ax.set_title("Object-centric (TOKEN)\none token per object, from frozen PARA-Drive queries",
             fontsize=12, pad=10)

plt.tight_layout(rect=[0, 0, 1, 0.92])
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/022-VLA-token/fig-tokenization.png"
plt.savefig(out, dpi=150)
print("saved", out)
