"""Fig: channel matching matrix M as a translation table.
Ego has C1=3 channels, neighbor has C2=4 channels.
M in R^{3x4}: each row is a soft selection over neighbor channels.
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
import numpy as np

# role colors (light theme, from skill 5.1)
IN, OUT, OP, WIRE = "#1a6fa3", "#9e5c00", "#0b7a5c", "#6e737b"
IN_FILL, OUT_FILL, OP_FILL = "#e2eff5", "#f8ecd2", "#dcf2e8"

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.35, 1]})
fig.patch.set_facecolor("#faf9f5")
for ax in axes:
    ax.set_facecolor("#faf9f5")

# --- left: neighbor channels (input) ---
ax = axes[0]
M_vals = np.array([[0.04, 0.82, 0.14, 0.00],
                   [0.70, 0.05, 0.20, 0.05],
                   [0.10, 0.10, 0.15, 0.65]])
for j in range(4):
    rect = plt.Rectangle((0.1, 0.12 + j * 0.21), 0.8, 0.17, facecolor=IN_FILL,
                         edgecolor=IN, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(0.5, 0.205 + j * 0.21, f"neb ch{j+1}", ha="center", va="center",
            fontsize=12, color=IN)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("neighbor feature $F_{neb}$\n4 channels", fontsize=12, color=WIRE)

# --- middle: M matrix heatmap ---
ax = axes[1]
im = ax.imshow(M_vals, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(3):
    for j in range(4):
        ax.text(j, i, f"{M_vals[i, j]:.2f}", ha="center", va="center",
                fontsize=12, color="white" if M_vals[i, j] > 0.5 else "black")
ax.set_xticks(range(4)); ax.set_xticklabels([f"neb ch{j+1}" for j in range(4)], fontsize=10, color=IN)
ax.set_yticks(range(3)); ax.set_yticklabels([f"ego ch{i+1}" for i in range(3)], fontsize=10, color=OUT)
ax.set_xlabel("softmax along each row: each ego channel picks\nfrom neighbor channels", fontsize=10, color=WIRE)
ax.set_title("$M = \\mathrm{softmax}(QK^\\top/\\sqrt{d_k}) \\in \\mathbb{R}^{3\\times 4}$",
             fontsize=13)
for spine in ax.spines.values():
    spine.set_color(OP); spine.set_linewidth(1.5)

# --- right: reorganized ego-aligned channels ---
ax = axes[2]
labels = ["$0.82\\cdot$c2 + $0.14\\cdot$c3 + $0.04\\cdot$c1$",
          "$0.70\\cdot$c1 + $0.20\\cdot$c3 + \\dots$",
          "$0.65\\cdot$c4 + $0.15\\cdot$c3 + \\dots$"]
for i in range(3):
    rect = plt.Rectangle((0.05, 0.12 + i * 0.21), 0.9, 0.17, facecolor=OUT_FILL,
                         edgecolor=OUT, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(0.5, 0.205 + i * 0.21, labels[i], ha="center", va="center", fontsize=11, color=OUT)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("reorganized $F^r_{neb} = \\mathrm{LN}(M F_{neb})$\n3 channels, ego's coordinates", fontsize=12, color=WIRE)

# arrows between panels
for x0, x1 in [(0.30, 0.36), (0.63, 0.70)]:
    fig.patches.append(plt.Rectangle((0, 0), 0, 0))  # placeholder no-op
fig.text(0.315, 0.5, "→", fontsize=28, color=OP, ha="center", va="center",
         transform=fig.transFigure)
fig.text(0.655, 0.5, "→", fontsize=28, color=OP, ha="center", va="center",
         transform=fig.transFigure)

plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/085-Agent-polyinter/fig-channel-matching.png",
            dpi=150, facecolor="#faf9f5")
print("saved")
