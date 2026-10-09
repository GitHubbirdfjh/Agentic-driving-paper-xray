# VLP: ALP (agent-wise) vs SLP (sample-wise) contrastive alignment.
# Each panel shows an idealized similarity matrix S_pred. The training target is
# the identity matrix: every BEV-side feature must be closest to its own
# language-side expectation feature. The loss is the mean of cross-entropy
# taken along rows and along columns.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "figure.facecolor": "#faf9f5",
    "axes.facecolor": "#faf9f5",
    "font.size": 12,
    "mathtext.fontset": "cm",
})

IN, OUT, OP, ALT, LOSS, WIRE = "#1a6fa3", "#9e5c00", "#0b7a5c", "#6a55c2", "#b34a06", "#6e737b"

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
fig.suptitle("VLP: two contrastive alignments (training only, LM frozen, discarded at inference)",
             fontsize=13, y=0.98)

# ---------- Left: ALP, agent-wise ----------
agents = ["ego-car", "truck", "pedestrian", "divider", "crosswalk"]
n = len(agents)
rng = np.random.default_rng(7)
S_alp = rng.normal(0, 0.8, size=(n, n))
np.fill_diagonal(S_alp, 0)
diag = np.array([6.5, 6.2, 2.0, 6.8, 6.4])
np.fill_diagonal(S_alp, diag)
# one confused row: pedestrian BEV feature is closer to truck's description
S_alp[2, 1] = 4.5

ax = axes[0]
im = ax.imshow(S_alp, cmap="Blues", vmin=-2, vmax=7)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{S_alp[i, j]:.1f}", ha="center", va="center",
                fontsize=11, color="white" if S_alp[i, j] > 3.5 else "#141413")
# diagonal = target (identity)
for i in range(n):
    ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                           edgecolor=ALT, lw=2.6))
# confused cell: row-softmax would put most mass on the wrong column
ax.add_patch(Rectangle((1 - 0.5, 2 - 0.5), 1, 1, fill=False,
                       edgecolor=LOSS, lw=2.6, linestyle=(0, (4, 2))))
ax.text(1, 2.62, "row max off-diagonal", ha="center", va="top",
        fontsize=10, color=LOSS)
ax.set_xticks(range(n), agents, rotation=18, ha="right", fontsize=11)
ax.set_yticks(range(n), agents, fontsize=11)
ax.set_xlabel("expectation features $A_{exp}$ (from frozen LM + $F_{bev}$)", fontsize=11)
ax.set_ylabel("BEV agent features $A_{bev}$", fontsize=11)
ax.set_title("ALP: agent-wise,  $S_{pred} \\in \\mathbb{R}^{N_B \\times N_B}$\n"
             "$L_{alp}=(CE_{rows}+CE_{cols})/2$,  target = identity",
             fontsize=12, pad=10)

# ---------- Right: SLP, sample-wise ----------
samples = ["s1", "s2", "s3", "s4"]
m = len(samples)
S_slp = rng.normal(0, 0.8, size=(m, m))
np.fill_diagonal(S_slp, 0)
np.fill_diagonal(S_slp, [6.1, 6.6, 6.3, 6.9])

ax = axes[1]
im2 = ax.imshow(S_slp, cmap="Greens", vmin=-2, vmax=7)
for i in range(m):
    for j in range(m):
        ax.text(j, i, f"{S_slp[i, j]:.1f}", ha="center", va="center",
                fontsize=11, color="white" if S_slp[i, j] > 3.5 else "#141413")
for i in range(m):
    ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                           edgecolor=OP, lw=2.6))
ax.set_xticks(range(m), ["$E_{prompt}$ " + s for s in samples], fontsize=11)
ax.set_yticks(range(m), ["$E^q_{feat}$ " + s for s in samples], fontsize=11)
ax.set_xlabel("text planning features (from frozen LM + $F_{ego}$)", fontsize=11)
ax.set_ylabel("ego-query features", fontsize=11)
ax.set_title("SLP: sample-wise,  $S^{slp}_{pred} \\in \\mathbb{R}^{B \\times B}$\n"
             "$L_{slp}=(CE_{rows}+CE_{cols})/2$,  target = identity",
             fontsize=12, pad=10)

fig.text(0.5, 0.02,
         "Left: each agent's BEV crop must match its own GT-filled description. "
         "Right: each sample's ego-query must match its own planning description.",
         ha="center", fontsize=11, color=WIRE)
plt.tight_layout(rect=[0, 0.05, 1, 0.93])
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/021-VLA-vlp/fig-alignment.png"
plt.savefig(out, dpi=150)
print("saved", out)
