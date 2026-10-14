"""COME Eq.(1): the visibility-aware mask, worked on a toy pillar.

One pillar covers delta_h x delta_w x D voxels. m_i = 1 where the voxel center was
historically observed. If fewer than eps=0.5 of the pillar's voxels are observed,
M(h,w) = 1 marks the control feature as unreliable; the control feature is then
zeroed: c^n_i <- c^n_i * M_i (kept as binary mask in this toy).
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "serif"
import matplotlib.pyplot as plt
import numpy as np

ALT = "#6a55c2"   # control features
NORM = "#7a6b00"  # mask
WIRE = "#6e737b"

rng = np.random.default_rng(3)
D, PH, PW = 3, 4, 4   # depth, pillar height, pillar width (toy sizes)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
fig.patch.set_facecolor("white")

# --- left: observed voxels m_i (3 depth slices) ---
obs = rng.random((D, PH, PW)) < 0.45
for d in range(D):
    ax = axes[0]
for d in range(D):
    pass
ax = axes[0]
ax.set_facecolor("white")
for d in range(D):
    for i in range(PH):
        for j in range(PW):
            x = j + d * (PW + 1.2)
            face = ALT + "55" if obs[d, i, j] else "white"
            ax.add_patch(plt.Rectangle((x, PH - 1 - i), 0.92, 0.92,
                                       facecolor=face, edgecolor="#c9c6bb"))
    ax.text(d * (PW + 1.2) + PW / 2, -0.9, f"d={d}", ha="center", color=WIRE, fontsize=11)
ax.set_xlim(-0.3, D * (PW + 1.2)); ax.set_ylim(-1.2, PH + 0.4)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_title(r"$m_i(d,u,v)$: 1 = historically observed", fontsize=12)

# --- middle: pillar pooling, eps = 0.5 ---
frac = obs.mean()
M = 1 if frac < 0.5 else 0
ax = axes[1]
ax.set_facecolor("white")
ax.add_patch(plt.Rectangle((0.6, 0.9), 1.8, 1.8, facecolor=(NORM + "55" if M else "white"),
                           edgecolor=NORM, lw=2))
ax.text(1.5, 3.1, f"observed fraction = {frac:.2f}", ha="center", fontsize=12)
ax.text(1.5, 2.55, r"$< \varepsilon = 0.5$", ha="center", fontsize=12, color=NORM)
ax.text(1.5, 1.8, f"$M(h,w) = {M}$", ha="center", fontsize=14, color=NORM, weight="bold")
ax.text(1.5, 0.4, "pillar marked unreliable", ha="center", fontsize=11, color=WIRE)
ax.set_xlim(0, 3); ax.set_ylim(0, 3.6)
ax.axis("off")
ax.set_title(r"Eq.(1): $M(h,w)=\mathbb{I}\left(\frac{1}{D\delta_h\delta_w}\sum m_i < \varepsilon\right)$",
             fontsize=11)

# --- right: control feature before / after masking ---
ax = axes[2]
ax.set_facecolor("white")
before = rng.normal(0, 1, 6)
after = before * (1 - M)
xs = np.arange(6)
ax.bar(xs - 0.22, before, width=0.4, color=ALT, label=r"$c^n_i$ before")
ax.bar(xs + 0.22, after, width=0.4, color=NORM, label=r"$c^n_i \odot M_i$ after")
ax.axhline(0, color=WIRE, lw=0.8)
ax.set_xticks(xs)
ax.set_xlabel("channel", fontsize=11)
ax.set_title("control feature: gated by the mask", fontsize=12)
ax.legend(fontsize=10.5, frameon=False)
for sp in ax.spines.values():
    sp.set_color("#c9c6bb")

fig.suptitle("Visibility-aware masking: do not trust forecasts of what was never seen", fontsize=14)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/045-WM-come/fig-mask.png",
            dpi=150, facecolor="white")
print("saved fig-mask.png")
