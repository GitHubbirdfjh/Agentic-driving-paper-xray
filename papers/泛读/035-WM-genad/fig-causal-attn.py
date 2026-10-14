"""Figure: causal vs bidirectional temporal attention masks (T=4 timesteps)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

T = 4
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))

# Panel A: bidirectional (what prior work used)
ax = axes[0]
mask_bi = np.ones((T, T))
im = ax.imshow(mask_bi, cmap="Greys", vmin=0, vmax=1)
for i in range(T):
    for j in range(T):
        ax.text(j, i, "1", ha="center", va="center", fontsize=16,
                color="white" if mask_bi[i, j] > 0.5 else "black")
ax.set_xticks(range(T)); ax.set_yticks(range(T))
ax.set_xticklabels([f"$t{'+'}{k}$" for k in range(T)])
ax.set_yticklabels([f"query $t{'+'}{k}$" for k in range(T)])
ax.set_xlabel("key time", fontsize=13)
ax.set_title("Bidirectional (prior work):\nquery at $t$ sees future frames", fontsize=13, pad=10)

# Panel B: causal (GenAD)
ax = axes[1]
mask_causal = np.tril(np.ones((T, T)))
ax.imshow(mask_causal, cmap="Greys", vmin=0, vmax=1)
for i in range(T):
    for j in range(T):
        txt = "1" if mask_causal[i, j] > 0.5 else "0"
        ax.text(j, i, txt, ha="center", va="center", fontsize=16,
                color="white" if mask_causal[i, j] > 0.5 else "#999999")
ax.set_xticks(range(T)); ax.set_yticks(range(T))
ax.set_xticklabels([f"$t{'+'}{k}$" for k in range(T)])
ax.set_yticklabels([f"query $t{'+'}{k}$" for k in range(T)])
ax.set_xlabel("key time", fontsize=13)
ax.set_title("Causal (GenAD):\nquery at $t$ sees only $\\leq t$", fontsize=13, pad=10)

fig.suptitle("Temporal attention masks over a 4-frame clip", fontsize=15, y=1.02)
fig.tight_layout()
out = "/home/hatch/workspace/goals/cross-domain-frontier-paper-reading-vla-world-models-vln-multi-agent-driving/files/paper-xray-40/world-model-driving/genad-xray/fig-causal-attn.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("saved", out)
