"""LAW: replot of Table 6 - latent prediction horizon vs planning quality.

One figure, one idea: the self-supervision has a sweet spot. 0.5s ahead the
scene barely changes (nothing to learn); 1.5s is best; at 10s the future is
too unpredictable and the world-model term stops helping (worse than the
0.5s point on L2, collision back up). Data: LAW Table 6, nuScenes, avg over 1/2/3s.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"

horizons = ["0.5s", "1.5s", "3.0s", "10.0s"]
l2_avg = [0.61, 0.58, 0.63, 0.72]
coll_avg = [0.30, 0.25, 0.27, 0.43]
x = np.arange(len(horizons))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharex=True)

ax = axes[0]
ax.plot(x, l2_avg, "o-", color="#1a6fa3", ms=9, lw=2)
ax.set_xticks(x); ax.set_xticklabels(horizons)
ax.set_ylabel("avg L2 displacement (m)  lower=better")
ax.set_xlabel("latent prediction horizon")
ax.set_title("planning error vs world-model horizon", fontsize=11)
ax.annotate("sweet spot", xy=(1, 0.58), xytext=(1.9, 0.63),
            arrowprops=dict(arrowstyle="->", color="#0b7a5c"), color="#0b7a5c", weight="bold", fontsize=10)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(x, coll_avg, "s-", color="#b34a06", ms=9, lw=2)
ax.set_xticks(x); ax.set_xticklabels(horizons)
ax.set_ylabel("avg collision rate (%)  lower=better")
ax.set_xlabel("latent prediction horizon")
ax.set_title("collision vs world-model horizon", fontsize=11)
ax.annotate("10s: future too chaotic,\ngains vanish", xy=(3, 0.43), xytext=(1.7, 0.5),
            arrowprops=dict(arrowstyle="->", color="#b34a06"), color="#b34a06", fontsize=10)
ax.grid(alpha=0.3)

fig.suptitle("LAW Table 6 replotted: predicting too near teaches nothing, too far teaches noise",
             fontsize=11, weight="bold", y=1.02)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/051-WM-law/fig-horizon.png", dpi=150)
print("saved fig-horizon.png")
