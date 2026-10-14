"""Self-drawn figure for 028 xray: LeapAD ablation curves (Tables 4 & 5).

Left: Driving Score vs number of few-shot examples (Table 4).
Right: Driving Score vs memory-bank size on a log scale (Table 5).
Both from the Town05 Short benchmark, verified against the paper text.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/hatch/workspace/your_files/paper-xray-100/papers/028-VLA-leapad/fig-ablation.png"

# Table 4: shots -> DS / RC / IS (verified from paper)
shots = np.array([0, 1, 2, 3])
ds_shots = np.array([51.77, 64.95, 71.20, 72.33])

# Table 5: memory size -> DS / RC / IS (verified from paper)
mem = np.array([0, 90, 900, 9000])
ds_mem = np.array([51.77, 60.72, 66.68, 72.33])

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

ax = axes[0]
ax.plot(shots, ds_shots, marker="o", color="#1a6fa3", linewidth=2, markersize=7)
for s, d in zip(shots, ds_shots):
    ax.annotate(f"{d:.2f}", xy=(s, d), xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=9)
deltas = np.diff(ds_shots)
for i, dd in enumerate(deltas, start=1):
    ax.annotate(f"+{dd:.1f}", xy=(i - 0.5, (ds_shots[i] + ds_shots[i-1]) / 2),
                fontsize=8, color="#666666", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="#f0f0f0", ec="none"))
ax.set_xticks(shots)
ax.set_xlabel("number of few-shot examples")
ax.set_ylabel("Driving Score (%)")
ax.set_title("(a) DS vs few-shot count  (Table 4)", fontsize=11)
ax.grid(alpha=0.25)

ax = axes[1]
ax.plot(mem, ds_mem, marker="s", color="#0b7a5c", linewidth=2, markersize=7)
for m, d in zip(mem, ds_mem):
    ax.annotate(f"{d:.2f}", xy=(m, d), xytext=(8, 6), textcoords="offset points",
                ha="left", fontsize=9)
ax.set_xscale("symlog", linthresh=10)
ax.set_xticks([0, 90, 900, 9000])
ax.set_xticklabels(["0", "90", "900", "9000"])
ax.set_xlabel("memory bank size (log scale)")
ax.set_ylabel("Driving Score (%)")
ax.set_title("(b) DS vs memory size  (Table 5)", fontsize=11)
ax.grid(alpha=0.25)

fig.suptitle("LeapAD ablations: more experience -> better closed-loop driving",
             fontsize=12, fontweight="bold", y=1.02)
fig.text(0.5, -0.04,
         "Town05 Short. Left: 0->1 shot gives +13.2 (diminishing returns after). "
         "Right: ~+6 DS points per 10x memory growth; 90 samples alone add +9.0.",
         ha="center", fontsize=8.5, color="#555555")
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved", OUT)
