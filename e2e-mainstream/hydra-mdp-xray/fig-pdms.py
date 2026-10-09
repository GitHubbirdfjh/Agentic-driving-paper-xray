import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# ---- Panel A: PDMS multiplier collapse ----
# PDMS = NC * DAC * (5*TTC + 5*EP + 2*C) / 12
ax = axes[0]
scenarios = ["clean run\nNC=1, DAC=1\nTTC=1, EP=0.8, C=1", "collision\nNC=0, DAC=1\nTTC=0, EP=0.8, C=1"]
pdms = [1*1*(5*1+5*0.8+2*1)/12, 0*1*(5*0+5*0.8+2*1)/12]
colors = ["#2a7f62", "#c0392b"]
bars = ax.bar([0, 1], pdms, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
ax.set_xticks([0, 1])
ax.set_xticklabels(scenarios, fontsize=9)
ax.set_ylabel("PDMS", fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title("PDMS = NC x DAC x (5*TTC + 5*EP + 2*C)/12\none zero multiplier kills everything", fontsize=11, fontweight="bold")
for b, v in zip(bars, pdms):
    ax.text(b.get_x() + b.get_width()/2, v + 0.03, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# ---- Panel B: inference weight ranges (Eq. 11) ----
ax = axes[1]
labels = ["w1 (imitation)", "w2 (NC)", "w3 (DAC)", "w4 (TTC/C/EP)"]
lo = [0.01, 0.1, 0.1, 1.0]
hi = [0.1, 1.0, 1.0, 10.0]
y = np.arange(len(labels))
widths = [h - l for l, h in zip(lo, hi)]
bars = ax.barh(y, widths, left=lo, color=["#7f8c9b", "#2a7f62", "#2a7f62", "#b3541e"],
               edgecolor="black", linewidth=1.2)
ax.set_xscale("log")
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("grid-searched weight range (log scale)", fontsize=10)
ax.set_title("Eq.(11): rule heads outweigh imitation by ~100x\nf~(T,O) = -(w1 log S_im + w2 log S_NC + w3 log S_DAC + w4 log(5S_TTC+2S_C+5S_EP))",
             fontsize=10, fontweight="bold")
ax.grid(axis="x", alpha=0.3, which="both")
for yy, l, h in zip(y, lo, hi):
    ax.text(h * 1.15, yy, f"[{l}, {h}]", va="center", fontsize=9)

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-40/e2e-mainstream/hydra-mdp-xray/fig-pdms.png", dpi=150)
print("saved")
