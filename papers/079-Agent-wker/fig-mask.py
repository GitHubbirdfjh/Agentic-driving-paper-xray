# -*- coding: utf-8 -*-
"""079-WKER Fig2: appendix Table 12 -- token redundancy experiment (mask sky-region tokens)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

labels = ["Exp1\nblind", "Exp2\nmask 0%", "Exp3\nmask 10%", "Exp4\nmask 30%", "Exp5\nmask 50%"]
mae = np.array([15.6, 4.33, 4.43, 3.8, 4.63])        # MEA, lower is better
acc = np.array([25.14, 52.71, 51.08, 52.88, 46.24])   # ACC, higher is better
x = np.arange(len(labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
fig.suptitle("Appendix Table 12: masking sky-region visual tokens barely hurts until 50%", fontsize=12, fontweight="bold")

bars1 = ax1.bar(x, mae, color=["#f1bd9c", "#b5d8ec", "#b5d8ec", "#b5d8ec", "#f1bd9c"])
ax1.set_xticks(x); ax1.set_xticklabels(labels)
ax1.set_ylabel("MAE (lower = better)")
ax1.set_title("masking 10%-30% leaves MAE flat", fontsize=10)
ax1.axhline(mae[1], ls="--", color="#6e737b", lw=1)
for b, v in zip(bars1, mae):
    ax1.text(b.get_x()+b.get_width()/2, v+0.25, f"{v}", ha="center", fontsize=9)

bars2 = ax2.bar(x, acc, color=["#f1bd9c", "#b5d8ec", "#b5d8ec", "#b5d8ec", "#f1bd9c"])
ax2.set_xticks(x); ax2.set_xticklabels(labels)
ax2.set_ylabel("ACC (higher = better)")
ax2.set_title("only 50% masking drops ACC: most visual tokens are instruction-irrelevant", fontsize=10)
ax2.axhline(acc[1], ls="--", color="#6e737b", lw=1)
for b, v in zip(bars2, acc):
    ax2.text(b.get_x()+b.get_width()/2, v+0.8, f"{v}", ha="center", fontsize=9)

fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/079-Agent-wker/fig-mask.png",
            dpi=160, bbox_inches="tight", facecolor="white")
print("saved fig-mask.png")
