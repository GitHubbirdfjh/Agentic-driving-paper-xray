# -*- coding: utf-8 -*-
"""Fig B: the PDUS sampling weight w(k) = k / sum_{j=1..k} j = 2/(k+1).
Shows: early updates let the freshest restored video dominate the novel-view
dataset; later the weight decays, reusing accumulated history.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "cm",
})
C_ALT, C_ALT_FILL = "#6a55c2", "#ebe7f8"
C_IN, C_IN_FILL = "#1a6fa3", "#e2eff5"
C_WIRE = "#6e737b"

ks = np.arange(1, 13)
w = ks / (ks * (ks + 1) / 2)  # = 2/(k+1)

fig, ax = plt.subplots(figsize=(10.5, 4.2))
ax.set_xlim(0.4, 12.6); ax.set_ylim(0, 1.12)
ax.bar(ks, w, width=0.62, color=C_ALT_FILL, edgecolor=C_ALT, lw=1.4, zorder=3)
ax.plot(ks, w, "o", color=C_ALT, ms=5, zorder=4)
ax.set_xticks(ks)
ax.set_xlabel("update round $k$", fontsize=12)
ax.set_ylabel("new-video weight $w$", fontsize=12)
ax.set_title(r"$w = \dfrac{k}{\sum_{j=1}^{k} j} = \dfrac{2}{k+1}$: freshness preferred early, history reused later",
             fontsize=13, pad=14)
for k, v in zip(ks, w):
    ax.text(k, v + 0.045, f"{v:.2f}", ha="center", fontsize=9, color=C_ALT)
ax.annotate("k=1: whole novel set is\nthe first restored video",
            xy=(1, 1.0), xytext=(3.4, 0.92),
            arrowprops=dict(arrowstyle="->", color=C_WIRE, lw=1.2),
            fontsize=10, color=C_WIRE)
ax.annotate("k=5: new video gets 1/3,\nhistory keeps 2/3",
            xy=(5, 1/3), xytext=(8.2, 0.62),
            arrowprops=dict(arrowstyle="->", color=C_WIRE, lw=1.2),
            fontsize=10, color=C_WIRE)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-40/world-model-driving/recondreamer-xray/fig-weight.png",
            dpi=150, facecolor="#faf9f5")
print("ok")
