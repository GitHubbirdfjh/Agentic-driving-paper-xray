# -*- coding: utf-8 -*-
"""NavGPT-2 数据效率：表 3 的 Val Unseen SR 随训练数据量变化（论文只给了表，这里画成曲线）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "cm"

BG = "#faf9f5"
INK = "#141413"; WIRE = "#6e737b"
C_DUET = "#6e737b"
C_NAV2 = "#d97757"

fig, ax = plt.subplots(figsize=(7.2, 4.6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

x = [10, 50, 100]
duet = [48.19, 59.90, 63.90]      # Table 3, Val Unseen SR
nav2 = [52.02, 63.30, 67.52]

ax.plot(x, duet, marker="o", ms=7, lw=2, color=C_DUET, label="DUET (LXMERT init, no VLN pretrain)")
ax.plot(x, nav2, marker="s", ms=7, lw=2, color=C_NAV2, label="NavGPT-2 FlanT5-XL (frozen LLM latents)")
for xi, d, n in zip(x, duet, nav2):
    ax.text(xi, d + 1.1, f"{d:.2f}", ha="center", fontsize=10, color=C_DUET)
    ax.text(xi, n + 1.1, f"{n:.2f}", ha="center", fontsize=10, color=C_NAV2)

# the "50% matches 100%" claim
ax.annotate("", xy=(100, 63.90), xytext=(50, 63.30),
            arrowprops=dict(arrowstyle="<->", color=WIRE, lw=1.2, ls="--"))
ax.text(75, 65.6, "gap 0.6", ha="center", fontsize=10, color=WIRE, style="italic")

ax.set_xlabel("R2R training data (%)", fontsize=12, color=INK)
ax.set_ylabel("Val Unseen SR (%)", fontsize=12, color=INK)
ax.set_title("Table 3 redrawn: SR vs. training data scale", fontsize=13, color=INK, weight="bold")
ax.set_xticks([10, 50, 100])
ax.set_ylim(44, 72)
ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#deddd8", fontsize=10.5)
ax.grid(True, color="#deddd8", lw=0.8, alpha=0.7)
ax.tick_params(colors=INK)

plt.tight_layout()
plt.savefig("/home/hatch/workspace/goals/cross-domain-frontier-paper-reading-vla-world-models-vln-multi-agent-driving/files/paper-xray-40/vln/navgpt2-xray/fig-data-efficiency.png",
            dpi=150, facecolor=BG, bbox_inches="tight")
print("saved fig-data-efficiency.png")
