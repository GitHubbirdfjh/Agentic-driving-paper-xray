"""Self-drawn: GenieDrive Table 3 — end-to-end training helps our continuous
tri-plane representation, but collapses DOME and degrades I2-World.
Bars: forecasting avg mIoU (1s/2s/3s).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

labels = ["DOME", "DOME+E2E", "I2-World", "I2-World+E2E",
          "Ours w/o CR+E2E\n(discrete)", "Ours Full\n(continuous+E2E)"]
miou = [27.10, 0.43, 39.73, 10.09, 35.24, 42.59]
colors = ["#9aa0a8", "#b34a06", "#9aa0a8", "#b34a06", "#b34a06", "#0b7a5c"]

fig, ax = plt.subplots(figsize=(10.5, 5.2))
x = np.arange(len(labels))
bars = ax.bar(x, miou, color=colors, edgecolor="black", linewidth=0.9, width=0.62)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("Forecasting avg mIoU (%) (1s/2s/3s)", fontsize=11)
ax.set_title("E2E training: same recipe, opposite outcomes (GenieDrive Table 3)",
             fontsize=12.5, weight="bold", pad=12)
ax.set_ylim(0, 47)
for b, v in zip(bars, miou):
    ax.text(b.get_x() + b.get_width()/2, v + 0.9, f"{v:.2f}", ha="center", fontsize=10)

ax.annotate("collapse: diffusion loss favors an\noversimplified latent", xy=(1, 0.43), xytext=(2.2, 8),
            fontsize=9, color="#b34a06",
            arrowprops=dict(arrowstyle="->", color="#b34a06", lw=1.5))
ax.annotate("VQ latent also breaks under E2E\n(39.79 -> 35.24)", xy=(4, 35.24), xytext=(2.6, 30),
            fontsize=9, color="#b34a06",
            arrowprops=dict(arrowstyle="->", color="#b34a06", lw=1.5))
ax.annotate("only continuous latent\nimproves: 39.79 -> 42.59", xy=(5, 42.59), xytext=(5, 24),
            fontsize=9, color="#0b7a5c",
            arrowprops=dict(arrowstyle="->", color="#0b7a5c", lw=1.5))
ax.text(0.98, 0.02, "Takeaway: representation alignment via E2E works only when the latent is continuous.\n"
                    "A VQ latent fights the reconstruction/forecasting objective instead of aligning to it.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9.3, style="italic",
        bbox=dict(fc="#f5f4ed", ec="#deddd8", boxstyle="round,pad=0.5"))
plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/042-WM-geniedrive/fig-e2e-ablation.png", dpi=150)
print("saved fig-e2e-ablation.png")
