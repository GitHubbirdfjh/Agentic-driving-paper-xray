# 表 3 效率数字可视化：训练时间与推理 FPS
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
fig.patch.set_facecolor("white")

methods = ["UniAD", "SparseDrive-S", "SparseDrive-B"]
colors = ["#9aa0a8", "#1a6fa3", "#0b7a5c"]

# 左：训练时间（小时）
ax = axes[0]
times = [144, 20, 30]
bars = ax.bar(methods, times, color=colors, width=0.55, edgecolor="white")
for b, t in zip(bars, times):
    ax.text(b.get_x() + b.get_width() / 2, t + 3, f"{t} h", ha="center", fontsize=12)
ax.set_ylim(0, 170)
ax.set_title("训练总时长（8 卡）", fontsize=13, pad=10)
ax.set_ylabel("小时", fontsize=11)
ax.text(0.5, -0.32, "UniAD 在 8×A100 上测得，SparseDrive 在 8×4090 上测得\n——跨卡对比，倍数里有硬件差异的水分",
        ha="center", va="top", fontsize=10, color="#73726c", transform=ax.transAxes)
ax.annotate("7.2×", xy=(1, 20), xytext=(1.55, 80), fontsize=13, color="#1a6fa3",
            arrowprops=dict(arrowstyle="->", color="#1a6fa3", lw=1.4))

# 右：推理 FPS
ax = axes[1]
fps = [1.8, 9.0, 7.3]
bars = ax.bar(methods, fps, color=colors, width=0.55, edgecolor="white")
for b, f in zip(bars, fps):
    ax.text(b.get_x() + b.get_width() / 2, f + 0.25, f"{f}", ha="center", fontsize=12)
ax.set_ylim(0, 11)
ax.set_title("推理速度（单卡 batch=1）", fontsize=13, pad=10)
ax.set_ylabel("FPS", fontsize=11)
ax.text(0.5, -0.32, "UniAD 在单张 A100 上测得，SparseDrive 在单张 4090 上测得",
        ha="center", va="top", fontsize=10, color="#73726c", transform=ax.transAxes)
ax.annotate("5.0×", xy=(1, 9.0), xytext=(1.55, 6.5), fontsize=13, color="#1a6fa3",
            arrowprops=dict(arrowstyle="->", color="#1a6fa3", lw=1.4))

plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-40/e2e-mainstream/sparsedrive-xray/fig-efficiency.png",
            dpi=150, bbox_inches="tight")
print("saved fig-efficiency.png")
