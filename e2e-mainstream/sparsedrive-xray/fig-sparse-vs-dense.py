# 稠密 BEV vs 稀疏实例：为什么"又贵又没必要"
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

rng = np.random.default_rng(7)
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
fig.patch.set_facecolor("white")

# ---------- 左：稠密 BEV ----------
ax = axes[0]
ax.set_facecolor("white")
G = 40
grid = np.zeros((G, G))
# 随机撒一点"被占据"的格子，模拟真实路况的稀疏占用
occ = rng.random((G, G))
grid[occ > 0.965] = 1.0
ax.imshow(grid, cmap=plt.cm.Blues, vmin=0, vmax=1.2, origin="upper",
          extent=(0, 100, 0, 100), interpolation="nearest", alpha=0.85)
# 网格线
for i in range(0, 101, 5):
    ax.plot([i, i], [0, 100], color="#9aa0a8", lw=0.4)
    ax.plot([0, 100], [i, i], color="#9aa0a8", lw=0.4)
ax.plot(50, 6, marker="^", color="#d97757", ms=14, zorder=5)
ax.text(50, 14, "自车", ha="center", fontsize=11, color="#141413")
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("稠密 BEV：整张网格逐格计算", fontsize=13, pad=10)
fig.text(0.245, 0.02, "UniAD 训练显存 50 GB / batch=1\n推理 FLOPs 1709 G",
         ha="center", fontsize=10.5, color="#73726c",
         bbox=dict(boxstyle="round,pad=0.4", fc="#f5f4ed", ec="#deddd8"))

# ---------- 右：稀疏实例 ----------
ax = axes[1]
ax.set_facecolor("white")
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.set_xticks([]); ax.set_yticks([])
# 画几条车道线（map polyline 实例）
for xs, ys in [([10, 30, 50, 70, 90], [30, 32, 35, 40, 48]),
               ([10, 30, 50, 70, 90], [70, 68, 65, 60, 52]),
               ([55, 55], [5, 95])]:
    ax.plot(xs, ys, color="#0b7a5c", lw=2.2, ls=(0, (4, 2)), alpha=0.9)
# 几个检测框（anchor box 实例）
boxes = [(28, 40, 10, 5, 20), (66, 55, 10, 5, -15), (45, 78, 9, 5, 5), (78, 30, 10, 5, 0)]
for (x, y, w, h, ang) in boxes:
    t = matplotlib.transforms.Affine2D().rotate_deg(ang) + ax.transData
    r = mpatches.Rectangle((x - w / 2, y - h / 2), w, h, angle=0,
                           facecolor="#e2eff5", edgecolor="#1a6fa3", lw=1.8,
                           transform=t)
    ax.add_patch(r)
ax.plot(50, 6, marker="^", color="#d97757", ms=14, zorder=5)
ax.text(50, 14, "自车", ha="center", fontsize=11, color="#141413")
ax.annotate("实例 = 特征向量 $F$\n+ 几何锚 $B$",
            xy=(66, 55), xytext=(80, 82), fontsize=10.5, color="#141413",
            arrowprops=dict(arrowstyle="->", color="#6e737b", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.35", fc="#ebe7f8", ec="#6a55c2"))
ax.set_title("稀疏实例：只算 900 个框 + 100 条折线", fontsize=13, pad=10)
fig.text(0.755, 0.02, "SparseDrive-S 训练显存 15.2 GB / batch=6\n推理 FLOPs 192 G",
         ha="center", fontsize=10.5, color="#73726c",
         bbox=dict(boxstyle="round,pad=0.4", fc="#f5f4ed", ec="#deddd8"))

plt.subplots_adjust(bottom=0.18)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("/home/hatch/workspace/your_files/paper-xray-40/e2e-mainstream/sparsedrive-xray/fig-sparse-vs-dense.png",
            dpi=150, bbox_inches="tight")
print("saved fig-sparse-vs-dense.png")
