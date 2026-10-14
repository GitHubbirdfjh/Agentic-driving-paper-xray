"""AutoDrive-R2 附录 Aha Moment 的 BEV 重演：自反思如何推翻“继续前行”。"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# ---- 左：BEV 轨迹对比 ----
for ax in (ax1, ax2):
    ax.set_xlim(-1, 8); ax.set_ylim(-2.5, 2.5); ax.set_aspect("equal")
    ax.set_xlabel("前向 x (m)"); ax.set_ylabel("横向 y (m)")

# 车道
ax1.axhspan(-1.8, 1.8, color="#eef0f2", zorder=0)
ax1.plot([-1, 8], [0, 0], color="#9aa0a8", ls="--", lw=1.2)
# 物理屏障（路障墙）
ax1.add_patch(plt.Rectangle((4.2, -1.8), 0.5, 3.6, fc="#b34a06", ec="#7a2f04",
                            alpha=0.85, zorder=3))
ax1.text(4.45, 2.05, "物理屏障（路障墙）", ha="center", fontsize=9.5,
         color="#b34a06", weight="bold")
# 自车
ax1.add_patch(plt.Rectangle((-0.5, -0.35), 1.0, 0.7, fc="#1a6fa3", ec="#0d4a73", zorder=4))
ax1.text(0, -0.75, "自车", ha="center", fontsize=9, color="#1a6fa3", weight="bold")

# 前三步的结论：继续前行（会撞墙）
x_bad = np.array([0.45, 0.65, 0.70, 0.70, 0.70, 0.70])
y_bad = np.array([-0.01, -0.02, -0.03, -0.03, -0.03, -0.03])
# 为了画面表现力，把“前冲撞墙”画成示意延长线
x_bad_long = np.linspace(0, 4.4, 30)
ax1.plot(x_bad_long, -0.02 + 0*x_bad_long, color="#b34a06", lw=2.5,
         ls=(0, (4, 2)), label="前三步结论：继续前行（撞墙）", zorder=5)
ax1.scatter([4.2], [-0.02], s=160, marker="X", color="#b34a06", zorder=6, linewidths=1.5)

# 自反思修正：减速停在墙前
x_good = np.array([0.0, 0.45, 0.65, 0.70, 0.70, 0.70])
y_good = np.array([0.0, -0.01, -0.02, -0.03, -0.03, -0.03])
ax1.plot(x_good, y_good + 0.55, color="#0b7a5c", lw=3, marker="o", ms=5,
         label="自反思修正：减速停在墙前", zorder=5)
ax1.legend(fontsize=9, loc="lower right")
ax1.set_title("BEV：Reflection 发现“路根本走不通”", fontsize=11.5, weight="bold")

# ---- 右：速度剖面 ----
t = np.arange(6)
v_bad = np.array([3.03, 2.86, 2.7, 2.6, 2.5, 2.4])   # 继续前行的速度假设
v_good = np.array([3.03, 2.1, 1.2, 0.4, 0.0, 0.0])  # 修正后的减速停车
ax2.plot(t, v_bad, "o--", color="#b34a06", lw=2, label="前三步：缓慢减速（仍会撞墙）")
ax2.plot(t, v_good, "o-", color="#0b7a5c", lw=2.5, label="自反思后：果断刹停")
ax2.axhline(0, color="#9aa0a8", lw=1)
ax2.set_xticks(t); ax2.set_xticklabels([f"{0.5*(i+1):.1f}s" for i in t])
ax2.set_xlabel("未来时刻"); ax2.set_ylabel("速度 (m/s)")
ax2.set_ylim(-0.6, 3.6)
ax2.legend(fontsize=9)
ax2.set_title("速度剖面：修正意味着承认“走不通”", fontsize=11.5, weight="bold")

fig.suptitle("Aha Moment 重演：模型在第 4 步自己推翻了前 3 步（论文附录 Fig.4 场景）",
             fontsize=12.5, weight="bold", y=1.02)
fig.text(0.5, -0.02, "原文的反思独白：“Wait, I just realized that there's a physical barrier directly ahead…"
         " the entire premise of moving forward is flawed.”\n"
         "关键：前三步的“继续前行”在运动学上完全自洽（速度、加速度都对得上），错的是前提假设——路是通的。"
         "只有显式的回查步骤能抓住这种“算得都对但假设错了”的错误。",
         ha="center", fontsize=9.5, color="#3d3d3a")
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/082-Agent-autodrive-r2/fig-aha.png",
            dpi=150, bbox_inches="tight")
print("saved fig-aha.png")
