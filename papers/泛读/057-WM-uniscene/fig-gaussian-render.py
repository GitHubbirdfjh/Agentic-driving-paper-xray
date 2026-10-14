"""UniScene 高斯联合渲染：一条相机光线穿过多个高斯，alpha合成得到深度图和语义图。
演示论文公式(3)(4)：D = Σ d_i α'_i Π_{j<i}(1-α'_j)，S = argmax(Σ onehot(s_i) α'_i Π(1-α'_j))。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Ellipse, FancyArrowPatch

FP = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc")
plt.rcParams["font.family"] = FP.get_name()
plt.rcParams["axes.unicode_minus"] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                              gridspec_kw={"width_ratios": [1.25, 1]})
for ax in (ax1, ax2):
    ax.set_aspect("equal")
    ax.axis("off")

# ---------- 左：光线穿过高斯 ----------
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 7)
ax1.set_title("一条相机光线穿过 3 个占据体素的高斯", fontsize=12, fontweight="bold")

# 相机
ax1.add_patch(plt.Polygon([[0.3, 3.2], [0.3, 3.8], [1.0, 3.5]], fc="#333333"))
ax1.text(0.55, 2.9, "相机", fontsize=10, ha="center")
# 光线
ax1.annotate("", xy=(9.5, 3.5), xytext=(1.0, 3.5),
             arrowprops=dict(arrowstyle="-|>", color="#1a6fa3", lw=2))
ax1.text(5.2, 3.75, "光线 r", fontsize=10, color="#1a6fa3")

gaussians = [
    (3.0, 3.5, 0.55, "#e74c3c", "G1\n车 d=8m\nα'=0.9"),
    (5.2, 3.1, 0.75, "#2ecc71", "G2\n树 d=14m\nα'=0.6"),
    (7.4, 3.7, 0.6, "#9b59b6", "G3\n建筑 d=20m\nα'=0.8"),
]
for (x, y, s, col, label) in gaussians:
    e = Ellipse((x, y), 2.2 * s, 1.6 * s, fc=col, alpha=0.45, ec=col, lw=2)
    ax1.add_patch(e)
    ax1.plot([x], [y], "o", color=col, ms=5)
    ax1.text(x, y - 1.35, label, fontsize=9, ha="center",
             bbox=dict(fc="white", ec="#cccccc", boxstyle="round,pad=0.25"))

ax1.text(5, 0.75,
         "每个高斯：μ=体素中心，α=1（占据）/0（空），Σ=(0.125)²I\n"
         "α' 由投影后的2D高斯+3D不透明度决定（公式15）",
         ha="center", fontsize=9.5, color="#555555",
         bbox=dict(fc="white", ec="#cccccc", boxstyle="round,pad=0.35"))

# ---------- 右：合成计算 ----------
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 7)
ax2.set_title("该像素的深度 / 语义合成（前向alpha混合）", fontsize=12, fontweight="bold")

# 权重条
ax2.text(0.5, 6.2, "各高斯的有效权重 w_i = α'_i · Π(1-α'_j)，j<i（前面的先挡）",
         fontsize=10, ha="left", color="#333333")
items = [
    ("G1(车)", 0.9, 8.0, "#e74c3c", 5.35),
    ("G2(树)", 0.6 * (1 - 0.9), 14.0, "#2ecc71", 4.45),
    ("G3(建筑)", 0.8 * (1 - 0.9) * (1 - 0.6), 20.0, "#9b59b6", 3.55),
]
for name, w, d, col, y in items:
    ax2.add_patch(plt.Rectangle((0.5, y - 0.35), w * 7, 0.7, fc=col, alpha=0.7, ec=col))
    ax2.text(0.5 + w * 7 + 0.2, y, f"{name}: w={w:.3f}, d={d:.0f}m",
             fontsize=10, va="center")

ax2.text(0.5, 2.35,
         "深度 D = Σ w_i·d_i\n"
         "      = 0.900×8 + 0.060×14 + 0.032×20 = 8.68m\n"
         "      → 最近的车主导，符合直觉",
         fontsize=11, ha="left", va="top",
         bbox=dict(fc="#f8ecd2", ec="#9e5c00", boxstyle="round,pad=0.4"))
ax2.text(0.5, 1.05,
         "语义 S = argmax(Σ w_i·onehot(s_i))\n"
         "      = argmax(0.9·[车]+0.06·[树]+0.032·[建筑]) = 车",
         fontsize=11, ha="left", va="top",
         bbox=dict(fc="#ebe7f8", ec="#6a55c2", boxstyle="round,pad=0.4"))

fig.suptitle("高斯联合渲染：occupancy 体素 → 2D 语义/深度条件图（重构论文公式 3/4）",
             fontsize=13, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/057-WM-uniscene/fig-gaussian-render.png",
            dpi=150, bbox_inches="tight")
print("saved fig-gaussian-render.png")
