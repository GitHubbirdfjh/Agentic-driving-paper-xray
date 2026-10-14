# -*- coding: utf-8 -*-
"""NuScenes-QA 的六方向关系：以 ego 前进方向为 0°，逆时针为正，
theta = arccos( ((B1-B2)·Vego) / (|B1-B2| |Vego|) )，按 60° 宽分箱。
左：分箱示意图；右：一个具体数值计算。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.4), gridspec_kw={"width_ratios": [1, 1.1]})
fig.suptitle("NuScenes-QA 的“空间关系”：其实是对 ego 前进方向的 60° 角分箱（式1–2）", fontsize=13, weight="bold", y=0.98)

# ---------- 左：六个扇区 ----------
ax1.set_xlim(-11, 11); ax1.set_ylim(-11, 11); ax1.set_aspect("equal")
ax1.set_title("关系 = 两物体中心连线与 ego 前进方向的夹角 θ", fontsize=11)
# bins: (name, start_deg, end_deg, color)
# 论文角度 a：0°=前向(+y)，逆时针为正。matplotlib Wedge 角度从 +x 轴逆时针起算，
# 映射关系：phi = a + 90（验证：a=0→phi=90°(+y)；a=+90→phi=180°(−x，驾驶员左手边））
bins = [("front\n前", -30, 30, "#c8e6c9"),
        ("front left\n前左", 30, 90, "#bbdefb"),
        ("front right\n前右", -90, -30, "#bbdefb"),
        ("back left\n后左", 90, 150, "#ffe0b2"),
        ("back right\n后右", -150, -90, "#ffe0b2"),
        ("back\n后", 150, 180, "#ffcdd2"),
        ("back\n后", -180, -150, "#ffcdd2")]
for name, a0, a1, c in bins:
    w = mpatches.Wedge((0, 0), 10, a0 + 90, a1 + 90, fc=c, ec="#666", alpha=0.75)
    ax1.add_patch(w)
    mid = np.deg2rad((a0 + a1) / 2 + 90)
    ax1.text(6.4 * np.cos(mid), 6.4 * np.sin(mid), name, ha="center", va="center", fontsize=9.5)
# ego 车
ax1.add_patch(mpatches.Rectangle((-1, -2), 2, 4, fc="#333", ec="k"))
ax1.text(0, 0, "ego", color="white", ha="center", va="center", fontsize=10, weight="bold")
ax1.annotate("", xy=(0, 10.6), xytext=(0, 3.2), arrowprops=dict(arrowstyle="-|>", color="#c62828", lw=2))
ax1.text(0.5, 8.6, "ego 前进方向 = 0°\n逆时针为正", fontsize=9, color="#c62828")
ax1.axis("off")

# ---------- 右：数值实例 ----------
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis("off")
ax2.set_title("实例：bus 在 (4,−1)，car 在 (10,2)，ego 以 (5,0) 前进", fontsize=11)
ax2.text(0.3, 9.0, "问：car 在 bus 的哪个方向？", fontsize=11, weight="bold", ha="left")
steps = [
    "连线向量 B1 − B2 = (10−4, 2−(−1)) = (6, 3)",
    "ego 速度方向 V_ego = (5, 0)",
    "cos θ = (6·5 + 3·0) / (√45 · 5) = 30 / 33.54 ≈ 0.894",
    "θ = arccos(0.894) ≈ 26.6°",
    "26.6° 落在 (−30°, 30°]  →  关系 = front（前）",
]
for i, s in enumerate(steps):
    y = 7.9 - i * 0.95
    bg = "#e8f5e9" if i == 4 else "white"
    ax2.add_patch(mpatches.Rectangle((0.3, y - 0.42), 9.4, 0.8, fc=bg, ec="#bbb", lw=0.8))
    ax2.text(0.55, y, s, fontsize=10, va="center", ha="left",
             weight="bold" if i == 4 else "normal")
ax2.text(5, 1.9, "关键细节：θ 用的是 ego 的前进方向，不是参照物（bus）自己的朝向。\n"
                 "“在 bus 的前左”实际意思是“在 ego 前进方向的左前方、相对于 bus 的位置”。\n"
                 "当 bus 与 ego 朝向不一致时，这个关系会偏离人的直觉——这是纯规则分箱的代价。",
         fontsize=9.5, ha="center", color="#b34a06", linespacing=1.7,
         bbox=dict(fc="#fff8e1", ec="#e0c36a", boxstyle="round,pad=0.4"))

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/090-Agent-nuscenes-qa/fig-relation-bins.png", dpi=150)
print("saved fig-relation-bins.png")
