# -*- coding: utf-8 -*-
"""Overtaking SQL（Algorithm 11）在一个具体三帧场景上的走查：
ego 坐标系：x 垂直车头方向，y 沿车头方向。l = x^2+y^2（论文式，未开根）。
注：论文印刷的条件行有笔误（Mi-1[d][0]<0 且 >0 不可能同时成立），
此处按“y 由负变正 = 从车后跑到车前”的本意重建逻辑，并在正文说明。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.2), gridspec_kw={"width_ratios": [1, 1.15]})
fig.suptitle("走查 Overtaking SQL：三辆车，谁算“超车”？（dis 取 30，论文未公布阈值）", fontsize=13, weight="bold", y=0.98)

# ---------- 左：BEV 轨迹 ----------
ax1.set_xlim(-8, 8); ax1.set_ylim(-14, 20); ax1.set_aspect("equal")
ax1.set_xlabel("x（垂直车头方向）"); ax1.set_ylabel("y（沿车头方向，+ 为前方）")
ax1.set_title("BEV：ego 在原点，车头朝 +y", fontsize=11)
ax1.axhline(0, color="#999", lw=0.8, ls="--"); ax1.axvline(0, color="#999", lw=0.8, ls="--")
ax1.add_patch(mpatches.Rectangle((-1, -2.2), 2, 4.4, fc="#333", ec="k"))
ax1.text(0, 0, "ego", color="white", ha="center", va="center", fontsize=10, weight="bold")

cars = {
    "A": {"pos": [(-2.0, -8.0), (-2.0, 3.0), (-2.0, 14.0)], "c": "#1a6fa3", "name": "车 A（超车）"},
    "B": {"pos": [(2.0, 16.0), (2.0, 11.0), (2.0, 6.0)], "c": "#e69f00", "name": "车 B（接近中）"},
    "C": {"pos": [(3.2, 5.0), (3.2, 5.0), (3.2, 5.0)], "c": "#999999", "name": "车 C（停放）"},
}
for k, car in cars.items():
    xs = [p[0] for p in car["pos"]]; ys = [p[1] for p in car["pos"]]
    ax1.plot(xs, ys, color=car["c"], lw=2, marker="o", ms=6)
    for (x, y), f in zip(car["pos"], ["i-1", "i", "i+1"]):
        ax1.text(x + 0.45, y + 0.3, f, fontsize=8, color=car["c"])
ax1.legend([mpatches.Patch(color=cars[k]["c"]) for k in "ABC"],
           [cars[k]["name"] for k in "ABC"], loc="upper left", fontsize=9)
ax1.annotate("车 A：y 从 -8 → +3\n穿过 ego 横轴", xy=(-2, 3), xytext=(-7.5, 8),
             fontsize=9, color="#1a6fa3",
             arrowprops=dict(arrowstyle="->", color="#1a6fa3"))

# ---------- 右：条件检查表 ----------
ax2.axis("off"); ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
ax2.set_title("逐条件检查（本意重建版）", fontsize=11)

rows = [
    ("条件", "车 A", "车 B", "车 C"),
    ("y(i-1) < 0（上一帧在车后）", "−8.0 ✓", "+16.0 ✗", "+5.0 ✗"),
    ("y(i) > 0（当前帧在车前）", "+3.0 ✓", "+11.0 ✓", "+5.0 ✓"),
    ("l = x²+y² < dis=30", "4+9=13 ✓", "4+121=125 ✗", "10.2+25=35.2 ✗"),
    ("v_i > 0 且 v_{i−1} > 0（在动）", "5.5, 5.5 ✓", "4.0, 4.0 ✓", "0, 0 ✗"),
    ("结论", "超车 ✓", "不算（太远）", "不算（没动）"),
]
y0 = 9.2
for ri, row in enumerate(rows):
    y = y0 - ri * 1.25
    bg = "#f5f5f5" if ri == 0 else ("#e8f5e9" if ri == 5 else "white")
    ax2.add_patch(mpatches.Rectangle((0.2, y - 0.55), 9.6, 1.1, fc=bg, ec="#bbb", lw=0.8))
    for ci, cell in enumerate(row):
        x = [2.6, 5.0, 7.2, 9.2][ci] if ci else 0.35
        ax2.text(x if ci else 0.35, y, cell, fontsize=9.2, ha="center" if ci else "left",
                 va="center", weight="bold" if ri == 0 else "normal")
    if ri == 5:
        for cx, col in zip([5.0, 7.2, 9.2], ["#1a6fa3", "#e69f00", "#999999"]):
            ax2.text(cx, y, rows[ri][[5.0, 7.2, 9.2].index(cx)+1], fontsize=9.2, ha="center",
                     va="center", weight="bold", color=col)

ax2.text(5, 0.9, "注意两点：① 论文 Algorithm 1 的“距离” l = x²+y²，没有开根号，\n"
                 "是平方距离；② 印刷版条件行有笔误，阈值 dis/dis_x/dis_y/s\n"
                 "的数值论文和附录都没有公布——这是复现缺口。",
         fontsize=9, ha="center", color="#b34a06", linespacing=1.6,
         bbox=dict(fc="#fff8e1", ec="#e0c36a", boxstyle="round,pad=0.4"))

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/089-Agent-nuinstruct/fig-overtake-example.png", dpi=150)
print("saved fig-overtake-example.png")
