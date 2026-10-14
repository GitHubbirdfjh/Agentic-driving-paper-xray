# -*- coding: utf-8 -*-
"""PreWorld: 体渲染 (volume rendering) 的几何含义, 用具体数字算一遍。

论文公式 (6)-(7):
  T(um) = exp(-sum_{p<m} sigma(up) * delta_p)      # 累积透射率: 光线走到 um 还剩多少
  w(um)  = T(um) * (1 - exp(-sigma(um) * delta_m)) # 该点对成像的贡献权重
  d_hat = sum_m w(um) * t_m                        # 渲染深度 = 距离按权重平均
  s_hat = sum_m w(um) * s(um)                      # 渲染语义 = 语义向量按权重平均

取 M=4 个采样点, 体素尺寸 0.4m (论文 Occ3D-nuScenes 设置), 密度场在第 3 个点
处有一个尖峰 (sigma=3.0, 即"这里有表面"), 其余接近 0 (自由空间)。

左图: 光线穿过 4 个采样点, 每个点标注 sigma, 权重 w 用圆点大小表示。
右表: 逐步算出 T, w, 以及渲染出的深度/语义, 并点出"权重集中在第一个高密度点"
这一几何事实 —— 密度场的任务就是把尖峰放在真表面上, 语义/RGB 则被密度"门控"。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ---- 具体数字 ----
sigma = np.array([0.05, 0.10, 8.00, 0.20])   # 密度场在各采样点的值: 真表面处尖峰
delta = np.full(4, 0.4)                        # 采样间隔 = 体素尺寸 0.4 m
t = np.array([10.0, 10.4, 10.8, 11.2])        # 各采样点沿光线的距离 (m)

T = np.zeros(4); w = np.zeros(4)
acc = 0.0
for m in range(4):
    T[m] = np.exp(-acc)
    w[m] = T[m] * (1 - np.exp(-sigma[m] * delta[m]))
    acc += sigma[m] * delta[m]

depth = float(np.sum(w * t))
print("T =", np.round(T, 4))
print("w =", np.round(w, 4), "sum =", round(w.sum(), 4))
print("rendered depth =", round(depth, 3), "m")

fig = plt.figure(figsize=(11.5, 4.6))
ax1 = fig.add_subplot(1, 2, 1)

# 光线
ax1.plot([0, 12.6], [0, 0], color="#6e737b", lw=2)
ax1.plot(0, 0, marker=(3, 0, -90), ms=22, color="#6e737b")  # 相机
ax1.text(0, 0.55, "camera\n(pixel ray)", ha="center", fontsize=10, color="#6e737b")
x_offsets = [-1.5, -0.35, 0.7, 1.15]
y_rows = [-1.00, -1.75, -1.00, -1.75]
for i in range(4):
    x = t[i]
    ax1.plot([x, x], [-0.35, 0.35], color="#1a6fa3", lw=1.2, ls="--", alpha=0.6)
    ax1.scatter([x], [0], s=90 + 1400 * w[i], color="#b34a06", alpha=0.85,
                edgecolors="#7a2e00", zorder=3)
    tx = x + x_offsets[i]
    ax1.annotate(f"$u_{i+1}$: $w$={w[i]:.4f}",
                 xy=(x, -0.05), xytext=(tx, y_rows[i]),
                 ha="center", va="center", fontsize=10,
                 color="#b34a06" if i == 2 else "#3d3d3a",
                 fontweight="bold" if i == 2 else "normal",
                 arrowprops=dict(arrowstyle="-", color="#6e737b", lw=0.9),
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#deddd8",
                           alpha=0.95))
ax1.annotate("weight concentrates\nat the first dense sample\n(the surface)",
             xy=(t[2], 0.35), xytext=(7.2, 1.9),
             arrowprops=dict(arrowstyle="->", color="#b34a06", lw=1.5),
             fontsize=10.5, color="#b34a06",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#b34a06"))
ax1.set_xlim(-0.8, 13.2); ax1.set_ylim(-2.1, 2.6)
ax1.set_xlabel("distance along the ray (m)", fontsize=11)
ax1.set_title("A ray, 4 samples: circle size = rendering weight $w(u_m)$",
              fontsize=12)
ax1.set_yticks([])

# 右: 数值表
ax2 = fig.add_subplot(1, 2, 2)
ax2.axis("off")
rows = []
for i in range(4):
    rows.append([f"$u_{i+1}$", f"{t[i]:.1f} m", f"{sigma[i]:.2f}",
                 f"{T[i]:.4f}", f"{w[i]:.4f}"])
col_labels = ["sample", "$t_m$", "$\\sigma(u_m)$", "$T(u_m)$", "$w(u_m)$"]
tab = ax2.table(cellText=rows, colLabels=col_labels, loc="center",
                colWidths=[0.14, 0.16, 0.20, 0.20, 0.20])
tab.auto_set_font_size(False); tab.set_fontsize(11); tab.scale(1, 1.9)
for (r, c), cell in tab.get_celld().items():
    if r == 0:
        cell.set_facecolor("#e2eff5")
    elif r == 3:
        cell.set_facecolor("#f9e3d6")
ax2.text(0.5, 0.92,
         "Worked example: $\\delta=0.4$m, density spike at $u_3$",
         ha="center", fontsize=12, transform=ax2.transAxes)
ax2.text(0.5, 0.08,
         f"Rendered depth $\\hat d=\\sum w_m t_m={depth:.2f}$ m  (true surface at 10.8 m)\n"
         f"Rendered semantic $\\hat s=\\sum w_m \\tilde s(u_m)\\approx \\tilde s(u_3)$: "
         "density gates semantics.\n"
         f"Leftover weight $1-\\sum w={1 - w.sum():.3f}$ escapes to background.",
         ha="center", fontsize=10.5, transform=ax2.transAxes, color="#3d3d3a",
         bbox=dict(boxstyle="round,pad=0.4", fc="#f5f4ed", ec="#deddd8"))

fig.suptitle("PreWorld Eq.(6)-(7), computed on a concrete ray: "
             "rendering turns a density field into 2D supervision",
             fontsize=13, y=1.00)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/053-WM-preworld/fig-rendering-geometry.png",
            dpi=150, bbox_inches="tight")
print("saved fig-rendering-geometry.png")
