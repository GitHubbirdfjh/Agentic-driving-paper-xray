# -*- coding: utf-8 -*-
"""AdaWM: Theorem 1 的判决规则画成几何。

横轴: 策略失配 E_pi (论文定义的 max_x DTV(pi || pi_hat))
纵轴: 动力学模型失配 E_{Phat} (E_pi[DTV(P || Phat)] 的上界)

理论判据: 若 E_{Phat} >= C1 * E_pi - C2, 则模型失配主导 -> 更新动力学模型;
否则策略失配主导 -> 更新策略。
即 (E_pi, E_{Phat}) 平面上的一条分界直线, 直线之上更新模型, 之下更新策略。

图中还放了两条线做对比:
  蓝线: 用附录 Table 6 的 DreamerV3 默认值算出的理论 C1
        (gamma=0.997, K=H=15, r_max=1 -> C1 ~= 31.5)
  红虚线: 实现中实际用的简化判据 DTV(P||Phat) > C * DTV(pi||pihat),
        取论文 Table 4 最优的 C=5 附近。
两条线的斜率差, 就是"理论判据太保守、实现只能调参"的直观证据。

另画一条示意的微调轨迹: 起点模型失配主导(直线上方), 随微调进行
两点失配都下降, 穿过分界线后策略切换为更新策略。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ---- 理论 C1 的具体数值 ----
gamma = 0.997
K = 15
r_max = 1.0
C1_theory = 2 * r_max * (1 - gamma ** K) / (gamma - gamma ** 2) + 2
print(f"C1_theory (gamma={gamma}, K={K}, r_max={r_max}) = {C1_theory:.2f}")
C_practice = 5.0  # Table 4 最优附近

fig, ax = plt.subplots(figsize=(8.2, 6.2))

x = np.linspace(0, 0.30, 200)
C2 = 0.02  # 示意值: C2 = gamma^{K-1} E_max / (2 Gamma), 量级小, 只做平移
ax.plot(x, C1_theory * x - C2, color="#1a6fa3", lw=2.4,
        label=f"Theory boundary: $E_{{\\hat P}} = C_1 E_\\pi - C_2$  ($C_1\\approx{C1_theory:.1f}$)")
ax.plot(x, C_practice * x, color="#b34a06", lw=2.2, ls="--",
        label=f"Practice (Alg.1): $DTV(P\\|\\hat P) = C\\cdot DTV(\\pi\\|\\hat\\pi)$, $C=5$")

# 区域标注
ax.fill_between(x, C1_theory * x - C2, 9.5, color="#1a6fa3", alpha=0.10)
ax.fill_between(x, 0, np.minimum(C_practice * x, 9.5), color="#b34a06", alpha=0.08)
ax.text(0.235, 5.4, "above the line:\nupdate dynamics\nmodel", color="#1a6fa3",
        fontsize=11, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#1a6fa3", alpha=0.9))
ax.text(0.20, 0.55, "below the line:\nupdate policy", color="#b34a06",
        fontsize=11, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#b34a06", alpha=0.9))

# 示意微调轨迹: (E_pi, E_Phat) 随步骤下降, 穿过理论分界线
traj = np.array([
    [0.030, 2.60], [0.055, 2.05], [0.075, 1.55], [0.095, 1.10],
    [0.120, 0.80], [0.150, 0.62], [0.185, 0.50], [0.225, 0.42],
])
ax.plot(traj[:, 0], traj[:, 1], "o-", color="#0b7a5c", ms=7, lw=1.6,
        label="Example finetuning trajectory $(E_\\pi, E_{\\hat P})$")
for i, (px, py) in enumerate(traj[::2]):
    ax.annotate(f"t={i * 2}", (px, py), textcoords="offset points",
                xytext=(10, 8), fontsize=9, color="#0b7a5c")
ax.annotate("strategy switches here:\nmodel updates -> policy updates",
            xy=(0.150, 0.62), xytext=(0.055, 3.6),
            arrowprops=dict(arrowstyle="->", color="#0b7a5c", lw=1.4),
            fontsize=10, color="#0b7a5c",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#0b7a5c", alpha=0.9))

ax.set_xlim(0, 0.30)
ax.set_ylim(0, 6.5)
ax.set_xlabel("Policy mismatch  $E_\\pi = \\max_x D_{TV}(\\pi|\\hat\\pi)$", fontsize=12)
ax.set_ylabel("Dynamics-model mismatch  $E_{\\hat P}$  (bounds expected TV distance)",
              fontsize=12)
ax.set_title("AdaWM's decision rule as geometry: which mismatch to fix first?",
             fontsize=13, pad=12)
ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/052-WM-adawm/fig-decision-plane.png",
            dpi=150)
print("saved fig-decision-plane.png")
