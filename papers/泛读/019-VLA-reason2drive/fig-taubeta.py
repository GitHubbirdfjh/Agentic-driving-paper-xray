# -*- coding: utf-8 -*-
"""ADRScore-S 的几何相似度归一化: s = tau^{-M/beta}, tau=15, beta=10.
标出论文图5例子对应的点: s=0.6 -> M≈1.89; s=0.5 -> M≈2.56.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

tau, beta = 15.0, 10.0
M = np.linspace(0, 20, 400)
s = tau ** (-M / beta)

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(M, s, color="#b34a06", lw=2.5, label=r"$s = \tau^{-M/\beta},\ \tau=15,\ \beta=10$")

for sv, color in [(0.6, "#1a6fa3"), (0.5, "#0b7a5c")]:
    Mv = -beta * np.log(sv) / np.log(tau)
    ax.plot([Mv], [sv], "o", color=color, ms=9)
    ax.annotate(f"s={sv}\nM={Mv:.2f}", xy=(Mv, sv), xytext=(Mv + 3.2, sv + 0.12),
                fontsize=10, color=color,
                arrowprops=dict(arrowstyle="->", color=color))

for Mv in [10, 20]:
    sv = tau ** (-Mv / beta)
    ax.plot([Mv], [sv], "x", color="gray", ms=8)
    ax.text(Mv + 0.3, sv + 0.02, f"M={Mv}: s={sv:.3f}", fontsize=9, color="gray")

ax.set_xlabel("M = MSE between two perceptual elements", fontsize=11)
ax.set_ylabel("geometric similarity s(i,j)", fontsize=11)
ax.set_title("How harsh is ADRScore-S: geometric similarity vs MSE "
             "(tau=15, beta=10)", fontsize=12)
ax.set_xlim(0, 20); ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/"
            "019-VLA-reason2drive/fig-taubeta.png", dpi=150, bbox_inches="tight")
print("saved fig-taubeta.png")
