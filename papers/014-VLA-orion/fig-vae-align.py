# -*- coding: utf-8 -*-
"""ORION 生成式规划器的核心机制: VAE 把推理空间的 planning token 和
动作空间的 GT 轨迹投影到同一个高斯隐空间, 用 KL 把两个分布拽到一起。

左图: 训练前, present(只看 planning token s)与 future(看到 GT 轨迹 t)两个
高斯相距很远; 右图: KL 损失把 present 拽向 future。
下方面板用 1 维小例子手算 KL, 验证代码里的 ProbabilisticLoss 公式。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

def gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))

def kl_1d(mu_p, sig_p, mu_q, sig_q):
    # D_KL(P||Q), P=present, Q=future —— 代码 ProbabilisticLoss 的方向是 D_KL(future||present),
    # 这里演示对称写法, 数值例子用代码的真实方向
    return (np.log(sig_q / sig_p)
            + (sig_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sig_q ** 2) - 0.5)

x = np.linspace(-4, 6, 400)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)

# 训练前: present 与 future 相距远
axes[0].plot(x, gauss(x, 0.2, 0.6), color="#1a6fa3", lw=2.5, label="present $p(z|s)$: 只看 planning token")
axes[0].plot(x, gauss(x, 3.0, 0.5), color="#9e5c00", lw=2.5, label="future $p(z|t,s)$: 看到 GT 轨迹")
axes[0].fill_between(x, gauss(x, 0.2, 0.6), alpha=0.15, color="#1a6fa3")
axes[0].fill_between(x, gauss(x, 3.0, 0.5), alpha=0.15, color="#9e5c00")
axes[0].set_title("KL 对齐前: 两个分布各说各话", fontsize=12)
axes[0].legend(fontsize=9.5)
axes[0].set_xlabel("隐变量 z (示意为 1 维, 实际 32 维)")

# 训练后: present 被拽向 future
axes[1].plot(x, gauss(x, 2.6, 0.55), color="#1a6fa3", lw=2.5, label="present $p(z|s)$: 被 KL 拽过来")
axes[1].plot(x, gauss(x, 3.0, 0.5), color="#9e5c00", lw=2.5, label="future $p(z|t,s)$: 目标")
axes[1].fill_between(x, gauss(x, 2.6, 0.55), alpha=0.15, color="#1a6fa3")
axes[1].fill_between(x, gauss(x, 3.0, 0.5), alpha=0.15, color="#9e5c00")
axes[1].annotate("", xy=(2.75, 0.55), xytext=(0.6, 0.55),
                 arrowprops=dict(arrowstyle="->", color="#b34a06", lw=2))
axes[1].text(1.6, 0.62, "$L_{vae}$ 的梯度方向", color="#b34a06", fontsize=10, ha="center")
axes[1].set_title("KL 对齐后: 推理时只用 present 分支采样", fontsize=12)
axes[1].legend(fontsize=9.5)
axes[1].set_xlabel("隐变量 z (示意为 1 维, 实际 32 维)")

fig.suptitle("ORION 的生成式规划器: 两个高斯在隐空间里被 KL 拽到一起", fontsize=14, y=1.0)

# 手算验证: 代码 ProbabilisticLoss 的闭式
# 取 present: mu=0.2, log_sig=log(0.6); future: mu=3.0, log_sig=log(0.5)
# D_KL(future||present) = log(sig_p/sig_f) + (sig_f^2 + (mu_f-mu_p)^2)/(2 sig_p^2) - 0.5
mu_p, sig_p, mu_f, sig_f = 0.2, 0.6, 3.0, 0.5
kl_code = (np.log(sig_p / sig_f)
           + (sig_f ** 2 + (mu_f - mu_p) ** 2) / (2 * sig_p ** 2) - 0.5)
fig.text(0.5, -0.06,
         f"手算验证(代码方向 $D_{{KL}}(future\\Vert present)$): "
         f"μp={mu_p}, σp={sig_p}, μf={mu_f}, σf={sig_f} → "
         f"KL = ln({sig_p}/{sig_f}) + ({sig_f}² + {mu_f-mu_p}²)/(2×{sig_p}²) − 0.5 = {kl_code:.3f}。\n"
         "KL 越大, present 离 future 越远, 梯度就把 present 的 (μ, σ) 往 future 推; "
         "推理时 GT 不存在, 只能从 present 采样 —— 所以训练时采样走 future 分支(后验), 推理走 present 分支(先验)。",
         ha="center", fontsize=10, color="#3d3d3a",
         bbox=dict(boxstyle="round", fc="#f5f4ed", ec="#deddd8"))
plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/014-VLA-orion/fig-vae-align.png",
            dpi=150, bbox_inches="tight")
print("saved fig-vae-align.png, KL =", round(float(kl_code), 3))
