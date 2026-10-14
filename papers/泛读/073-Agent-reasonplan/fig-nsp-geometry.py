# -*- coding: utf-8 -*-
"""NSP 的几何动作：潜空间里的向量预测。
把 1152 维降到示意 2D：当前帧 H_vt、LLM 输出的预测 H^_{t+3}、
真值 H_{t+3}，MSE 把预测点往真值方向拉。
数字为示意（论文没有给出实际向量）。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

IN = "#1a6fa3"; OUT = "#9e5c00"; LOSS = "#b34a06"; WIRE = "#6e737b"; ALT = "#6a55c2"

fig, ax = plt.subplots(figsize=(8.5, 6.4))
ax.set_xlim(-0.5, 6.5); ax.set_ylim(-0.5, 6.5)
ax.set_aspect("equal")
ax.set_xlabel("潜空间维度 1（示意）"); ax.set_ylabel("潜空间维度 2（示意）")
ax.set_title("NSP：潜空间里的三点关系（2D 示意，真实维度为 1152）", fontsize=13, weight="bold")

H_t = np.array([1.0, 1.0])        # 当前帧视觉嵌入（示意）
H_t3 = np.array([4.2, 5.1])       # t+3 真值视觉嵌入（示意）
H_hat = np.array([2.6, 3.9])      # LLM 预测（示意）

ax.scatter(*H_t, s=170, color=IN, zorder=5, edgecolors="white", linewidths=1.5)
ax.annotate("H_vt：t 时刻视觉嵌入\n（concat 上下文编码后进入 LLM）",
            xy=tuple(H_t), xytext=(-0.2, 3.2), fontsize=11, color=IN,
            arrowprops=dict(arrowstyle="->", color=IN, lw=1.5))

ax.scatter(*H_hat, s=170, color=ALT, zorder=5, edgecolors="white", linewidths=1.5)
ax.annotate("Ĥ_{t+3}：LLM 预测的\nt+3 视觉嵌入",
            xy=tuple(H_hat), xytext=(1.3, 4.35), fontsize=11, color=ALT,
            arrowprops=dict(arrowstyle="->", color=ALT, lw=1.5))

ax.scatter(*H_t3, s=170, color=OUT, zorder=5, edgecolors="white", linewidths=1.5)
ax.annotate("H_{t+3}：真值（同一 SigLIP+MLP\n从 t+3 帧图像编码，参数共享）",
            xy=tuple(H_t3), xytext=(4.4, 4.0), fontsize=11, color=OUT,
            arrowprops=dict(arrowstyle="->", color=OUT, lw=1.5))

# LLM 的预测动作：从当前点出发
ax.annotate("", xy=tuple(H_hat), xytext=tuple(H_t),
            arrowprops=dict(arrowstyle="->", color=ALT, lw=2.2, ls="--"))
ax.text(1.7, 2.2, "LLM 的预测\n（上下文引导）", fontsize=10, color=ALT)

# MSE 梯度动作：把预测点拉向真值
ax.annotate("", xy=tuple(H_t3), xytext=tuple(H_hat),
            arrowprops=dict(arrowstyle="->", color=LOSS, lw=2.6))
ax.text(3.55, 4.7, "MSE 梯度方向\n‖Ĥ-H‖² 的负梯度", fontsize=10, color=LOSS, ha="center")

ax.text(-0.45, 6.15,
        "注意：MSE 的比较对象不是像素，而是 SigLIP 潜空间向量 ——\n"
        "这就是它比像素重建便宜、但比纯文本监督多视觉细节的原因。",
        fontsize=10.5, color="#3d3d3a",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#faf9f5", edgecolor="#deddd8"))

ax.text(-0.45, -0.28, "图：NSP 在潜空间的几何动作 —— 预测点 Ĥ_{t+3} 被 MSE 拉向真值 H_{t+3}（示意数）",
        fontsize=10, color="#73726c")
plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/073-Agent-reasonplan/fig-nsp-geometry.png",
            dpi=150, bbox_inches="tight")
print("saved fig-nsp-geometry.png")
