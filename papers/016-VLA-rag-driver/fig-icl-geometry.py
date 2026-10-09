# -*- coding: utf-8 -*-
"""RAG-Driver: ICL 作为隐式梯度步的几何示意 (2D 小实例).
设定: 单头, W_Q=W_K=W_V=I.
z_icl1=[1,0] (样本1), z_icl2=[1,1] (样本2), z_q=[1,0.1] (当前query).
DeltaW_ICL = sum_i z_icl,i z_icl,i^T = [[2,1],[1,1]]
W_ZSL = z_q z_q^T = [[1,0.1],[0.1,0.01]]
zero-shot 输出 = W_ZSL z_q = [1.01, 0.101]
ICL 输出 = (DeltaW_ICL + W_ZSL) z_q = [3.11, 1.201]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Noto Serif CJK JP"
plt.rcParams["axes.unicode_minus"] = False

z_q = np.array([1.0, 0.1])
z1 = np.array([1.0, 0.0])
z2 = np.array([1.0, 1.0])
dW = np.outer(z1, z1) + np.outer(z2, z2)
Wzsl = np.outer(z_q, z_q)
out_zs = Wzsl @ z_q
out_icl = (dW + Wzsl) @ z_q

fig, ax = plt.subplots(figsize=(7.5, 6))
ax.set_xlim(-0.3, 3.6)
ax.set_ylim(-0.3, 2.2)
ax.set_aspect("equal")
ax.set_xlabel("第 1 维")
ax.set_ylabel("第 2 维")

def arrow(v, color, label, ls="-", width=0.02, alpha=1.0, dx=0.05, dy=0.05):
    ax.annotate("", xy=v, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2,
                                linestyle=ls, shrinkA=0, shrinkB=4))
    ax.text(v[0] * 1.06 + dx, v[1] * 1.06 + dy, label, color=color, fontsize=11)

arrow(z1, "#6e737b", "$z_{icl,1}$=[1,0]", ls="--", dy=-0.22)
arrow(z2, "#6a55c2", "$z_{icl,2}$=[1,1]", ls="--", dy=0.08)
arrow(z_q, "#1a6fa3", "$z_q$=[1,0.1]", ls="--", dx=0.02, dy=0.10)
arrow(out_zs, "#9e5c00", "zero-shot\n[1.01, 0.10]", width=0.025, dx=0.05, dy=-0.30)
arrow(out_icl, "#0b7a5c", "ICL 输出\n[3.11, 1.20]", width=0.025, dx=0.08, dy=0.05)

# 画出"被拽"的位移
ax.annotate("", xy=out_icl, xytext=out_zs,
            arrowprops=dict(arrowstyle="->", color="#b34a06", lw=1.5,
                            linestyle=":", shrinkA=6, shrinkB=6))
mx = (out_zs + out_icl) / 2
ax.text(mx[0] + 0.12, mx[1] - 0.12, "ΔW_ICL 的拖拽\n∝ (z_icl·z_q)", color="#b34a06", fontsize=10)

ax.set_title("ICL 前向 ≈ 在虚拟权重上做了一步梯度更新", fontsize=13, fontweight="bold", pad=12)
ax.text(0.02, 0.98,
        "ΔW_ICL = Σ z_icl,i z_icl,iᵀ，把输出往 ICL 样本的 value 方向拽；\n"
        "拽的力度 = query 与样本 key 的点积（样本2: 1.1 > 样本1: 1.0）。",
        transform=ax.transAxes, va="top", ha="left", fontsize=10, color="#3d3d3a",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f4ed", edgecolor="#deddd8"))
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/016-VLA-rag-driver/fig-icl-geometry.png", dpi=150)
print("saved fig-icl-geometry.png")
