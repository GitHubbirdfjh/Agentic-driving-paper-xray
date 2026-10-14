# -*- coding: utf-8 -*-
"""ADRScore 玩具验算：论文图5的两个数值例子.
左: 语义相似度矩阵 -> RA=0.75, RD=0.6, MS=0.4, ADRScore=0.583
右: 注入几何相似度后 -> RA=0.6, RD=0.5, MS=0.4, ADRScore-S=0.5
矩阵为示意构造, 仅保证列最大值/行最大值的均值与最小值和论文图5一致.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 行=r1..r5 (reference), 列=h1..h4 (hypothesis)
S_sem = np.array([
    [0.80, 0.30, 0.20, 0.10],
    [0.30, 0.90, 0.20, 0.10],
    [0.20, 0.30, 0.60, 0.20],
    [0.20, 0.20, 0.30, 0.70],
    [0.40, 0.20, 0.10, 0.10],
])
S_geo = np.array([
    [0.70, 0.30, 0.20, 0.10],
    [0.30, 0.60, 0.20, 0.10],
    [0.20, 0.30, 0.50, 0.20],
    [0.20, 0.20, 0.30, 0.60],
    [0.40, 0.20, 0.10, 0.10],
])

def scores(S):
    col_max = S.max(axis=0)          # alpha_i = max_j s_{i,j}
    row_max = S.max(axis=1)          # 反向: align(r_i -> h)
    RA = col_max.mean()
    RD = col_max.min()
    MS = row_max.min()
    return RA, RD, MS, (RA + RD + MS) / 3, col_max, row_max

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
for ax, S, title in zip(axes,
                        [S_sem, S_geo],
                        ["semantic similarity  (cos of BERT embeddings)",
                         "visual-adapted  (geometric similarity injected)"]):
    RA, RD, MS, ADR, col_max, row_max = scores(S)
    im = ax.imshow(S, cmap="YlOrRd", vmin=0, vmax=1)
    for i in range(5):
        for j in range(4):
            ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center",
                    fontsize=10, color="black" if S[i, j] < 0.55 else "white")
    # 标出每列最大值(蓝色框)= alpha_i, 每行最大值(绿色框)
    for j in range(4):
        i = int(np.argmax(S[:, j]))
        ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                   fill=False, edgecolor="#1a6fa3", lw=2.5))
    ax.set_xticks(range(4)); ax.set_xticklabels([f"h{j+1}" for j in range(4)])
    ax.set_yticks(range(5)); ax.set_yticklabels([f"r{i+1}" for i in range(5)])
    ax.set_title(title, fontsize=11)
    ax.text(4.55, 0.4,
            f"col max (alpha) = [{', '.join(f'{v:.2f}' for v in col_max)}]\n"
            f"RA = mean = {RA:.3f}\nRD = min = {RD:.3f}\n"
            f"row max min: MS = {MS:.3f}\nADRScore = {ADR:.3f}",
            fontsize=10, va="top", ha="left",
            bbox=dict(boxstyle="round", fc="#f5f4ed", ec="#deddd8"))
    ax.text(0.5, -0.9, "blue box = per-column max (alpha_i)", fontsize=9,
            color="#1a6fa3", transform=ax.transData, ha="left", va="top")

fig.suptitle("ADRScore toy verification: numbers match Fig.5 of the paper "
             "(schematic matrices)", fontsize=12, y=1.02)
fig.colorbar(im, ax=axes, shrink=0.85, label="similarity s(i,j)")
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/"
            "019-VLA-reason2drive/fig-adrscore.png", dpi=150, bbox_inches="tight")
print("saved fig-adrscore.png")
for S, name in [(S_sem, "semantic"), (S_geo, "visual")]:
    RA, RD, MS, ADR, _, _ = scores(S)
    print(name, f"RA={RA:.3f} RD={RD:.3f} MS={MS:.3f} ADR={ADR:.3f}")
