"""自绘图：传输代价(BPS,对数) vs 规划碰撞率 —— 表2/表3的数据
一句话：花100倍带宽传BEV特征，碰撞率几乎不动；传1/100带宽的混合数据，碰撞率降到约1/3。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

methods = ["No Fusion", "Vanilla", "BEV Feature\nFusion", "CooperNaut", "UniV2X"]
bps = [0, 8.19e7, 8.19e7, 8.19e7, 8.09e5]          # 表2, Byte Per Second
col = [0.89, 2.12, 0.83, 1.23, 0.34]               # 表2, 平均碰撞率 %
l2  = [3.44, 3.71, 3.30, 5.35, 3.48]               # 表2, 平均L2误差 m

colors = ["#9aa0a8", "#b34a06", "#b34a06", "#b34a06", "#0b7a5c"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
fig.patch.set_facecolor("#faf9f5")
for ax in axes:
    ax.set_facecolor("#faf9f5")

# (a) 传输代价, log scale. No Fusion = 0 单独标注
x = np.arange(len(methods))
vals = [1 if v == 0 else v for v in bps]  # 0画不出来, 用1占位再标注
bars = axes[0].bar(x, vals, color=colors, edgecolor="#141413", linewidth=0.8, width=0.62)
axes[0].set_yscale("log")
axes[0].set_xticks(x)
axes[0].set_xticklabels(methods, fontsize=10)
axes[0].set_ylabel("Bytes Per Second (log scale)", fontsize=11)
axes[0].set_title("(a) How much to send: bytes per second", fontsize=12, pad=10)
axes[0].text(0, 2.2, "0 (no transmission)", ha="center", fontsize=10, color="#141413",
             bbox=dict(boxstyle="round,pad=0.25", fc="#f5f4ed", ec="#deddd8"))
axes[0].text(4, 8.09e5 * 3.2, "8.09e5\n(~1/100 of BEV fusion)", ha="center", fontsize=9, color="#0b7a5c")
axes[0].tick_params(axis="y", labelsize=9)
axes[0].grid(axis="y", which="major", color="#deddd8", linewidth=0.6, alpha=0.8)

# (b) 平均碰撞率
bars2 = axes[1].bar(x, col, color=colors, edgecolor="#141413", linewidth=0.8, width=0.62)
axes[1].set_xticks(x)
axes[1].set_xticklabels(methods, fontsize=10)
axes[1].set_ylabel("Avg. Collision Rate (%)", fontsize=11)
axes[1].set_title("(b) What it buys: avg. collision rate (lower is better)", fontsize=12, pad=10)
for i, v in enumerate(col):
    axes[1].text(i, v + 0.06, f"{v:.2f}", ha="center", fontsize=10)
axes[1].text(4, 0.34 + 0.28, "0.89 -> 0.34\n(-61%)", ha="center", fontsize=9, color="#0b7a5c")
axes[1].set_ylim(0, 2.6)
axes[1].grid(axis="y", color="#deddd8", linewidth=0.6, alpha=0.8)

# L2 小字注记: UniV2X 的 L2 并没有变好
fig.text(0.5, 0.01,
         "Note: UniV2X avg. L2 error is 3.48m, slightly worse than No Fusion 3.44m - the planning gain is all in safety metrics, not trajectory fitting.",
         ha="center", fontsize=9, color="#73726c")

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("/home/hatch/workspace/your_files/paper-xray-40/multi-agent-driving/univ2x-xray/fig-bandwidth.png",
            dpi=150, bbox_inches="tight", facecolor="#faf9f5")
print("saved fig-bandwidth.png")
