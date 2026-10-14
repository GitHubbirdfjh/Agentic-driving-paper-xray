# -*- coding: utf-8 -*-
"""OPV2V 上 AP@0.7 随带宽的变化（全部数字取自论文表 1、表 3，可核实）。
关键观察：中间协作在低带宽下快速跌破晚期融合的水平线；
CoSDH 的晚期融合分支把它托住，形成全带宽范围的上包络。
注：论文图 1a 声称 Where2comm 在 5Mbps 跌 ~7.5%，与表 1（5.4Mbps 仅跌 ~1.5%）矛盾，
此处只画表里的可核实数字，矛盾在正文怀疑式阅读中讨论。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(10, 6))

# --- 可核实数字 ---
# Where2comm: 表1 No Limit (48.7, 91.39)；表1 BD<=6.75 (5.4, 89.86)
w2c_bd = np.array([48.7, 5.4])
w2c_ap = np.array([91.39, 89.86])
# 晚期融合：表1 (0.2, 87.18)；无融合 (0, 67.16)
# CoSDH 带晚期融合：表3 ✓行：(13.40,92.99) (1.97,92.92) (0.68,92.73) (0.26,92.23) (0.19,91.68)
cosdh_bd = np.array([13.40, 1.97, 0.68, 0.26, 0.19])
cosdh_ap = np.array([92.99, 92.92, 92.73, 92.23, 91.68])
# CoSDH 不带晚期融合：表3 上半：(13.26,92.95) (1.82,92.60) (0.54,92.27) (0.12,90.67) (0.05,87.60)
cosdh_nolate_bd = np.array([13.26, 1.82, 0.54, 0.12, 0.05])
cosdh_nolate_ap = np.array([92.95, 92.60, 92.27, 90.67, 87.60])

ax.plot(w2c_bd, w2c_ap, "o-", color="#5a626b", linewidth=2, markersize=7,
        label="Where2comm（表 1 两档带宽）")
ax.plot(cosdh_nolate_bd, cosdh_nolate_ap, "s--", color="#9e5c00", linewidth=1.8,
        markersize=6, label="CoSDH 去掉晚期融合（表 3）")
ax.plot(cosdh_bd, cosdh_ap, "D-", color="#0b7a5c", linewidth=2.4, markersize=7,
        label="CoSDH 完整版（表 3，εc 从 0.01 调到 0.07）")
ax.axhline(87.18, color="#1a6fa3", linewidth=1.6, linestyle=":",
           label="纯晚期融合 87.18（0.2 Mbps，表 1）")
ax.axhline(67.16, color="#b9c2cc", linewidth=1.2, linestyle=":",
           label="无协作 67.16")

# 真实世界限制线：每智能体 6.75 Mbps
ax.axvline(6.75, color="#b34a06", linewidth=1.6, linestyle="--", alpha=0.8)
ax.text(6.2, 94.15, "真实世界限制\n6.75 Mbps/智能体", ha="right", va="top",
        fontsize=9.5, color="#b34a06",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#b34a06"))

# 标注关键落差
ax.annotate("去掉晚期融合后\n0.05 Mbps 处跌到 87.60",
            xy=(0.05, 87.60), xytext=(0.045, 84.6),
            fontsize=9.5, color="#9e5c00",
            arrowprops=dict(arrowstyle="->", color="#9e5c00", lw=1.3),
            bbox=dict(boxstyle="round,pad=0.3", fc="#f8ecd2", ec="#9e5c00"))
ax.annotate("晚期融合托住下限：\n0.19 Mbps 仍有 91.68",
            xy=(0.19, 91.68), xytext=(1.1, 89.5),
            fontsize=9.5, color="#0b5345",
            arrowprops=dict(arrowstyle="->", color="#0b7a5c", lw=1.3),
            bbox=dict(boxstyle="round,pad=0.3", fc="#dcf2e8", ec="#0b7a5c"))

ax.set_xscale("log")
ax.set_xlim(0.03, 80)
ax.set_ylim(66, 94.7)
ax.set_xlabel("每智能体带宽 (Mbps，对数轴；检测频率 10Hz)", fontsize=11.5)
ax.set_ylabel("AP@0.7 (OPV2V)", fontsize=11.5)
ax.set_title("中间-晚期混合的真正价值：晚期融合给低带宽托底", fontsize=13, pad=12)
ax.legend(fontsize=9.5, loc="lower left")
ax.grid(True, alpha=0.25, which="both")

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/084-Agent-cosdh/fig-bandwidth-curve.png",
            dpi=150, bbox_inches="tight")
print("saved fig-bandwidth-curve.png")
