"""AutoDrive-R2 表4：物理接地奖励四项各自的贡献（nuScenes L2 avg）。"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

labels = ["完整模型\n(0.19m)", "去掉 r_pos\n空间对齐", "去掉 r_ste\n方向盘转角",
          "去掉 r_vel\n速度", "去掉 r_tem\n时序平滑"]
vals = [0.19, 0.53, 0.21, 0.22, 0.24]
colors = ["#0b7a5c", "#b34a06", "#9e5c00", "#1a6fa3", "#6a55c2"]

fig, ax = plt.subplots(figsize=(10.5, 4.8))
bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.2f} m",
            ha="center", fontsize=11, weight="bold")
ax.set_ylabel("nuScenes L2 平均误差 (m) ↓", fontsize=11)
ax.set_ylim(0, 0.62)
ax.axhline(0.19, color="#0b7a5c", ls="--", lw=1.2, alpha=0.7)
ax.set_title("四个物理奖励项，哪个最不能少：空间对齐 r_pos 是承重墙",
             fontsize=12.5, weight="bold", pad=12)
fig.text(0.5, -0.02, "数据来源：论文表 4（RL 消融，7B 模型）。"
         "去掉位置对齐误差直接从 0.19 涨到 0.53（+179%）；"
         "时序平滑去掉涨 26.3%，方向盘/速度约束各贡献约 10%。\n"
         "注意：这组消融只动 RL 阶段，SFT 的四步链都还在，所以数字反映的是奖励设计的净贡献。",
         ha="center", fontsize=9.5, color="#3d3d3a")
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/082-Agent-autodrive-r2/fig-reward-ablation.png",
            dpi=150, bbox_inches="tight")
print("saved fig-reward-ablation.png")
