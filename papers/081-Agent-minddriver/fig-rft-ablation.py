"""MindDriver 表7：渐进式 RFT vs 单阶段 RFT 的消融（nuScenes, ST-P3 口径）。"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

strategies = ["无 RL\n(SFT 基线)", "单阶段 RL\n(图像+轨迹联合奖励)", "渐进 RL\n(先图像后轨迹)"]
l2_avg  = [0.95, 0.97, 0.93]
col_avg = [0.41, 0.42, 0.38]

x = np.arange(len(strategies)); w = 0.34
fig, ax1 = plt.subplots(figsize=(10, 4.6))
b1 = ax1.bar(x - w/2, l2_avg, w, label="L2 平均误差 (m)", color="#1a6fa3", edgecolor="white")
ax1.set_ylabel("L2 平均误差 (m) ↓", fontsize=11, color="#1a6fa3")
ax1.set_xticks(x); ax1.set_xticklabels(strategies, fontsize=10.5)
ax1.set_ylim(0.85, 1.0)
ax2 = ax1.twinx()
b2 = ax2.bar(x + w/2, col_avg, w, label="碰撞率 (%)", color="#b34a06", edgecolor="white")
ax2.set_ylabel("碰撞率 (%) ↓", fontsize=11, color="#b34a06")
ax2.set_ylim(0.30, 0.48)
for b in list(b1) + list(b2):
    h = b.get_height()
    (ax1 if b in b1 else ax2).text(b.get_x() + b.get_width()/2, h + 0.004,
                                   f"{h:.2f}", ha="center", fontsize=10, weight="bold")
ax1.set_title("为什么 RFT 要分两阶段：一次优化两件事，模型顾不过来",
              fontsize=12.5, weight="bold", pad=12)
fig.text(0.5, -0.02, "数据来源：论文表 7。单阶段把图像奖励和轨迹奖励按 0.33/0.67 加权求和，"
         "结果几乎和不做 RL 一样；分两阶段先对齐“文本→图像”，再对齐“图像→轨迹”，两项都降。",
         ha="center", fontsize=9.5, color="#3d3d3a")
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/081-Agent-minddriver/fig-rft-ablation.png",
            dpi=150, bbox_inches="tight")
print("saved fig-rft-ablation.png")
