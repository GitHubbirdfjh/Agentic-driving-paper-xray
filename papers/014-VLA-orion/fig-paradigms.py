# -*- coding: utf-8 -*-
"""ORION 论文图 5 右侧数据的复现: 四种 VLM 输出范式在 Bench2Drive 闭环上的对比。
数据来源: 论文 Sec 4.5 文字 —— plain text (42.23/13.14/15.39);
MLP decode 由 "+7.01/+9.5/+6.28" 反推为 (70.73/45.12/48.44);
dual-system (meta-action guided VAD) 在图中, 文字未给精确值, 按图上柱高估读标注为近似值;
ours (77.74/54.62/54.72)。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

labels = ["Plain Text\n(VLM 直接吐文本轨迹)", "Dual-System\n(meta-action 指导 VAD)",
          "MLP Decode\n(planning token 接 MLP)", "ORION\n(VAE 对齐+生成式规划)"]
ds = [42.23, 44.94, 70.73, 77.74]
sr = [13.14, 15.45, 45.12, 54.62]
ability = [15.39, 20.65, 48.44, 54.72]
# dual-system 的精确值论文文字未给, 图上柱高约为 DS~45/SR~15/Ability~21, 标注为估读
note_dual = True

x = np.arange(len(labels))
w = 0.24
fig, ax = plt.subplots(figsize=(12, 5.2))
b1 = ax.bar(x - w, ds, w, label="Driving Score", color="#1a6fa3", edgecolor="white")
b2 = ax.bar(x, sr, w, label="Success Rate (%)", color="#9e5c00", edgecolor="white")
b3 = ax.bar(x + w, ability, w, label="Ability Mean (%)", color="#0b7a5c", edgecolor="white")

for bars in (b1, b2, b3):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                f"{b.get_height():.2f}", ha="center", fontsize=9.5)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylabel("分数", fontsize=12)
ax.set_title("四种 VLM 输出范式的 Bench2Drive 闭环对比 (论文图 5 数据复现)", fontsize=14)
ax.legend(fontsize=11)
ax.set_ylim(0, 92)
ax.grid(axis="y", alpha=0.3)
fig.text(0.5, -0.03,
         "数据来源: 论文 4.5 节。dual-system 一组论文只给了图、未给精确数字, 此处按柱高估读 "
         "(DS≈44.94/SR≈15.45/Ability≈20.65), 仅作示意, 不得引用。\n"
         "MLP decode 一组由文字 \"+7.01 DS/+9.5% SR/+6.28% ability\" 反推。核心结论: "
         "plain text 最差, dual-system 只比它好一点, MLP decode 已很强, VAE 对齐再往上提一截。",
         ha="center", fontsize=10, color="#3d3d3a",
         bbox=dict(boxstyle="round", fc="#f5f4ed", ec="#deddd8"))
plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/014-VLA-orion/fig-paradigms.png",
            dpi=150, bbox_inches="tight")
print("saved fig-paradigms.png")
