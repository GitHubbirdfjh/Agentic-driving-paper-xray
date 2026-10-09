# -*- coding: utf-8 -*-
"""RAG-Driver: 零样本泛化 (Spoken-SAX) 的 CIDEr 对比.
数据来源: 论文表 I 下半部分. Action Explanation 与 Justification 的 CIDEr.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Noto Serif CJK JP"
plt.rcParams["axes.unicode_minus"] = False

labels = ["动作解释\nCIDEr", "动作理由\nCIDEr"]
base = [5.7, 4.7]      # Base w/o ICL
adapt = [22.3, 8.6]    # ADAPT (specialist)
ours = [48.9, 17.1]    # OURS w ICL

x = np.arange(len(labels))
w = 0.24
fig, ax = plt.subplots(figsize=(8, 4.6))
b1 = ax.bar(x - w, base, w, label="Base w/o ICL (MLLM, 无检索)", color="#6e737b")
b2 = ax.bar(x, adapt, w, label="ADAPT (specialist)", color="#1a6fa3")
b3 = ax.bar(x + w, ours, w, label="OURS w ICL", color="#0b7a5c")
for bars in (b1, b2, b3):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                f"{b.get_height()}", ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("CIDEr (越高越好)")
ax.set_title("零样本泛化到 Spoken-SAX（伦敦）：只在 BDD-X 上训练，不微调（论文表 I）",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10, loc="upper left")
ax.set_ylim(0, 60)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/016-VLA-rag-driver/fig-ood-cider.png", dpi=150)
print("saved fig-ood-cider.png")
