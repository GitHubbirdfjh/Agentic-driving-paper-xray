# -*- coding: utf-8 -*-
"""Fig: Dolphins' honest limitation — inference latency.
Paper reports 1.34s average inference on an NVIDIA A100 (DriveLM dataset).
A data-center A100 is far above any in-vehicle computer; 1.34s > human
brake reaction time. Hence "conversational driving assistant", not controller.
"""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams["font.family"] = ["Noto Serif CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

BG = "#faf9f5"; INK = "#141413"
C_LOSS, C_LOSS_F = "#b34a06", "#f9e3d6"
C_OP, C_OP_F = "#0b7a5c", "#dcf2e8"
C_IN, C_IN_F = "#1a6fa3", "#e2eff5"
C_WIRE = "#6e737b"

fig, ax = plt.subplots(figsize=(12, 4.4))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
ax.set_title("1.34 秒意味着什么：Dolphins 只能当“副驾驶”，当不了“驾驶员”",
             fontsize=12.5, color=INK, pad=12)

# timeline axis: 0 .. 2.0 s
ax.plot([1, 11], [2.2, 2.2], color=C_WIRE, lw=2)
for s in [0, 0.5, 1.0, 1.5, 2.0]:
    x = 1 + s*5
    ax.plot([x, x], [2.2, 2.05], color=C_WIRE, lw=1.5)
    ax.text(x, 1.75, f"{s:.1f}s", fontsize=10, color=C_WIRE, ha="center")

# human reaction bar
ax.add_patch(patches.FancyBboxPatch((1, 2.9), 5.0, 0.7, boxstyle="round,pad=0.02",
                                    fc=C_OP_F, ec=C_OP, lw=1.8))
ax.text(3.5, 3.25, "人类驾驶员刹车反应 ≈ 1.0s", fontsize=10.5, color=C_OP, ha="center", weight="bold")

# dolphins bar
ax.add_patch(patches.FancyBboxPatch((1, 3.95), 6.7, 0.7, boxstyle="round,pad=0.02",
                                    fc=C_LOSS_F, ec=C_LOSS, lw=1.8))
ax.text(4.35, 4.3, "Dolphins 单次推理 1.34s（A100，论文第 6 节自报）", fontsize=10.5, color=C_LOSS,
        ha="center", weight="bold")

ax.annotate("", xy=(7.7, 3.6), xytext=(6.1, 3.6),
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
ax.text(8.6, 3.35, "还没算上车载芯片\n比 A100 慢一个数量级", fontsize=9.5, color=INK, ha="left",
        bbox=dict(fc="#ffffff", ec=C_WIRE, boxstyle="round,pad=0.3"))

ax.text(6, 0.85, "论文自己在结论里承认了这一点：算力开销和功耗是上车的硬伤，出路是蒸馏/定制小模型。\n"
        "所以 Dolphins 的定位从一开始就是“conversational driving assistant”（对话式驾驶助手）——\n"
        "做高层行为理解、解释和建议，不做毫秒级的底层控制。这是 2023 年底把 VLM 引进驾驶时必须回答的第一个问题：\n"
        "开放世界理解（VLM 的长处）和精确实时控制（VLM 的短处）之间，到底让 VLM 站哪个位置？Dolphins 的答案是：站理解和对话这边。",
        fontsize=9.2, color=INK, ha="center",
        bbox=dict(fc="#f5f4ed", ec="#deddd8", boxstyle="round,pad=0.4"))

plt.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/076-Agent-dolphins/fig-latency.png"
plt.savefig(out, dpi=150, facecolor=BG)
print("saved", out)
