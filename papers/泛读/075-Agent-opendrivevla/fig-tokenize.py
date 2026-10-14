# -*- coding: utf-8 -*-
"""Fig: trajectory tokenization in OpenDriveVLA (Stage 3, eq. 5-6).
Continuous 6 waypoints (3s @ 0.5s) -> serialized as TEXT tokens ->
LLM autoregressive next-token prediction (temp=0) -> regex-parsed back
to floats. The geometric point: planning becomes language modeling.
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
C_IN, C_IN_F = "#1a6fa3", "#e2eff5"
C_OP, C_OP_F = "#0b7a5c", "#dcf2e8"
C_OUT, C_OUT_F = "#9e5c00", "#f8ecd2"
C_WIRE = "#6e737b"; C_LOSS = "#b34a06"

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.4))
fig.patch.set_facecolor(BG)
for ax in (ax1, ax2, ax3):
    ax.set_facecolor(BG)

# ---------- panel 1: continuous waypoints ----------
ax1.set_xlim(-1, 11); ax1.set_ylim(-1, 9.8); ax1.set_aspect("equal"); ax1.axis("off")
ax1.set_title("连续轨迹：6 个路点（3 秒，每 0.5 秒一个）", fontsize=11, color=INK, pad=8)
t = np.linspace(0, 1, 200)
cx, cy = 8*(1-np.cos(t*np.pi/2.6)), 8*np.sin(t*np.pi/2.6)  # right-turn-ish curve
ax1.plot(cx, cy, color=C_IN, lw=2.5)
wp_t = np.linspace(0.08, 1, 6)
wx, wy = 8*(1-np.cos(wp_t*np.pi/2.6)), 8*np.sin(wp_t*np.pi/2.6)
for i, (x, y) in enumerate(zip(wx, wy)):
    ax1.scatter([x], [y], s=110, c=C_IN_F, edgecolors=C_IN, linewidths=2, zorder=3)
    if i == 0:
        ax1.text(x+0.35, y+0.15, f"w{i+1}\n({x:.1f},{y:.1f})", fontsize=8, color=C_IN, ha="left", va="center")
    else:
        ax1.text(x, y+0.55, f"w{i+1}\n({x:.1f},{y:.1f})", fontsize=8, color=C_IN, ha="center")
ax1.scatter([0], [0], s=160, c="#ffffff", edgecolors=INK, linewidths=2, marker="s", zorder=3)
ax1.text(0, -0.7, "自车 (0,0)", fontsize=9, color=INK, ha="center")
ax1.text(5.5, 9.15, "W_ego = {w1..w6}，每个 wt = (xt, yt) 连续坐标", fontsize=9, color=INK, ha="center",
         bbox=dict(fc="#ffffff", ec=C_IN, boxstyle="round,pad=0.3"))

# ---------- panel 2: text serialization -> LLM ----------
ax2.set_xlim(0, 10); ax2.set_ylim(0, 5); ax2.axis("off")
ax2.set_title("文本序列化：数字变成 token", fontsize=11, color=INK, pad=8)
ax2.add_patch(patches.FancyBboxPatch((0.4, 3.3), 9.2, 1.1, boxstyle="round,pad=0.05",
                                     fc="#ffffff", ec=C_OP, lw=1.6))
ax2.text(5, 4.05, "<traj_start> [(0.4,0.1),(1.2,0.5),…,(6.8,5.9)] <traj_end>",
         fontsize=9.5, color=C_OP, ha="center", va="center", family="monospace")
ax2.text(5, 3.5, "T_traj = Tokenizer(W_ego)：浮点数按字符切分", fontsize=8.8, color=INK, ha="center")
ax2.annotate("", xy=(5, 2.9), xytext=(5, 3.28),
             arrowprops=dict(arrowstyle="->", color=C_WIRE, lw=1.8))
ax2.add_patch(patches.FancyBboxPatch((1.6, 1.15), 6.8, 1.7, boxstyle="round,pad=0.05",
                                     fc=C_OUT_F, ec=C_OUT, lw=1.8))
ax2.text(5, 2.45, "LLM 自回归生成", fontsize=11, color=C_OUT, ha="center", weight="bold")
ax2.text(5, 2.0, "p(wt | w1:t-1, V_env, S_ego, X_dri)\ntemperature = 0（代码确认：确定性解码）",
         fontsize=8.8, color=INK, ha="center")
ax2.text(5, 0.55, "几何动作：规划被翻译成“续写一句话”——\n"
         "下一个 token 是什么，完全由语言模型的 next-token 分布决定；\n"
         "数值精度 = 文本能写出的精度（论文未说明小数位数）",
         fontsize=8.8, color=INK, ha="center",
         bbox=dict(fc="#f5f4ed", ec="#deddd8", boxstyle="round,pad=0.35"))

# ---------- panel 3: decode back ----------
ax3.set_xlim(-1, 11); ax3.set_ylim(-1, 9.8); ax3.set_aspect("equal"); ax3.axis("off")
ax3.set_title("解码：正则解析回数字", fontsize=11, color=INK, pad=8)
ax3.plot(cx, cy, color=C_IN, lw=2.0, ls=(0, (4, 3)), alpha=0.7)
ax3.scatter(wx, wy, s=110, c=C_OUT_F, edgecolors=C_OUT, linewidths=2, zorder=3)
for i, (x, y) in enumerate(zip(wx, wy)):
    ax3.text(x, y+0.55, f"w{i+1}*", fontsize=8, color=C_OUT, ha="center", weight="bold")
ax3.scatter([0], [0], s=160, c="#ffffff", edgecolors=INK, linewidths=2, marker="s", zorder=3)
ax3.text(5.5, 9.15, "W_ego（预测）= Decoder(T_traj)：正则提取浮点数", fontsize=9, color=INK, ha="center",
         bbox=dict(fc="#ffffff", ec=C_OUT, boxstyle="round,pad=0.3"))
ax3.text(8.2, 0.2, "* = 预测值\n代码 trajectory_utils.retrieve_traj()\n用正则从文本中抠出 (x, y) 浮点对",
         fontsize=8.8, color=C_WIRE, ha="center",
         bbox=dict(fc="#ffffff", ec=C_WIRE, alpha=0.9, boxstyle="round,pad=0.3"))

plt.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/075-Agent-opendrivevla/fig-tokenize.png"
plt.savefig(out, dpi=150, facecolor=BG)
print("saved", out)
