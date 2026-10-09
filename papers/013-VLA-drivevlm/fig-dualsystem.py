# -*- coding: utf-8 -*-
"""DriveVLM-Dual 快慢双系统时间线示意。

VLM 分支低频(论文实测 OrinX 上约 410ms ≈ 2.4Hz)输出 W_slow;
传统规划器分支高频(几十 Hz)运行, 异步地把最新的 W_slow 当作
参考轨迹/优化初值做精修, 得到 W_fast = Planner([W_slow, f])。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(12, 4.2))
T = 2.5  # 画 2.5 秒

# VLM 分支: 410ms 一次
vlm_times = np.arange(0, T, 0.41)
# 规划器分支: 50ms 一次(示意)
plan_times = np.arange(0, T, 0.05)

ax.eventplot(vlm_times, lineoffsets=1.6, linelengths=0.55, linewidths=3,
             colors="#6a55c2", label="VLM 分支输出 $W_{slow}$ (约 410ms/次, 低频)")
ax.eventplot(plan_times, lineoffsets=0.6, linelengths=0.4, linewidths=1.6,
             colors="#0b7a5c", label="传统规划器输出 $W_{fast}$ (约 50ms/次, 高频)")

# 异步引用: 几个 W_slow 被后续的 planner 周期引用
for t in [0.0, 0.82, 1.64]:
    ax.annotate("", xy=(t + 0.30, 0.82), xytext=(t, 1.32),
                arrowprops=dict(arrowstyle="->", color="#6a55c2", lw=1.6, ls="--"))
ax.text(1.15, 1.15, "规划器异步取用最新的 $W_{slow}$\n作为参考轨迹/优化初值", fontsize=10,
        color="#6a55c2", ha="center",
        bbox=dict(boxstyle="round", fc="#ebe7f8", ec="#6a55c2"))

ax.set_xlim(-0.05, T + 0.05)
ax.set_ylim(0.1, 2.1)
ax.set_yticks([0.6, 1.6])
ax.set_yticklabels(["传统规划器\n(快思考)", "VLM 分支\n(慢思考)"], fontsize=11)
ax.set_xlabel("时间 (秒)", fontsize=12)
ax.set_title("DriveVLM-Dual: 两个分支异步运行, $W_{fast} = Planner([W_{slow}, f])$",
             fontsize=14)
ax.legend(fontsize=10.5, loc="upper right")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/013-VLA-drivevlm/fig-dualsystem.png",
            dpi=150, bbox_inches="tight")
print("saved fig-dualsystem.png")
