# -*- coding: utf-8 -*-
"""PreWorld: 代码里藏着的课程学习 (curriculum), 论文正文没有提。

出自官方仓库 mmdet3d/models/detectors/preworld_temporal_traj.py 的 forward_train,
由 CustomSetEpochInfoHook 每个 epoch 设置 self.curr_epoch:

  预训练 (if_render=True):
      epoch <= 2 : future_intervals = [0, 1]          # 只监督未来第 1 步
      epoch >= 3 : range(0, min(epoch - 1, 6))        # 逐步放开到未来第 6 步
  微调 (if_render=False):
      epoch <= 4 : future_intervals = [0, 1]
      epoch >= 5 : range(0, min((epoch - 3) // 2 + 1, 6))

即: 训练早期只让递归预测走 1 步 (监督信号最可靠), 随着 epoch 增加才逐
步解锁更远的未来帧 (最多 6 步 = 未来 3 秒, nuScenes 关键帧 2Hz)。
这是一个标准的"先学近、再学远"的课程, 但论文第 4.1 节的 Implementation
Details 只写了 epoch 数, 没有提这件事。

注意: 时序预训练配置 max_epochs=9, 论文正文写的是 8 个 epoch, 差 1。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

def pretrain_maxint(ep):
    # if_render=True 分支
    if ep <= 2:
        return 1
    return min(ep - 1, 6)

def finetune_maxint(ep):
    # if_render=False 分支
    if ep <= 4:
        return 1
    return min((ep - 3) // 2 + 1, 6)

ep_pre = np.arange(1, 10)    # 时序预训练配置 max_epochs=9
ep_ft = np.arange(1, 19)     # 时序微调配置 max_epochs=18

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)

axes[0].step(ep_pre, [pretrain_maxint(e) for e in ep_pre], where="mid",
             color="#1a6fa3", lw=2.6)
axes[0].plot(ep_pre, [pretrain_maxint(e) for e in ep_pre], "o",
             color="#1a6fa3", ms=6)
axes[0].set_title("Pre-training (if_render=True), 9 epochs", fontsize=12)
axes[0].set_xlabel("epoch", fontsize=11)
axes[0].annotate("first 2 epochs:\nonly 1 future step\nsupervised",
                 xy=(1.5, 1.05), xytext=(3.2, 2.6),
                 arrowprops=dict(arrowstyle="->", color="#1a6fa3", lw=1.3),
                 fontsize=10, color="#1a6fa3",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1a6fa3"))

axes[1].step(ep_ft, [finetune_maxint(e) for e in ep_ft], where="mid",
             color="#b34a06", lw=2.6)
axes[1].plot(ep_ft, [finetune_maxint(e) for e in ep_ft], "o",
             color="#b34a06", ms=6)
axes[1].set_title("Fine-tuning (if_render=False), 18 epochs", fontsize=12)
axes[1].set_xlabel("epoch", fontsize=11)
axes[1].annotate("first 4 epochs:\nonly 1 future step\nsupervised",
                 xy=(2.5, 1.05), xytext=(8, 2.8),
                 arrowprops=dict(arrowstyle="->", color="#b34a06", lw=1.3),
                 fontsize=10, color="#b34a06",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#b34a06"))

for ax in axes:
    ax.set_ylabel("max supervised future interval\n(steps; 2 steps = 1 s)",
                  fontsize=11)
    ax.set_ylim(0.4, 6.8)
    ax.set_yticks(range(1, 7))
    ax.grid(alpha=0.25, axis="y")

fig.suptitle("Hidden curriculum in the official code: near future first, far future later "
             "(not mentioned in the paper)",
             fontsize=13, y=1.02)
fig.text(0.5, -0.03,
         "From preworld_temporal_traj.py forward_train, driven by CustomSetEpochInfoHook; "
         "x-axis epochs follow the repo configs (pretrain 9, finetune 18).",
         ha="center", fontsize=9, style="italic", color="#6e737b")
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/053-WM-preworld/fig-curriculum.png",
            dpi=150, bbox_inches="tight")
print("saved fig-curriculum.png")
