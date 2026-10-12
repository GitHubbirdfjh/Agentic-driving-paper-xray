import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# DiffGRPO 中 gamma=0.6 对去噪步的加权：越靠近干净轨迹的步权重越大
# 代码：discount = gamma ** (num_steps - indices - 1)，indices=0 是从纯噪声出发的第一步
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

for ax, T, title in zip(axes, [5, 100], ['T=5 (inference steps)', 'T=100 (training steps)']):
    idx = np.arange(T)
    w = 0.6 ** (T - idx - 1)
    ax.bar(idx, w, color='#2a7fbf', edgecolor='white')
    ax.set_xlabel('denoising step index (0 = from pure noise, T-1 = to clean traj)')
    ax.set_ylabel('weight')
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    # 标出有效步数：权重>0.01 的步
    eff = int(np.sum(w > 0.01))
    ax.text(0.97, 0.93, f'steps with w>0.01: {eff}/{T}', ha='right', va='top',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', fc='wheat', alpha=0.6))

fig.suptitle('DiffGRPO discount weights  gamma=0.6  (later denoising steps dominate)', fontsize=12)
fig.tight_layout()
fig.savefig('fig-diffgrpo-weights.png', dpi=150)
print('saved fig-diffgrpo-weights.png')
