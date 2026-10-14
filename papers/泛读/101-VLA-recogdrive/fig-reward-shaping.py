import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 训练奖励 vs 评估指标：progress 权重 10 vs 5
# 官方 PDMS = NC * DAC * (5*EP + 5*TTC + 2*C)/12
# 训练 scorer: progress_weight=10, ttc=5, comfort=2
labels = ['EP\n(progress)', 'TTC', 'Comfort']
eval_w = [5, 5, 2]
train_w = [10, 5, 2]

x = np.arange(len(labels))
w = 0.32
fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.bar(x - w/2, eval_w, width=w, color='#08519c', label='eval PDMS (Eq.13)')
ax.bar(x + w/2, train_w, width=w, color='#e34a33', label='RL training reward (code)')
for i, (e, t) in enumerate(zip(eval_w, train_w)):
    ax.text(i - w/2, e + 0.25, str(e), ha='center', fontweight='bold')
    ax.text(i + w/2, t + 0.25, str(t), ha='center', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('weight in score')
ax.set_title('Training reward != eval metric: progress is doubled during RL')
ax.legend()
ax.set_ylim(0, 12)
fig.tight_layout()
fig.savefig('fig-reward-shaping.png', dpi=150)
print('saved fig-reward-shaping.png')
