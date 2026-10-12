import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 表3 消融：各阶段 PDMS 增益与子指标变化
stages = ['traj only\n(ID1)', '+ driving\npretrain (ID2)', '+ diffusion\nplanner (ID3)', '+ DiffGRPO\n(ID4)']
pdms = [82.4, 84.1, 86.5, 90.8]
nc  = [97.4, 97.6, 98.1, 97.9]
dac = [91.3, 93.1, 94.7, 97.3]
ttc = [93.0, 92.7, 94.2, 94.9]
ep  = [77.2, 79.1, 80.9, 87.3]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
x = np.arange(len(stages))
ax1.bar(x, pdms, color=['#999999', '#6aaed6', '#4292c6', '#08519c'], edgecolor='white')
for i, v in enumerate(pdms):
    ax1.text(i, v + 0.4, f'{v}', ha='center', fontsize=11, fontweight='bold')
    if i > 0:
        ax1.annotate('', xy=(i, pdms[i] - 0.6), xytext=(i - 1, pdms[i - 1] + 0.6),
                     arrowprops=dict(arrowstyle='->', color='crimson', lw=1.6))
        ax1.text(i - 0.5, (pdms[i] + pdms[i - 1]) / 2 + 0.9, f'+{pdms[i]-pdms[i-1]:.1f}',
                 ha='center', fontsize=10, color='crimson', fontweight='bold')
ax1.set_xticks(x); ax1.set_xticklabels(stages, fontsize=9)
ax1.set_ylabel('PDMS'); ax1.set_title('NAVSIM PDMS by stage (Tab.3)')
ax1.set_ylim(78, 94)

w = 0.18
for j, (vals, c, lbl) in enumerate(zip([nc, dac, ttc, ep], ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'], ['NC', 'DAC', 'TTC', 'EP'])):
    ax2.bar(x + (j - 1.5) * w, vals, width=w, color=c, label=lbl)
ax2.set_xticks(x); ax2.set_xticklabels(stages, fontsize=9)
ax2.set_ylabel('sub-metric'); ax2.legend(fontsize=9, ncol=4)
ax2.set_title('Sub-metrics: RL stage is EP-driven (+6.4)')
ax2.set_ylim(70, 102)
fig.tight_layout()
fig.savefig('fig-ablation-waterfall.png', dpi=150)
print('saved fig-ablation-waterfall.png')
