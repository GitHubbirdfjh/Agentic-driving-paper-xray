import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Complexity per rollout step (from paper Sec 4.1):
# decoder-only: O(N_A*N_T) + O(N_A*N_R) + O(N_A^2)
# encoder-decoder (e.g. QCNeXt-style): O(N_A^2*N_M) + O(N_A*N_M*N_R)  (+ encoder cost, omitted)
N_R = 500      # road tokens per scene (typical)
N_T = 18       # motion tokens per agent (9s / 0.5s)
N_M = 6        # modalities of the encoder-decoder baseline

NA = np.arange(8, 129, 4)
dec_only = NA * N_T + NA * N_R + NA ** 2
enc_dec = (NA ** 2) * N_M + NA * N_M * N_R

plt.figure(figsize=(8, 4.6))
plt.plot(NA, dec_only / 1e6, label='decoder-only (SMART)', linewidth=2.2)
plt.plot(NA, enc_dec / 1e6, label='encoder-decoder (6 modalities)', linewidth=2.2, linestyle='--')
plt.xlabel('number of agents $N_A$')
plt.ylabel('attention ops per step (millions)')
plt.title('Per-step inference cost: decoder-only vs encoder-decoder\n($N_R$=500 road tokens, $N_M$=6 modalities; encoder cost omitted)')
plt.legend(frameon=True)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig-inference-cost.png', dpi=150)
print('saved fig-inference-cost.png')
