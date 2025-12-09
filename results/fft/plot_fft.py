import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV
df = pd.read_csv('fft_timings_1_concurrent.csv')

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Forward FFT timing
ax1 = axes[0]
ax1.plot(df['num_gpus'], df['fwd_full_mean_ms'], 'o-', label='Full FFT', linewidth=2, markersize=8)
ax1.plot(df['num_gpus'], df['fwd_pruned_mean_ms'], 's-', label='Pruned FFT', linewidth=2, markersize=8)
ax1.set_xlabel('Number of GPUs')
ax1.set_ylabel('Time (ms)')
ax1.set_title('Forward FFT')
ax1.set_xscale('log', base=2)
ax1.set_xticks(df['num_gpus'])
ax1.set_xticklabels(df['num_gpus'])
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Backward FFT timing
ax2 = axes[1]
ax2.plot(df['num_gpus'], df['bwd_full_mean_ms'], 'o-', label='Full FFT', linewidth=2, markersize=8)
ax2.plot(df['num_gpus'], df['bwd_pruned_mean_ms'], 's-', label='Pruned FFT', linewidth=2, markersize=8)
ax2.set_xlabel('Number of GPUs')
ax2.set_ylabel('Time (ms)')
ax2.set_title('Backward FFT')
ax2.set_xscale('log', base=2)
ax2.set_xticks(df['num_gpus'])
ax2.set_xticklabels(df['num_gpus'])
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('benchmark_scaling.png', dpi=150, bbox_inches='tight')
plt.show()

print("Plot saved to benchmark_scaling.png")