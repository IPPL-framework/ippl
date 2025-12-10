import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load CSV
csv_file = 'landau_results.csv'
df = pd.read_csv(csv_file)

print(f"Loaded {len(df)} rows from {csv_file}")
print(df.head())

# Set up professional style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'legend.frameon': False,
    'figure.facecolor': 'white',
})

# Colors for methods
colors = {
    'standard': '#1f77b4',  # Blue
    'pruned': '#2ca02c',    # Green
}

# Get unique GPU counts
gpu_counts = sorted(df['num_gpus'].unique())

# Key timing columns to plot
timing_columns = [
    ('mainTimer_s', 'Total Time'),
    ('ScatterPIFNUFFT_s', 'Scatter NUFFT'),
    ('GatherPIFNUFFT_s', 'Gather NUFFT'),
    ('scatterKernel_s', 'Scatter Kernel'),
    ('gatherKernel_s', 'Gather Kernel'),
]

# Create figure with subplots for key timings
n_plots = len(timing_columns)
n_cols = 3
n_rows = (n_plots + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
axes = axes.flatten()

for idx, (col_name, col_label) in enumerate(timing_columns):
    ax = axes[idx]

    # Extract data for each variant
    standard_data = df[df['variant'] == 'standard']
    pruned_data = df[df['variant'] == 'pruned']

    # Get values for each GPU count
    standard_times = [standard_data[standard_data['num_gpus'] == g][col_name].values for g in gpu_counts]
    pruned_times = [pruned_data[pruned_data['num_gpus'] == g][col_name].values for g in gpu_counts]

    # Bar positions
    x = np.arange(len(gpu_counts))
    width = 0.35

    # Plot bars (use mean if multiple runs, otherwise single value)
    standard_means = [np.mean(t) if len(t) > 0 else 0 for t in standard_times]
    pruned_means = [np.mean(t) if len(t) > 0 else 0 for t in pruned_times]

    bars1 = ax.bar(x - width/2, standard_means, width, label='Standard', color=colors['standard'], alpha=0.8)
    bars2 = ax.bar(x + width/2, pruned_means, width, label='Pruned', color=colors['pruned'], alpha=0.8)

    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Time (s)')
    ax.set_title(col_label, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gpu_counts)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    if idx == 0:
        ax.legend()

# Hide unused subplots
for idx in range(len(timing_columns), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('landau_damping_benchmark.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('landau_damping_benchmark.pdf', bbox_inches='tight', facecolor='white')
plt.show()

print("Plots saved to landau_damping_benchmark.png and landau_damping_benchmark.pdf")

# --- Speedup plot ---
fig2, ax2 = plt.subplots(figsize=(8, 5))

standard_main = df[df['variant'] == 'standard'].set_index('num_gpus')['mainTimer_s']
pruned_main = df[df['variant'] == 'pruned'].set_index('num_gpus')['mainTimer_s']

speedups = standard_main / pruned_main

ax2.bar(range(len(speedups)), speedups.values, color=colors['pruned'], alpha=0.8)
ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1)
ax2.set_xlabel('Number of GPUs')
ax2.set_ylabel('Speedup (Standard / Pruned)')
ax2.set_title('Pruned FFT Speedup over Standard FFT', fontweight='bold')
ax2.set_xticks(range(len(speedups)))
ax2.set_xticklabels(speedups.index)
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

# Add speedup values on top of bars
for i, (gpu, speedup) in enumerate(speedups.items()):
    ax2.annotate(f'{speedup:.2f}x', xy=(i, speedup), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('landau_damping_speedup.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('landau_damping_speedup.pdf', bbox_inches='tight', facecolor='white')
plt.show()

print("Speedup plot saved to landau_damping_speedup.png and landau_damping_speedup.pdf")

# --- Print summary table ---
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(f"{'GPUs':<8} {'Standard (s)':<15} {'Pruned (s)':<15} {'Speedup':<10}")
print("-"*60)
for gpu in gpu_counts:
    std_time = df[(df['variant'] == 'standard') & (df['num_gpus'] == gpu)]['mainTimer_s'].values
    pru_time = df[(df['variant'] == 'pruned') & (df['num_gpus'] == gpu)]['mainTimer_s'].values
    if len(std_time) > 0 and len(pru_time) > 0:
        speedup = std_time[0] / pru_time[0]
        print(f"{gpu:<8} {std_time[0]:<15.4f} {pru_time[0]:<15.4f} {speedup:<10.2f}x")
print("="*60)