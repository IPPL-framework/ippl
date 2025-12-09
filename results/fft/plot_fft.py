import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load all CSVs from the output directory
dfs = []
for csv_file in sorted(Path('.').glob('benchmark_results_*/*.csv')):
    dfs.append(pd.read_csv(csv_file))

df = pd.concat(dfs, ignore_index=True)

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
colors = {'full': '#333333', 'pruned': '#2ca02c'}

# Get unique values
gpu_counts = sorted(df['num_gpus'].unique())
concurrent_ffts = sorted(df['num_concurrent'].unique())

# Create figure with 2x4 subplots (forward and backward for each concurrent value)
fig, axes = plt.subplots(2, len(concurrent_ffts), figsize=(4 * len(concurrent_ffts), 8))

for col, n_conc in enumerate(concurrent_ffts):
    subset = df[df['num_concurrent'] == n_conc]

    # --- Forward FFT (top row) ---
    ax_fwd = axes[0, col]

    # Prepare data for violin plot
    fwd_data = []
    positions = []
    violin_colors = []

    for i, num_gpu in enumerate(gpu_counts):
        gpu_subset = subset[subset['num_gpus'] == num_gpu]

        # Full FFT data
        full_data = gpu_subset[gpu_subset['method'] == 'full']
        full_fwd = full_data[full_data['direction'] == 'forward']['time_ms'].values
        if len(full_fwd) > 0:
            fwd_data.append(full_fwd)
            positions.append(i - 0.2)
            violin_colors.append(colors['full'])

        # Pruned FFT data
        pruned_data = gpu_subset[gpu_subset['method'] == 'pruned']
        pruned_fwd = pruned_data[pruned_data['direction'] == 'forward']['time_ms'].values
        if len(pruned_fwd) > 0:
            fwd_data.append(pruned_fwd)
            positions.append(i + 0.2)
            violin_colors.append(colors['pruned'])

    if fwd_data:
        vp = ax_fwd.violinplot(fwd_data, positions=positions, widths=0.35, showmeans=True, showmedians=False)
        for j, body in enumerate(vp['bodies']):
            body.set_facecolor(violin_colors[j])
            body.set_alpha(0.7)
        for partname in ['cmeans', 'cmins', 'cmaxes', 'cbars']:
            if partname in vp:
                vp[partname].set_color('black')
                vp[partname].set_linewidth(1)

    ax_fwd.set_xticks(range(len(gpu_counts)))
    ax_fwd.set_xticklabels(gpu_counts)
    ax_fwd.set_xlabel('Number of GPUs')
    ax_fwd.set_ylabel('Time (ms)')
    ax_fwd.set_title(f'Forward FFT (n_conc={n_conc})', fontweight='bold')
    ax_fwd.set_yscale('log')
    ax_fwd.grid(True, alpha=0.3, linestyle='--', axis='y')

    # --- Backward FFT (bottom row) ---
    ax_bwd = axes[1, col]

    bwd_data = []
    positions = []
    violin_colors = []

    for i, num_gpu in enumerate(gpu_counts):
        gpu_subset = subset[subset['num_gpus'] == num_gpu]

        # Full FFT data
        full_data = gpu_subset[gpu_subset['method'] == 'full']
        full_bwd = full_data[full_data['direction'] == 'backward']['time_ms'].values
        if len(full_bwd) > 0:
            bwd_data.append(full_bwd)
            positions.append(i - 0.2)
            violin_colors.append(colors['full'])

        # Pruned FFT data
        pruned_data = gpu_subset[gpu_subset['method'] == 'pruned']
        pruned_bwd = pruned_data[pruned_data['direction'] == 'backward']['time_ms'].values
        if len(pruned_bwd) > 0:
            bwd_data.append(pruned_bwd)
            positions.append(i + 0.2)
            violin_colors.append(colors['pruned'])

    if bwd_data:
        vp = ax_bwd.violinplot(bwd_data, positions=positions, widths=0.35, showmeans=True, showmedians=False)
        for j, body in enumerate(vp['bodies']):
            body.set_facecolor(violin_colors[j])
            body.set_alpha(0.7)
        for partname in ['cmeans', 'cmins', 'cmaxes', 'cbars']:
            if partname in vp:
                vp[partname].set_color('black')
                vp[partname].set_linewidth(1)

    ax_bwd.set_xticks(range(len(gpu_counts)))
    ax_bwd.set_xticklabels(gpu_counts)
    ax_bwd.set_xlabel('Number of GPUs')
    ax_bwd.set_ylabel('Time (ms)')
    ax_bwd.set_title(f'Backward FFT (n_conc={n_conc})', fontweight='bold')
    ax_bwd.set_yscale('log')
    ax_bwd.grid(True, alpha=0.3, linestyle='--', axis='y')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors['full'], alpha=0.7, label='Full FFT'),
    Patch(facecolor=colors['pruned'], alpha=0.7, label='Pruned FFT')
]
fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig('benchmark_violin.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('benchmark_violin.pdf', bbox_inches='tight', facecolor='white')
plt.show()

print("Plots saved to benchmark_violin.png and benchmark_violin.pdf")