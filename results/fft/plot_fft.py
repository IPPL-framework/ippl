import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load CSVs from different folders, detecting grid size from folder name or data
folders = sorted(Path('.').glob('benchmark_results_*'))

dfs_by_grid = {}
for folder in folders:
    for csv_file in folder.glob('*.csv'):
        df_temp = pd.read_csv(csv_file)
        # Try to detect grid size from the data or folder name
        if 'grid_size' in df_temp.columns:
            grid_size = df_temp['grid_size'].iloc[0]
        else:
            # Try to extract from folder name (e.g., benchmark_results_128, benchmark_results_256)
            try:
                grid_size = int(folder.name.split('_')[-1])
            except ValueError:
                # Default naming - assign based on folder order
                grid_size = folder.name

        if grid_size not in dfs_by_grid:
            dfs_by_grid[grid_size] = []
        dfs_by_grid[grid_size].append(df_temp)

# Combine dataframes for each grid size
grid_dfs = {}
for grid_size, dfs in dfs_by_grid.items():
    grid_dfs[grid_size] = pd.concat(dfs, ignore_index=True)

grid_sizes = sorted(grid_dfs.keys())
print(f"Found grid sizes: {grid_sizes}")

# Set up compact professional style for dual-column papers
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'legend.frameon': False,
    'figure.facecolor': 'white',
    'figure.dpi': 150,
})

# Colors for methods - using different shades for different grid sizes
color_palettes = {
    0: {'full': '#333333', 'pruned': '#2ca02c'},  # Dark gray and green
    1: {'full': '#1f77b4', 'pruned': '#ff7f0e'},  # Blue and orange
}

# Get unique values from all dataframes
all_gpu_counts = set()
all_concurrent_ffts = set()
for df in grid_dfs.values():
    all_gpu_counts.update(df['num_gpus'].unique())
    all_concurrent_ffts.update(df['num_concurrent'].unique())

gpu_counts = sorted(all_gpu_counts)

# Filter to only n_conc = 1, 4, 8
target_concurrent = [1, 4, 8]
concurrent_ffts = [n for n in sorted(all_concurrent_ffts) if n in target_concurrent]
print(f"Plotting concurrent FFT values: {concurrent_ffts}")

# Create compact figure for dual-column paper
# Typical dual-column width: ~7 inches total, single column ~3.5 inches
# Using full width with compact layout
fig, axes = plt.subplots(2, len(concurrent_ffts), figsize=(7, 3.2))

# Handle case where there's only one concurrent value
if len(concurrent_ffts) == 1:
    axes = axes.reshape(2, 1)

for col, n_conc in enumerate(concurrent_ffts):
    # --- Forward FFT (top row) ---
    ax_fwd = axes[0, col]

    fwd_data = []
    positions = []
    violin_colors = []

    for i, num_gpu in enumerate(gpu_counts):
        for g_idx, grid_size in enumerate(grid_sizes):
            if grid_size not in grid_dfs:
                continue
            df = grid_dfs[grid_size]
            subset = df[(df['num_concurrent'] == n_conc) & (df['num_gpus'] == num_gpu)]

            colors = color_palettes.get(g_idx, color_palettes[0])

            # Full FFT data
            full_data = subset[subset['method'] == 'full']
            full_fwd = full_data[full_data['direction'] == 'forward']['time_ms'].values
            if len(full_fwd) > 0:
                fwd_data.append(full_fwd)
                offset = -0.3 + g_idx * 0.3 - 0.1
                positions.append(i + offset)
                violin_colors.append(colors['full'])

            # Pruned FFT data
            pruned_data = subset[subset['method'] == 'pruned']
            pruned_fwd = pruned_data[pruned_data['direction'] == 'forward']['time_ms'].values
            if len(pruned_fwd) > 0:
                fwd_data.append(pruned_fwd)
                offset = -0.3 + g_idx * 0.3 + 0.1
                positions.append(i + offset)
                violin_colors.append(colors['pruned'])

    if fwd_data:
        vp = ax_fwd.violinplot(fwd_data, positions=positions, widths=0.15, showmeans=True, showmedians=False)
        for j, body in enumerate(vp['bodies']):
            body.set_facecolor(violin_colors[j])
            body.set_alpha(0.7)
        for partname in ['cmeans', 'cmins', 'cmaxes', 'cbars']:
            if partname in vp:
                vp[partname].set_color('black')
                vp[partname].set_linewidth(0.8)

    ax_fwd.set_xticks(range(len(gpu_counts)))
    ax_fwd.set_xticklabels(gpu_counts)
    if col == 0:
        ax_fwd.set_ylabel('Time (ms)')
    ax_fwd.set_title(f'Forward ($\\mathbf{{n_{{conc}}}}$={n_conc})', fontweight='bold', pad=3)
    ax_fwd.set_yscale('log')
    ax_fwd.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=0.5)

    # --- Backward FFT (bottom row) ---
    ax_bwd = axes[1, col]

    bwd_data = []
    positions = []
    violin_colors = []

    for i, num_gpu in enumerate(gpu_counts):
        for g_idx, grid_size in enumerate(grid_sizes):
            if grid_size not in grid_dfs:
                continue
            df = grid_dfs[grid_size]
            subset = df[(df['num_concurrent'] == n_conc) & (df['num_gpus'] == num_gpu)]

            colors = color_palettes.get(g_idx, color_palettes[0])

            # Full FFT data
            full_data = subset[subset['method'] == 'full']
            full_bwd = full_data[full_data['direction'] == 'backward']['time_ms'].values
            if len(full_bwd) > 0:
                bwd_data.append(full_bwd)
                offset = -0.3 + g_idx * 0.3 - 0.1
                positions.append(i + offset)
                violin_colors.append(colors['full'])

            # Pruned FFT data
            pruned_data = subset[subset['method'] == 'pruned']
            pruned_bwd = pruned_data[pruned_data['direction'] == 'backward']['time_ms'].values
            if len(pruned_bwd) > 0:
                bwd_data.append(pruned_bwd)
                offset = -0.3 + g_idx * 0.3 + 0.1
                positions.append(i + offset)
                violin_colors.append(colors['pruned'])

    if bwd_data:
        vp = ax_bwd.violinplot(bwd_data, positions=positions, widths=0.15, showmeans=True, showmedians=False)
        for j, body in enumerate(vp['bodies']):
            body.set_facecolor(violin_colors[j])
            body.set_alpha(0.7)
        for partname in ['cmeans', 'cmins', 'cmaxes', 'cbars']:
            if partname in vp:
                vp[partname].set_color('black')
                vp[partname].set_linewidth(0.8)

    ax_bwd.set_xticks(range(len(gpu_counts)))
    ax_bwd.set_xticklabels(gpu_counts)
    ax_bwd.set_xlabel('# GPUs')
    if col == 0:
        ax_bwd.set_ylabel('Time (ms)')
    ax_fwd.set_title(f'Backward ($\\mathbf{{n_{{conc}}}}$={n_conc})', fontweight='bold', pad=3)
    ax_bwd.set_yscale('log')
    ax_bwd.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=0.5)

# Add compact legend with grid size info
from matplotlib.patches import Patch
legend_elements = []
for g_idx, grid_size in enumerate(grid_sizes):
    colors = color_palettes.get(g_idx, color_palettes[0])
    legend_elements.append(Patch(facecolor=colors['full'], alpha=0.7, label=f'Full (${grid_size}^3$)'))
    legend_elements.append(Patch(facecolor=colors['pruned'], alpha=0.7, label=f'Pruned (${grid_size}^3$)'))

fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements),
           bbox_to_anchor=(0.5, 1.02), fontsize=7, handlelength=1.2, handletextpad=0.4,
           columnspacing=1.0)

plt.tight_layout()
plt.subplots_adjust(top=0.88, hspace=0.35, wspace=0.15)
plt.savefig('benchmark_violin_compact.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('benchmark_violin_compact.pdf', bbox_inches='tight', facecolor='white')
plt.show()

print("Plots saved to benchmark_violin_compact.png and benchmark_violin_compact.pdf")