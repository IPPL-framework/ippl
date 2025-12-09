import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV
df = pd.read_csv('fft_timings_all.csv')

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

# Colors for concurrent FFT options
colors = {1: '#1f77b4', 2: '#ff7f0e', 4: '#2ca02c', 8: '#d62728'}
markers = {1: 'o', 2: 's', 4: '^', 8: 'D'}

# Offset multipliers for each concurrent FFT (in log space)
offsets = {1: 0.92, 2: 0.97, 4: 1.03, 8: 1.08}

# Get unique values
gpu_counts = df['num_gpus'].unique()
concurrent_ffts = df['num_concurrent_ffts'].unique()

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# --- Plot 1: Forward FFT ---
ax1 = axes[0, 0]

# Full FFT (average across concurrent_ffts since it's the same)
full_fwd = df.groupby('num_gpus').agg({
    'fwd_full_mean_ms': 'mean',
    'fwd_full_min_ms': 'mean',
    'fwd_full_max_ms': 'mean'
}).reset_index()

yerr_full = [
    full_fwd['fwd_full_mean_ms'] - full_fwd['fwd_full_min_ms'],
    full_fwd['fwd_full_max_ms'] - full_fwd['fwd_full_mean_ms']
]
ax1.errorbar(full_fwd['num_gpus'] * 0.87, full_fwd['fwd_full_mean_ms'], yerr=yerr_full,
             fmt='k-', linewidth=2, markersize=8, marker='o', capsize=4,
             label='Full FFT', zorder=10)

# Pruned FFT for each concurrent option
for n_conc in concurrent_ffts:
    subset = df[df['num_concurrent_ffts'] == n_conc]
    yerr = [
        subset['fwd_pruned_mean_ms'] - subset['fwd_pruned_min_ms'],
        subset['fwd_pruned_max_ms'] - subset['fwd_pruned_mean_ms']
    ]
    x_offset = subset['num_gpus'] * offsets[n_conc]
    ax1.errorbar(x_offset, subset['fwd_pruned_mean_ms'], yerr=yerr,
                 fmt='-', color=colors[n_conc], marker=markers[n_conc],
                 linewidth=1.5, markersize=7, capsize=3,
                 label=f'Pruned (n={n_conc})')

ax1.set_xlabel('Number of GPUs')
ax1.set_ylabel('Time (ms)')
ax1.set_title('Forward FFT', fontweight='bold')
ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.set_xticks(gpu_counts)
ax1.set_xticklabels(gpu_counts)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3, linestyle='--')

# --- Plot 2: Backward FFT ---
ax2 = axes[0, 1]

# Full FFT
full_bwd = df.groupby('num_gpus').agg({
    'bwd_full_mean_ms': 'mean',
    'bwd_full_min_ms': 'mean',
    'bwd_full_max_ms': 'mean'
}).reset_index()

yerr_full = [
    full_bwd['bwd_full_mean_ms'] - full_bwd['bwd_full_min_ms'],
    full_bwd['bwd_full_max_ms'] - full_bwd['bwd_full_mean_ms']
]
ax2.errorbar(full_bwd['num_gpus'] * 0.87, full_bwd['bwd_full_mean_ms'], yerr=yerr_full,
             fmt='k-', linewidth=2, markersize=8, marker='o', capsize=4,
             label='Full FFT', zorder=10)

# Pruned FFT for each concurrent option
for n_conc in concurrent_ffts:
    subset = df[df['num_concurrent_ffts'] == n_conc]
    yerr = [
        subset['bwd_pruned_mean_ms'] - subset['bwd_pruned_min_ms'],
        subset['bwd_pruned_max_ms'] - subset['bwd_pruned_mean_ms']
    ]
    x_offset = subset['num_gpus'] * offsets[n_conc]
    ax2.errorbar(x_offset, subset['bwd_pruned_mean_ms'], yerr=yerr,
                 fmt='-', color=colors[n_conc], marker=markers[n_conc],
                 linewidth=1.5, markersize=7, capsize=3,
                 label=f'Pruned (n={n_conc})')

ax2.set_xlabel('Number of GPUs')
ax2.set_ylabel('Time (ms)')
ax2.set_title('Backward FFT', fontweight='bold')
ax2.set_xscale('log', base=2)
ax2.set_yscale('log')
ax2.set_xticks(gpu_counts)
ax2.set_xticklabels(gpu_counts)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3, linestyle='--')

# --- Plot 3: Speedup (Full / Pruned) ---
ax3 = axes[1, 0]

for n_conc in concurrent_ffts:
    subset = df[df['num_concurrent_ffts'] == n_conc]
    # Calculate roundtrip speedup
    full_time = subset['fwd_full_mean_ms'] + subset['bwd_full_mean_ms']
    pruned_time = subset['fwd_pruned_mean_ms'] + subset['bwd_pruned_mean_ms']
    speedup = full_time / pruned_time

    x_offset = subset['num_gpus'] * offsets[n_conc]
    ax3.plot(x_offset, speedup,
             color=colors[n_conc], marker=markers[n_conc],
             linewidth=1.5, markersize=7,
             label=f'n={n_conc}')

ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
ax3.set_xlabel('Number of GPUs')
ax3.set_ylabel('Speedup (Full / Pruned)')
ax3.set_title('Roundtrip Speedup', fontweight='bold')
ax3.set_xscale('log', base=2)
ax3.set_xticks(gpu_counts)
ax3.set_xticklabels(gpu_counts)
ax3.legend(title='Concurrent FFTs', loc='best')
ax3.grid(True, alpha=0.3, linestyle='--')

# --- Plot 4: Memory Usage ---
ax4 = axes[1, 1]

# Full FFT memory (same across concurrent)
full_mem = df.groupby('num_gpus')['fwd_full_mem_mb'].mean().reset_index()
ax4.plot(full_mem['num_gpus'] * 0.87, full_mem['fwd_full_mem_mb'],
         'k-', linewidth=2, markersize=8, marker='o',
         label='Full FFT', zorder=10)

# Pruned FFT memory for each concurrent option
for n_conc in concurrent_ffts:
    subset = df[df['num_concurrent_ffts'] == n_conc]
    x_offset = subset['num_gpus'] * offsets[n_conc]
    ax4.plot(x_offset, subset['fwd_pruned_mem_mb'],
             color=colors[n_conc], marker=markers[n_conc],
             linewidth=1.5, markersize=7,
             label=f'Pruned (n={n_conc})')

ax4.set_xlabel('Number of GPUs')
ax4.set_ylabel('Memory per GPU (MB)')
ax4.set_title('GPU Memory Usage', fontweight='bold')
ax4.set_xscale('log', base=2)
ax4.set_yscale('log')
ax4.set_xticks(gpu_counts)
ax4.set_xticklabels(gpu_counts)
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('benchmark_scaling.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('benchmark_scaling.pdf', bbox_inches='tight', facecolor='white')
plt.show()

print("Plots saved to benchmark_scaling.png and benchmark_scaling.pdf")