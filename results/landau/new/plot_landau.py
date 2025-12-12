#!/usr/bin/env python3
"""
Plot strong scaling results for LandauDampingPIF simulations.

Reads timing data from files like:
  LandauDampingPIF128_0.csv
  LandauDampingPIFPruned64_1.csv

Input CSV format:
  timer_name,rank,measurement_id,duration_seconds

Usage:
    python plot_landau_scaling.py -d benchmark_results/ -o fig_scaling
    python plot_landau_scaling.py -d benchmark_results/ -o fig_scaling --all
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import argparse
import glob
import re


def setup_style():
    """Set up publication-quality matplotlib style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 9,
        'mathtext.fontset': 'stix',
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'legend.fontsize': 8,
        'legend.frameon': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'lines.linewidth': 1.2,
        'lines.markersize': 5,
    })


# Colorblind-friendly palette (Okabe-Ito)
VERSION_COLORS = {
    'PIF': '#0072B2',        # Blue
    'PIFPruned': '#E69F00',  # Orange
}

VERSION_MARKERS = {
    'PIF': 'o',
    'PIFPruned': 's',
}

VERSION_LABELS = {
    'PIF': 'Standard',
    'PIFPruned': 'Pruned',
}

# NUFFT sub-component timers
TIMER_CATEGORIES_GATHER = {
    'PrecorrectionNUFFT2': 'Precorrection',
    'FFTNUFFT2': 'FFT',
    'FillHaloNUFFT2': 'Halo Exchange',
    'GatherNUFFT2': 'Interpolation',
}

TIMER_CATEGORIES_SCATTER = {
    'scatterTimerNUFFT1': 'Spreading',
    'accumulateHaloNUFFT1': 'Halo Exchange',
    'FFTNUFFT1': 'FFT',
    'deconvolutionNUFFT1': 'Deconvolution',
}

# Refined color palette for components (colorblind-friendly, visually distinct)
COMPONENT_COLORS = {
    # Scatter (Type-1) components
    'Spreading': '#4477AA',        # Blue
    'Deconvolution': '#EE6677',    # Red/Pink
    # Gather (Type-2) components
    'Interpolation': '#228833',    # Green
    'Precorrection': '#CCBB44',    # Yellow
    # Shared components
    'FFT': '#66CCEE',              # Cyan
    'Halo Exchange': '#AA3377',    # Purple
    # Other
    'Other': '#BBBBBB',            # Gray
    'updateParticle': '#999999',   # Dark gray
}

# Stack order for Scatter (Type-1): bottom to top
SCATTER_STACK_ORDER = ['Spreading', 'Halo Exchange', 'FFT', 'Deconvolution']

# Stack order for Gather (Type-2): bottom to top
GATHER_STACK_ORDER = ['Precorrection', 'FFT', 'Halo Exchange', 'Interpolation']

# High-level components
HIGH_LEVEL_COMPONENTS = ['updateParticle', 'GatherPIFNUFFT', 'ScatterPIFNUFFT']

# Component colors for high-level view
HIGH_LEVEL_COLORS = {
    'updateParticle': '#4477AA',      # Blue
    'GatherPIFNUFFT': '#228833',      # Green
    'ScatterPIFNUFFT': '#EE6677',     # Red
    'Other': '#BBBBBB',               # Gray
}

HIGH_LEVEL_LABELS = {
    'updateParticle': 'Particle Push',
    'GatherPIFNUFFT': 'Gather (NUFFT)',
    'ScatterPIFNUFFT': 'Scatter (NUFFT)',
    'Other': 'Other',
}

# Number of outliers to discard per group
N_OUTLIERS_DISCARD = 5


def parse_filename(filepath):
    """
    Parse filename to extract version and GPU count.

    Examples:
        LandauDampingPIF128_0.csv -> ('PIF', 128)
        LandauDampingPIFPruned64_1.csv -> ('PIFPruned', 64)
    """
    filename = Path(filepath).stem

    match = re.match(r'LandauDampingPIF(Pruned)?(\d+)_(\d+)', filename)

    if match:
        is_pruned = match.group(1) is not None
        num_gpus = int(match.group(2))
        version = 'PIFPruned' if is_pruned else 'PIF'
        return version, num_gpus

    return None, None


def load_timing_data(base_dir):
    """Load all timing CSVs from a directory."""
    base_path = Path(base_dir)

    all_data = []

    patterns = [
        'LandauDampingPIF*_*.csv',
        'LandauDampingPIFPruned*_*.csv',
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(str(base_path / pattern)))

    if not files:
        files = glob.glob(str(base_path / '*.csv'))

    for filepath in files:
        version, num_gpus = parse_filename(filepath)

        if version is None:
            print(f"Warning: Could not parse filename {filepath}, skipping")
            continue

        try:
            df = pd.read_csv(filepath)
            df['version'] = version
            df['num_gpus'] = num_gpus
            df['source_file'] = Path(filepath).name
            all_data.append(df)
            print(f"  Loaded {filepath}: version={version}, gpus={num_gpus}")
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")

    if not all_data:
        raise ValueError(f"No valid data files found in {base_dir}")

    combined = pd.concat(all_data, ignore_index=True)
    return combined


def remove_outliers(group, n_outliers=N_OUTLIERS_DISCARD):
    """Remove top n outliers from a group based on duration."""
    if len(group) <= n_outliers:
        return group
    # Sort by duration and remove top n
    sorted_group = group.sort_values('duration_seconds', ascending=False)
    return sorted_group.iloc[n_outliers:]


def compute_statistics(df):
    """Compute mean, std for total time at each GPU count.

    Total time is computed as the sum of all NUFFT sub-component timers,
    rather than using mainTimer.
    """
    # Get all NUFFT sub-timers
    gather_timers = list(TIMER_CATEGORIES_GATHER.keys())
    scatter_timers = list(TIMER_CATEGORIES_SCATTER.keys())
    all_nufft_timers = gather_timers + scatter_timers

    component_df = df[df['timer_name'].isin(all_nufft_timers)].copy()

    # Remove outliers per group
    component_df = component_df.groupby(
        ['version', 'num_gpus', 'source_file', 'rank', 'timer_name'], group_keys=False
    ).apply(lambda g: remove_outliers(g, N_OUTLIERS_DISCARD)).reset_index(drop=True)

    # Sum across iterations for each (version, num_gpus, source_file, rank, timer_name)
    summed_per_timer = component_df.groupby(
        ['version', 'num_gpus', 'source_file', 'rank', 'timer_name']
    )['duration_seconds'].sum().reset_index()

    # Sum all timers to get total per (version, num_gpus, source_file, rank)
    total_per_rank = summed_per_timer.groupby(
        ['version', 'num_gpus', 'source_file', 'rank']
    )['duration_seconds'].sum().reset_index()

    # Average across ranks to get per-run total
    per_run = total_per_rank.groupby(
        ['version', 'num_gpus', 'source_file']
    )['duration_seconds'].mean().reset_index()

    # Aggregate across runs
    grouped = per_run.groupby(['version', 'num_gpus'])

    stats = grouped['duration_seconds'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    stats.columns = ['version', 'num_gpus', 'mean', 'std', 'min', 'max', 'count']

    stats['mean_ms'] = stats['mean'] * 1000
    stats['std_ms'] = stats['std'] * 1000 if 'std' in stats.columns else 0

    return stats


def compute_component_statistics(df, components):
    """Compute total time for each component at each GPU count."""
    df = df.copy()

    component_df = df[df['timer_name'].isin(components)].copy()

    # Remove outliers per group
    component_df = component_df.groupby(
        ['version', 'num_gpus', 'source_file', 'rank', 'timer_name'], group_keys=False
    ).apply(lambda g: remove_outliers(g, N_OUTLIERS_DISCARD)).reset_index(drop=True)

    # Sum across iterations for each rank
    summed = component_df.groupby(
        ['version', 'num_gpus', 'source_file', 'rank', 'timer_name']
    )['duration_seconds'].sum().reset_index()

    # Average across ranks
    grouped = summed.groupby(['version', 'num_gpus', 'timer_name'])
    stats = grouped['duration_seconds'].agg(['mean', 'std']).reset_index()
    stats.columns = ['version', 'num_gpus', 'component', 'mean', 'std']

    stats['mean_ms'] = stats['mean'] * 1000
    stats['std_ms'] = stats['std'] * 1000

    return stats


def compute_nufft_subcomponent_statistics(df):
    """Compute statistics for NUFFT sub-components (Gather and Scatter breakdown)."""
    df = df.copy()

    # All NUFFT sub-timers
    gather_timers = list(TIMER_CATEGORIES_GATHER.keys())
    scatter_timers = list(TIMER_CATEGORIES_SCATTER.keys())
    all_timers = gather_timers + scatter_timers

    component_df = df[df['timer_name'].isin(all_timers)].copy()

    # Remove outliers
    component_df = component_df.groupby(
        ['version', 'num_gpus', 'source_file', 'rank', 'timer_name'], group_keys=False
    ).apply(lambda g: remove_outliers(g, N_OUTLIERS_DISCARD)).reset_index(drop=True)

    # Sum across iterations
    summed = component_df.groupby(
        ['version', 'num_gpus', 'source_file', 'rank', 'timer_name']
    )['duration_seconds'].sum().reset_index()

    # Average across ranks
    grouped = summed.groupby(['version', 'num_gpus', 'timer_name'])
    stats = grouped['duration_seconds'].agg(['mean', 'std']).reset_index()
    stats.columns = ['version', 'num_gpus', 'timer_name', 'mean', 'std']

    # Map to readable names and operation type
    def map_timer(timer_name):
        if timer_name in TIMER_CATEGORIES_GATHER:
            return TIMER_CATEGORIES_GATHER[timer_name], 'Gather'
        elif timer_name in TIMER_CATEGORIES_SCATTER:
            return TIMER_CATEGORIES_SCATTER[timer_name], 'Scatter'
        return timer_name, 'Unknown'

    stats['component'], stats['operation'] = zip(*stats['timer_name'].map(map_timer))

    stats['mean_ms'] = stats['mean'] * 1000
    stats['std_ms'] = stats['std'] * 1000

    return stats


def compute_efficiency(stats):
    """Compute parallel efficiency relative to smallest GPU count per version."""
    stats = stats.copy()
    stats['efficiency'] = np.nan

    for version in stats['version'].unique():
        mask = stats['version'] == version
        subset = stats[mask].copy()

        if subset.empty:
            continue

        min_gpus = subset['num_gpus'].min()
        baseline = subset[subset['num_gpus'] == min_gpus]['mean'].values[0]
        baseline_gpus = min_gpus

        for idx in subset.index:
            n_gpus = stats.loc[idx, 'num_gpus']
            time_n = stats.loc[idx, 'mean']
            ideal_time = baseline * baseline_gpus / n_gpus
            stats.loc[idx, 'efficiency'] = ideal_time / time_n * 100

    return stats


def plot_nufft_breakdown(nufft_stats, main_stats, output_prefix, version_filter=None):
    """
    Create publication-quality stacked bar chart showing NUFFT component breakdown
    for both Gather and Scatter operations side by side.

    Total time is computed as the sum of sub-components.
    """
    setup_style()

    if version_filter:
        nufft_stats = nufft_stats[nufft_stats['version'] == version_filter].copy()
        version_label = VERSION_LABELS.get(version_filter, version_filter)
    else:
        version_filter = nufft_stats['version'].iloc[0]
        nufft_stats = nufft_stats[nufft_stats['version'] == version_filter].copy()
        version_label = VERSION_LABELS.get(version_filter, version_filter)

    gpu_counts = sorted(nufft_stats['num_gpus'].unique())
    n_gpus = len(gpu_counts)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=False)

    bar_width = 0.7
    x = np.arange(n_gpus)

    # --- Left panel: Scatter (Type-1 NUFFT) ---
    ax = axes[0]
    scatter_data = nufft_stats[nufft_stats['operation'] == 'Scatter']

    bottoms = np.zeros(n_gpus)
    for component in SCATTER_STACK_ORDER:
        heights = []
        for gpu_count in gpu_counts:
            comp_data = scatter_data[
                (scatter_data['num_gpus'] == gpu_count) &
                (scatter_data['component'] == component)
                ]
            if not comp_data.empty:
                heights.append(comp_data['mean_ms'].values[0])
            else:
                heights.append(0)

        heights = np.array(heights)
        color = COMPONENT_COLORS.get(component, '#999999')

        ax.bar(x, heights, bar_width, bottom=bottoms, label=component,
               color=color, edgecolor='white', linewidth=0.5)
        bottoms += heights

    # Add total time annotations
    for i, gpu_count in enumerate(gpu_counts):
        ax.annotate(f'{bottoms[i]:.0f}',
                    xy=(x[i], bottoms[i]), xytext=(0, 2),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=7, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in gpu_counts])
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Scatter (Type-1 NUFFT)', fontweight='medium')
    ax.set_ylim(bottom=0)

    # --- Right panel: Gather (Type-2 NUFFT) ---
    ax = axes[1]
    gather_data = nufft_stats[nufft_stats['operation'] == 'Gather']

    bottoms = np.zeros(n_gpus)
    for component in GATHER_STACK_ORDER:
        heights = []
        for gpu_count in gpu_counts:
            comp_data = gather_data[
                (gather_data['num_gpus'] == gpu_count) &
                (gather_data['component'] == component)
                ]
            if not comp_data.empty:
                heights.append(comp_data['mean_ms'].values[0])
            else:
                heights.append(0)

        heights = np.array(heights)
        color = COMPONENT_COLORS.get(component, '#999999')

        ax.bar(x, heights, bar_width, bottom=bottoms, label=component,
               color=color, edgecolor='white', linewidth=0.5)
        bottoms += heights

    # Add total time annotations
    for i, gpu_count in enumerate(gpu_counts):
        ax.annotate(f'{bottoms[i]:.0f}',
                    xy=(x[i], bottoms[i]), xytext=(0, 2),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=7, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in gpu_counts])
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Gather (Type-2 NUFFT)', fontweight='medium')
    ax.set_ylim(bottom=0)

    # Create unified legend below the plots
    # Collect all unique components
    all_components = list(dict.fromkeys(SCATTER_STACK_ORDER + GATHER_STACK_ORDER))
    handles = [mpatches.Patch(facecolor=COMPONENT_COLORS.get(c, '#999999'),
                              edgecolor='white', linewidth=0.5, label=c)
               for c in all_components]

    fig.legend(handles=handles, loc='lower center', ncol=len(all_components),
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

    suffix = f'_{version_filter}' if version_filter else ''
    for fmt in ['pdf', 'png']:
        outpath = f'{output_prefix}_nufft_breakdown{suffix}.{fmt}'
        plt.savefig(outpath, facecolor='white', dpi=300 if fmt == 'png' else None)
    print(f"Saved: {output_prefix}_nufft_breakdown{suffix}.pdf")

    plt.close()


def plot_nufft_breakdown_percentage(nufft_stats, output_prefix, version_filter=None):
    """
    Create publication-quality percentage breakdown chart for NUFFT components.
    """
    setup_style()

    if version_filter:
        nufft_stats = nufft_stats[nufft_stats['version'] == version_filter].copy()
        version_label = VERSION_LABELS.get(version_filter, version_filter)
    else:
        version_filter = nufft_stats['version'].iloc[0]
        nufft_stats = nufft_stats[nufft_stats['version'] == version_filter].copy()
        version_label = VERSION_LABELS.get(version_filter, version_filter)

    gpu_counts = sorted(nufft_stats['num_gpus'].unique())
    n_gpus = len(gpu_counts)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=True)

    bar_width = 0.7
    x = np.arange(n_gpus)

    # --- Left panel: Scatter (Type-1 NUFFT) ---
    ax = axes[0]
    scatter_data = nufft_stats[nufft_stats['operation'] == 'Scatter']

    # Compute totals for each GPU count
    totals = []
    for gpu_count in gpu_counts:
        gpu_data = scatter_data[scatter_data['num_gpus'] == gpu_count]
        totals.append(gpu_data['mean_ms'].sum())
    totals = np.array(totals)

    bottoms = np.zeros(n_gpus)
    for component in SCATTER_STACK_ORDER:
        heights = []
        for i, gpu_count in enumerate(gpu_counts):
            comp_data = scatter_data[
                (scatter_data['num_gpus'] == gpu_count) &
                (scatter_data['component'] == component)
                ]
            if not comp_data.empty and totals[i] > 0:
                heights.append(comp_data['mean_ms'].values[0] / totals[i] * 100)
            else:
                heights.append(0)

        heights = np.array(heights)
        color = COMPONENT_COLORS.get(component, '#999999')

        bars = ax.bar(x, heights, bar_width, bottom=bottoms, label=component,
                      color=color, edgecolor='white', linewidth=0.5)

        # Add percentage labels for segments > 12%
        for i, (h, b) in enumerate(zip(heights, bottoms)):
            if h > 12:
                ax.text(x[i], b + h/2, f'{h:.0f}%',
                        ha='center', va='center', fontsize=6,
                        color='white', fontweight='bold')

        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in gpu_counts])
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Fraction (%)')
    ax.set_title('Scatter (Type-1 NUFFT)', fontweight='medium')
    ax.set_ylim(0, 100)

    # --- Right panel: Gather (Type-2 NUFFT) ---
    ax = axes[1]
    gather_data = nufft_stats[nufft_stats['operation'] == 'Gather']

    totals = []
    for gpu_count in gpu_counts:
        gpu_data = gather_data[gather_data['num_gpus'] == gpu_count]
        totals.append(gpu_data['mean_ms'].sum())
    totals = np.array(totals)

    bottoms = np.zeros(n_gpus)
    for component in GATHER_STACK_ORDER:
        heights = []
        for i, gpu_count in enumerate(gpu_counts):
            comp_data = gather_data[
                (gather_data['num_gpus'] == gpu_count) &
                (gather_data['component'] == component)
                ]
            if not comp_data.empty and totals[i] > 0:
                heights.append(comp_data['mean_ms'].values[0] / totals[i] * 100)
            else:
                heights.append(0)

        heights = np.array(heights)
        color = COMPONENT_COLORS.get(component, '#999999')

        bars = ax.bar(x, heights, bar_width, bottom=bottoms, label=component,
                      color=color, edgecolor='white', linewidth=0.5)

        for i, (h, b) in enumerate(zip(heights, bottoms)):
            if h > 12:
                ax.text(x[i], b + h/2, f'{h:.0f}%',
                        ha='center', va='center', fontsize=6,
                        color='white', fontweight='bold')

        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in gpu_counts])
    ax.set_xlabel('Number of GPUs')
    ax.set_title('Gather (Type-2 NUFFT)', fontweight='medium')
    ax.set_ylim(0, 100)

    # Unified legend
    all_components = list(dict.fromkeys(SCATTER_STACK_ORDER + GATHER_STACK_ORDER))
    handles = [mpatches.Patch(facecolor=COMPONENT_COLORS.get(c, '#999999'),
                              edgecolor='white', linewidth=0.5, label=c)
               for c in all_components]

    fig.legend(handles=handles, loc='lower center', ncol=len(all_components),
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

    suffix = f'_{version_filter}' if version_filter else ''
    for fmt in ['pdf', 'png']:
        outpath = f'{output_prefix}_nufft_breakdown_pct{suffix}.{fmt}'
        plt.savefig(outpath, facecolor='white', dpi=300 if fmt == 'png' else None)
    print(f"Saved: {output_prefix}_nufft_breakdown_pct{suffix}.pdf")

    plt.close()


def plot_scaling_time(stats, output_prefix, log_scale=True):
    """Plot execution time vs GPU count."""
    setup_style()

    fig, ax = plt.subplots(figsize=(4, 3))

    for version in sorted(stats['version'].unique()):
        version_data = stats[stats['version'] == version].sort_values('num_gpus')

        color = VERSION_COLORS.get(version, '#999999')
        marker = VERSION_MARKERS.get(version, 'o')
        label = VERSION_LABELS.get(version, version)

        ax.errorbar(
            version_data['num_gpus'],
            version_data['mean_ms'],
            yerr=version_data['std_ms'],
            marker=marker,
            color=color,
            markersize=5,
            linewidth=1.2,
            capsize=2,
            capthick=0.5,
            label=label
        )

    if not stats.empty:
        gpu_counts = sorted(stats['num_gpus'].unique())
        min_gpus = min(gpu_counts)
        ref_time = stats[stats['num_gpus'] == min_gpus]['mean_ms'].mean()
        ideal_times = [ref_time * min_gpus / g for g in gpu_counts]
        ax.plot(gpu_counts, ideal_times, 'k--', linewidth=1.0, alpha=0.6, label='Ideal')

    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Strong Scaling')

    if log_scale:
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')

    ax.legend(loc='best')

    if not stats.empty:
        gpu_counts = sorted(stats['num_gpus'].unique())
        ax.set_xticks(gpu_counts)
        ax.set_xticklabels([str(g) for g in gpu_counts])

    plt.tight_layout()

    plt.savefig(f'{output_prefix}_time.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_time.png', dpi=300, facecolor='white')
    print(f"Saved: {output_prefix}_time.pdf, {output_prefix}_time.png")
    plt.close()


def plot_scaling_efficiency(stats, output_prefix):
    """Plot parallel efficiency vs GPU count."""
    setup_style()

    stats = compute_efficiency(stats)

    fig, ax = plt.subplots(figsize=(4, 3))

    for version in sorted(stats['version'].unique()):
        version_data = stats[stats['version'] == version].sort_values('num_gpus')

        color = VERSION_COLORS.get(version, '#999999')
        marker = VERSION_MARKERS.get(version, 'o')
        label = VERSION_LABELS.get(version, version)

        ax.plot(
            version_data['num_gpus'],
            version_data['efficiency'],
            marker=marker,
            color=color,
            markersize=5,
            linewidth=1.2,
            label=label
        )

    ax.axhline(y=100, color='k', linestyle='--', linewidth=1.0, alpha=0.6)

    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Parallel Efficiency (%)')
    ax.set_title('Strong Scaling Efficiency')
    ax.set_xscale('log', base=2)
    ax.set_ylim(0, 110)

    ax.legend(loc='best')

    if not stats.empty:
        gpu_counts = sorted(stats['num_gpus'].unique())
        ax.set_xticks(gpu_counts)
        ax.set_xticklabels([str(g) for g in gpu_counts])

    plt.tight_layout()

    plt.savefig(f'{output_prefix}_efficiency.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_efficiency.png', dpi=300, facecolor='white')
    print(f"Saved: {output_prefix}_efficiency.pdf, {output_prefix}_efficiency.png")
    plt.close()


def plot_combined(stats, output_prefix):
    """Plot time and efficiency side by side."""
    setup_style()

    stats = compute_efficiency(stats)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # Left: execution time
    ax = axes[0]
    for version in sorted(stats['version'].unique()):
        version_data = stats[stats['version'] == version].sort_values('num_gpus')

        color = VERSION_COLORS.get(version, '#999999')
        marker = VERSION_MARKERS.get(version, 'o')
        label = VERSION_LABELS.get(version, version)

        ax.errorbar(
            version_data['num_gpus'],
            version_data['mean_ms'],
            yerr=version_data['std_ms'],
            marker=marker,
            color=color,
            markersize=5,
            linewidth=1.2,
            capsize=2,
            capthick=0.5,
            label=label
        )

    if not stats.empty:
        gpu_counts = sorted(stats['num_gpus'].unique())
        min_gpus = min(gpu_counts)
        ref_time = stats[stats['num_gpus'] == min_gpus]['mean_ms'].mean()
        ideal_times = [ref_time * min_gpus / g for g in gpu_counts]
        ax.plot(gpu_counts, ideal_times, 'k--', linewidth=1.0, alpha=0.6, label='Ideal')

    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Time (ms)')
    ax.set_title('(a) Execution Time', loc='left', fontweight='medium')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=7)

    if not stats.empty:
        gpu_counts = sorted(stats['num_gpus'].unique())
        ax.set_xticks(gpu_counts)
        ax.set_xticklabels([str(g) for g in gpu_counts])

    # Right: efficiency
    ax = axes[1]
    for version in sorted(stats['version'].unique()):
        version_data = stats[stats['version'] == version].sort_values('num_gpus')

        color = VERSION_COLORS.get(version, '#999999')
        marker = VERSION_MARKERS.get(version, 'o')
        label = VERSION_LABELS.get(version, version)

        ax.plot(
            version_data['num_gpus'],
            version_data['efficiency'],
            marker=marker,
            color=color,
            markersize=5,
            linewidth=1.2,
            label=label
        )

    ax.axhline(y=100, color='k', linestyle='--', linewidth=1.0, alpha=0.6)

    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title('(b) Parallel Efficiency', loc='left', fontweight='medium')
    ax.set_xscale('log', base=2)
    ax.set_ylim(0, 110)

    if not stats.empty:
        gpu_counts = sorted(stats['num_gpus'].unique())
        ax.set_xticks(gpu_counts)
        ax.set_xticklabels([str(g) for g in gpu_counts])

    plt.tight_layout()

    plt.savefig(f'{output_prefix}_combined.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_combined.png', dpi=300, facecolor='white')
    print(f"Saved: {output_prefix}_combined.pdf, {output_prefix}_combined.png")
    plt.close()


def plot_high_level_breakdown(component_stats, main_stats, output_prefix, version_filter=None):
    """Plot high-level component breakdown (Particle Push, Gather, Scatter).

    Note: Since we compute totals from sub-components, there is no "Other" category.
    """
    setup_style()

    if version_filter:
        component_stats = component_stats[component_stats['version'] == version_filter].copy()
        version_label = VERSION_LABELS.get(version_filter, version_filter)
    else:
        version_filter = component_stats['version'].iloc[0]
        component_stats = component_stats[component_stats['version'] == version_filter].copy()
        version_label = VERSION_LABELS.get(version_filter, version_filter)

    gpu_counts = sorted(component_stats['num_gpus'].unique())
    n_gpus = len(gpu_counts)

    fig, ax = plt.subplots(figsize=(5, 3))

    bar_width = 0.7
    x = np.arange(n_gpus)

    # Stack order without "Other" since totals come from sub-components
    stack_order = ['updateParticle', 'ScatterPIFNUFFT', 'GatherPIFNUFFT']

    bottoms = np.zeros(n_gpus)
    for component in stack_order:
        heights = []
        for gpu_count in gpu_counts:
            comp_data = component_stats[
                (component_stats['num_gpus'] == gpu_count) &
                (component_stats['component'] == component)
                ]
            if not comp_data.empty:
                heights.append(comp_data['mean_ms'].values[0])
            else:
                heights.append(0)
        heights = np.array(heights)

        color = HIGH_LEVEL_COLORS.get(component, '#999999')
        label = HIGH_LEVEL_LABELS.get(component, component)

        ax.bar(x, heights, bar_width, bottom=bottoms, label=label,
               color=color, edgecolor='white', linewidth=0.5)
        bottoms += heights

    # Total annotations
    for i, gpu_count in enumerate(gpu_counts):
        ax.annotate(f'{bottoms[i]:.0f}',
                    xy=(x[i], bottoms[i]), xytext=(0, 2),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=7, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in gpu_counts])
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Component Breakdown ({version_label})', fontweight='medium')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=7)

    plt.tight_layout()

    suffix = f'_{version_filter}' if version_filter else ''
    plt.savefig(f'{output_prefix}_components{suffix}.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_components{suffix}.png', dpi=300, facecolor='white')
    print(f"Saved: {output_prefix}_components{suffix}.pdf")
    plt.close()


def print_summary(stats):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("SCALING SUMMARY")
    print("=" * 70)

    stats = compute_efficiency(stats)

    for version in sorted(stats['version'].unique()):
        print(f"\n{VERSION_LABELS.get(version, version)}:")
        print("-" * 50)

        version_data = stats[stats['version'] == version]

        print(f"  {'GPUs':>6} {'Time (ms)':>12} {'Std (ms)':>10} {'Efficiency':>10}")
        print(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*10}")

        for _, row in version_data.sort_values('num_gpus').iterrows():
            std_str = f"{row['std_ms']:.1f}" if pd.notna(row['std_ms']) else "N/A"
            print(f"  {row['num_gpus']:>6} {row['mean_ms']:>12.1f} {std_str:>10} {row['efficiency']:>9.1f}%")


def print_nufft_summary(nufft_stats):
    """Print NUFFT sub-component breakdown summary."""
    print("\n" + "=" * 70)
    print("NUFFT COMPONENT BREAKDOWN")
    print("=" * 70)

    for version in sorted(nufft_stats['version'].unique()):
        print(f"\n{VERSION_LABELS.get(version, version)}:")

        ver_data = nufft_stats[nufft_stats['version'] == version]

        for operation in ['Scatter', 'Gather']:
            print(f"\n  {operation} (Type-{'1' if operation == 'Scatter' else '2'} NUFFT):")
            print("-" * 60)

            op_data = ver_data[ver_data['operation'] == operation]
            gpu_counts = sorted(op_data['num_gpus'].unique())

            stack_order = SCATTER_STACK_ORDER if operation == 'Scatter' else GATHER_STACK_ORDER

            header = f"    {'Component':<16}"
            for g in gpu_counts:
                header += f" {g:>7} GPUs"
            print(header)
            print("    " + "-" * (16 + 12 * len(gpu_counts)))

            for component in stack_order:
                row = f"    {component:<16}"
                for g in gpu_counts:
                    comp_data = op_data[
                        (op_data['num_gpus'] == g) &
                        (op_data['component'] == component)
                        ]
                    if not comp_data.empty:
                        row += f" {comp_data['mean_ms'].values[0]:>7.1f} ms"
                    else:
                        row += f" {'---':>10}"
                print(row)

            # Total row
            row = f"    {'Total':<16}"
            for g in gpu_counts:
                gpu_data = op_data[op_data['num_gpus'] == g]
                total = gpu_data['mean_ms'].sum()
                row += f" {total:>7.1f} ms"
            print(row)


def main():
    parser = argparse.ArgumentParser(description='Plot LandauDampingPIF scaling results')
    parser.add_argument('--dirs', '-d', nargs='+', required=True,
                        help='Directories containing benchmark CSV files')
    parser.add_argument('--output', '-o', default='landau_scaling',
                        help='Output file prefix')
    parser.add_argument('--efficiency', action='store_true',
                        help='Generate efficiency plot')
    parser.add_argument('--combined', action='store_true',
                        help='Generate combined time+efficiency plot')
    parser.add_argument('--components', action='store_true',
                        help='Generate high-level component breakdown plot')
    parser.add_argument('--nufft-breakdown', action='store_true',
                        help='Generate NUFFT sub-component breakdown plot')
    parser.add_argument('--nufft-breakdown-pct', action='store_true',
                        help='Generate NUFFT sub-component percentage plot')
    parser.add_argument('--all', action='store_true',
                        help='Generate all plot types')
    args = parser.parse_args()

    # Load data
    all_data = []

    for d in args.dirs:
        print(f"Loading data from {d}...")
        try:
            data = load_timing_data(d)
            all_data.append(data)
        except ValueError as e:
            print(f"  Warning: {e}")

    if not all_data:
        print("Error: No data loaded!")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal records loaded: {len(combined_df)}")
    print(f"Discarding top {N_OUTLIERS_DISCARD} outliers per group")

    # Compute statistics
    stats = compute_statistics(combined_df)
    component_stats = compute_component_statistics(combined_df, HIGH_LEVEL_COMPONENTS)
    nufft_stats = compute_nufft_subcomponent_statistics(combined_df)

    # Print summaries
    print_summary(stats)
    if not nufft_stats.empty:
        print_nufft_summary(nufft_stats)

    # Generate plots
    if args.all:
        args.efficiency = True
        args.combined = True
        args.components = True
        args.nufft_breakdown = True
        args.nufft_breakdown_pct = True

    # Always generate time plot
    plot_scaling_time(stats, args.output)

    if args.efficiency:
        plot_scaling_efficiency(stats, args.output)

    if args.combined:
        plot_combined(stats, args.output)

    if args.components and not component_stats.empty:
        for version in component_stats['version'].unique():
            plot_high_level_breakdown(component_stats, stats, args.output, version_filter=version)

    if args.nufft_breakdown and not nufft_stats.empty:
        for version in nufft_stats['version'].unique():
            plot_nufft_breakdown(nufft_stats, stats, args.output, version_filter=version)

    if args.nufft_breakdown_pct and not nufft_stats.empty:
        for version in nufft_stats['version'].unique():
            plot_nufft_breakdown_percentage(nufft_stats, args.output, version_filter=version)


if __name__ == '__main__':
    main()