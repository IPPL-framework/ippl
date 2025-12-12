#!/usr/bin/env python3
"""
Plot NUFFT strong scaling results across multiple clusters.

Reads timing data from files like:
  benchmark_results_alps/timing_gpus8.csv
  benchmark_results_juwels/timing_gpus16.csv

Input CSV format:
  num_ranks,grid_size,num_particles,transform_type,run_index,time_ms

Usage:
    python plot_nufft_scaling.py -d benchmark_results_alps benchmark_results_juwels -o fig_scaling
    python plot_nufft_scaling.py -d benchmark_results_alps -o fig_scaling --efficiency
"""

import pandas as pd
import matplotlib.pyplot as plt
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
        'font.size': 8,
        'mathtext.fontset': 'stix',
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'axes.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'legend.fontsize': 7,
        'legend.frameon': False,
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })


# Colorblind-friendly palette (Okabe-Ito)
CLUSTER_COLORS = {
    'alps': '#0072B2',      # Blue
    'juwels': '#E69F00',    # Orange
    'lumi': '#009E73',      # Bluish green
}

CLUSTER_MARKERS = {
    'alps': 'o',
    'juwels': 's',
    'lumi': '^',
}

CLUSTER_LABELS = {
    'alps': 'Alps (GH200)',
    'juwels': 'JUWELS (A100)',
    'lumi': 'LUMI (MI250X)',
}

TYPE_LINESTYLES = {
    'type1': '-',
    'type2': '--',
}

TYPE_LABELS = {
    'type1': 'Type-1',
    'type2': 'Type-2',
}

# Component colors (consistent with breakdown plot)
COMPONENT_COLORS = {
    'Scatter/Gather': '#0072B2',   # Blue
    'FFT': '#CC79A7',              # Sky blue
    'Halo': '#56B4E9',             # Pink
    'Deconv/Precorr': '#E69F00',   # Orange
}

# Timer name mappings
TIMER_CATEGORIES_T1 = {
    'scatterTimerNUFFT1': 'Scatter/Gather',
    'accumulateHaloNUFFT1': 'Halo',
    'FFTNUFFT1': 'FFT',
    'deconvolutionNUFFT1': 'Deconv/Precorr',
}

TIMER_CATEGORIES_T2 = {
    'PrecorrectionNUFFT2': 'Deconv/Precorr',
    'FFTNUFFT2': 'FFT',
    'FillHaloNUFFT2': 'Halo',
    'GatherNUFFT2': 'Scatter/Gather',
}

# Stack order (bottom to top)
STACK_ORDER = ['Scatter/Gather', 'FFT', 'Deconv/Precorr', 'Halo']

# Labels per type
COMPONENT_LABELS_T1 = {
    'Scatter/Gather': 'Spreading',
    'FFT': 'FFT',
    'Halo': 'Halo',
    'Deconv/Precorr': 'Deconvolution',
}

COMPONENT_LABELS_T2 = {
    'Scatter/Gather': 'Interpolation',
    'FFT': 'FFT',
    'Halo': 'Halo',
    'Deconv/Precorr': 'Precorrection',
}


def load_cluster_data(base_dir):
    """Load all timing CSVs from a cluster directory."""
    base_path = Path(base_dir)

    # Extract cluster name from directory
    cluster_name = base_path.name.replace('benchmark_results_', '').lower()

    all_data = []

    # Find all timing files
    pattern = str(base_path / 'timings_gpus*.csv')
    files = glob.glob(pattern)

    if not files:
        # Try alternative pattern
        pattern = str(base_path / 'timing_*.csv')
        files = glob.glob(pattern)

    for filepath in files:
        try:
            df = pd.read_csv(filepath)
            df['cluster'] = cluster_name
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")

    if not all_data:
        raise ValueError(f"No data files found in {base_dir}")

    combined = pd.concat(all_data, ignore_index=True)
    return combined


def load_component_data(base_dir):
    """Load all component timing CSVs from a cluster directory."""
    base_path = Path(base_dir)

    # Extract cluster name from directory
    cluster_name = base_path.name.replace('benchmark_results_', '').lower()

    all_data = []

    # Find all component files
    pattern = str(base_path / 'nufft_components_*.csv')
    files = glob.glob(pattern)

    if not files:
        # Try alternative patterns
        for alt_pattern in ['components_*.csv', 'nufft_components*.csv']:
            pattern = str(base_path / alt_pattern)
            files = glob.glob(pattern)
            if files:
                break

    for filepath in files:
        try:
            df = pd.read_csv(filepath)
            df['cluster'] = cluster_name
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")

    if not all_data:
        return None

    combined = pd.concat(all_data, ignore_index=True)
    return combined


def compute_statistics(df):
    """Compute mean, std, min, max for each configuration."""
    grouped = df.groupby(['cluster', 'num_ranks', 'grid_size', 'transform_type'])

    stats = grouped['time_ms'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()

    # Also get num_particles (should be constant per config)
    particles = grouped['num_particles'].first().reset_index()
    stats = stats.merge(particles, on=['cluster', 'num_ranks', 'grid_size', 'transform_type'])

    # Compute throughput in Mpts/s
    stats['throughput_mpts'] = stats['num_particles'] / stats['mean'] * 1000 / 1e6
    stats['throughput_std'] = stats['throughput_mpts'] * (stats['std'] / stats['mean'])

    return stats


def compute_efficiency(stats):
    """Compute parallel efficiency relative to smallest GPU count per cluster."""
    stats = stats.copy()
    stats['efficiency'] = np.nan

    for cluster in stats['cluster'].unique():
        for ttype in stats['transform_type'].unique():
            mask = (stats['cluster'] == cluster) & (stats['transform_type'] == ttype)
            subset = stats[mask].copy()

            if subset.empty:
                continue

            # Find baseline (smallest GPU count)
            min_ranks = subset['num_ranks'].min()
            baseline = subset[subset['num_ranks'] == min_ranks]['mean'].values[0]
            baseline_ranks = min_ranks

            # Compute efficiency: (T_base * N_base) / (T_n * N_n)
            for idx in subset.index:
                n_ranks = stats.loc[idx, 'num_ranks']
                time_n = stats.loc[idx, 'mean']
                ideal_time = baseline * baseline_ranks / n_ranks
                stats.loc[idx, 'efficiency'] = ideal_time / time_n * 100

    return stats


def compute_component_statistics(df):
    """Compute mean time for each component at each GPU count."""
    # Map timer names to categories
    df = df.copy()

    # Determine if Type-1 or Type-2 timer
    def categorize_timer(timer_name):
        if timer_name in TIMER_CATEGORIES_T1:
            return TIMER_CATEGORIES_T1[timer_name], 'type1'
        elif timer_name in TIMER_CATEGORIES_T2:
            return TIMER_CATEGORIES_T2[timer_name], 'type2'
        else:
            return None, None

    df['category'], df['transform_type'] = zip(*df['timer'].map(categorize_timer))
    df = df.dropna(subset=['category'])

    # Skip first run (warmup artifact in some data)
    df = df[df['run'] > 0]

    # Convert to milliseconds
    df['time_ms'] = df['time_s'] * 1000

    # Group and compute mean
    grouped = df.groupby(['cluster', 'num_ranks', 'transform_type', 'category'])
    stats = grouped['time_ms'].agg(['mean', 'std']).reset_index()

    return stats


def plot_component_scaling(component_stats, output_prefix, cluster_filter=None):
    """Plot stacked bar chart of component times vs GPU count."""
    setup_style()

    if cluster_filter:
        component_stats = component_stats[component_stats['cluster'] == cluster_filter]
        cluster_label = CLUSTER_LABELS.get(cluster_filter, cluster_filter)
    else:
        # Use first cluster if multiple
        cluster_filter = component_stats['cluster'].iloc[0]
        cluster_label = CLUSTER_LABELS.get(cluster_filter, cluster_filter)
        component_stats = component_stats[component_stats['cluster'] == cluster_filter]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    gpu_counts = sorted(component_stats['num_ranks'].unique())
    n_gpus = len(gpu_counts)
    x = np.arange(n_gpus)
    bar_width = 0.65

    for ax, ttype in zip(axes, ['type1', 'type2']):
        subset = component_stats[component_stats['transform_type'] == ttype]

        if ttype == 'type1':
            labels = COMPONENT_LABELS_T1
        else:
            labels = COMPONENT_LABELS_T2

        for gpu_idx, gpu_count in enumerate(gpu_counts):
            gpu_data = subset[subset['num_ranks'] == gpu_count]
            bottom = 0

            for category in STACK_ORDER:
                cat_data = gpu_data[gpu_data['category'] == category]
                if not cat_data.empty:
                    height = cat_data['mean'].values[0]
                    color = COMPONENT_COLORS.get(category, '#999999')
                    label = labels.get(category) if gpu_idx == 0 else None

                    ax.bar(x[gpu_idx], height, bar_width, bottom=bottom,
                           color=color, edgecolor='black', linewidth=0.3,
                           label=label)
                    bottom += height

            # Total annotation
            ax.annotate(f'{bottom:.0f}',
                        xy=(x[gpu_idx], bottom),
                        xytext=(0, 2), textcoords='offset points',
                        ha='center', va='bottom', fontsize=5)

        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in gpu_counts])
        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Time (ms)')
        ax.set_title(TYPE_LABELS[ttype], fontsize=9)
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper right', fontsize=6)

    # Add cluster name to figure
    fig.suptitle(cluster_label, fontsize=9, y=1.02)

    plt.tight_layout()

    suffix = f'_{cluster_filter}' if cluster_filter else ''
    plt.savefig(f'{output_prefix}_components{suffix}.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_components{suffix}.png', dpi=300, facecolor='white')
    print(f"Saved: {output_prefix}_components{suffix}.pdf, {output_prefix}_components{suffix}.png")


def plot_component_scaling_area(component_stats, output_prefix, cluster_filter=None):
    """Plot stacked area chart of component times vs GPU count."""
    setup_style()

    if cluster_filter:
        component_stats = component_stats[component_stats['cluster'] == cluster_filter]
        cluster_label = CLUSTER_LABELS.get(cluster_filter, cluster_filter)
    else:
        cluster_filter = component_stats['cluster'].iloc[0]
        cluster_label = CLUSTER_LABELS.get(cluster_filter, cluster_filter)
        component_stats = component_stats[component_stats['cluster'] == cluster_filter]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    gpu_counts = sorted(component_stats['num_ranks'].unique())

    for ax, ttype in zip(axes, ['type1', 'type2']):
        subset = component_stats[component_stats['transform_type'] == ttype]

        if ttype == 'type1':
            labels = COMPONENT_LABELS_T1
        else:
            labels = COMPONENT_LABELS_T2

        # Build arrays for stackplot
        y_data = []
        legend_labels = []
        colors = []

        for category in STACK_ORDER:
            cat_times = []
            for gpu_count in gpu_counts:
                cat_data = subset[(subset['num_ranks'] == gpu_count) &
                                  (subset['category'] == category)]
                if not cat_data.empty:
                    cat_times.append(cat_data['mean'].values[0])
                else:
                    cat_times.append(0)

            if any(t > 0 for t in cat_times):
                y_data.append(cat_times)
                legend_labels.append(labels.get(category, category))
                colors.append(COMPONENT_COLORS.get(category, '#999999'))

        ax.stackplot(gpu_counts, y_data, labels=legend_labels, colors=colors,
                     edgecolor='black', linewidth=0.3)

        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Time (ms)')
        ax.set_title(TYPE_LABELS[ttype], fontsize=9)
        ax.set_xscale('log', base=2)
        ax.set_xticks(gpu_counts)
        ax.set_xticklabels([str(g) for g in gpu_counts])
        ax.legend(loc='upper right', fontsize=6)

    fig.suptitle(cluster_label, fontsize=9, y=1.02)

    plt.tight_layout()

    suffix = f'_{cluster_filter}' if cluster_filter else ''
    plt.savefig(f'{output_prefix}_components_area{suffix}.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_components_area{suffix}.png', dpi=300, facecolor='white')
    print(f"Saved: {output_prefix}_components_area{suffix}.pdf")


def plot_component_percentage_scaling(component_stats, output_prefix, cluster_filter=None):
    """Plot stacked bar chart showing percentage breakdown vs GPU count."""
    setup_style()

    if cluster_filter:
        component_stats = component_stats[component_stats['cluster'] == cluster_filter]
        cluster_label = CLUSTER_LABELS.get(cluster_filter, cluster_filter)
    else:
        cluster_filter = component_stats['cluster'].iloc[0]
        cluster_label = CLUSTER_LABELS.get(cluster_filter, cluster_filter)
        component_stats = component_stats[component_stats['cluster'] == cluster_filter]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    gpu_counts = sorted(component_stats['num_ranks'].unique())
    n_gpus = len(gpu_counts)
    x = np.arange(n_gpus)
    bar_width = 0.65

    for ax, ttype in zip(axes, ['type1', 'type2']):
        subset = component_stats[component_stats['transform_type'] == ttype]

        if ttype == 'type1':
            labels = COMPONENT_LABELS_T1
        else:
            labels = COMPONENT_LABELS_T2

        for gpu_idx, gpu_count in enumerate(gpu_counts):
            gpu_data = subset[subset['num_ranks'] == gpu_count]

            # Compute total for this GPU count
            total = gpu_data['mean'].sum()
            if total == 0:
                continue

            bottom = 0

            for category in STACK_ORDER:
                cat_data = gpu_data[gpu_data['category'] == category]
                if not cat_data.empty:
                    height = cat_data['mean'].values[0] / total * 100
                    color = COMPONENT_COLORS.get(category, '#999999')
                    label = labels.get(category) if gpu_idx == 0 else None

                    ax.bar(x[gpu_idx], height, bar_width, bottom=bottom,
                           color=color, edgecolor='black', linewidth=0.3,
                           label=label)

                    # Add percentage label if > 10%
                    if height > 10:
                        ax.text(x[gpu_idx], bottom + height/2, f'{height:.0f}%',
                                ha='center', va='center', fontsize=5,
                                color='white', fontweight='bold')

                    bottom += height

        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in gpu_counts])
        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Percentage (%)')
        ax.set_title(TYPE_LABELS[ttype], fontsize=9)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right', fontsize=6)

    fig.suptitle(cluster_label, fontsize=9, y=1.02)

    plt.tight_layout()

    suffix = f'_{cluster_filter}' if cluster_filter else ''
    plt.savefig(f'{output_prefix}_components_pct{suffix}.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_components_pct{suffix}.png', dpi=300, facecolor='white')
    print(f"Saved: {output_prefix}_components_pct{suffix}.pdf")


def print_component_summary(component_stats):
    """Print summary of component breakdown."""
    print("\n" + "=" * 80)
    print("COMPONENT BREAKDOWN SUMMARY")
    print("=" * 80)

    for cluster in sorted(component_stats['cluster'].unique()):
        print(f"\n{CLUSTER_LABELS.get(cluster, cluster)}:")
        print("-" * 70)

        cluster_data = component_stats[component_stats['cluster'] == cluster]

        for ttype in ['type1', 'type2']:
            type_data = cluster_data[cluster_data['transform_type'] == ttype]
            if type_data.empty:
                continue

            if ttype == 'type1':
                labels = COMPONENT_LABELS_T1
            else:
                labels = COMPONENT_LABELS_T2

            print(f"\n  {TYPE_LABELS[ttype]}:")

            gpu_counts = sorted(type_data['num_ranks'].unique())

            # Header
            header = f"  {'Component':<15}"
            for g in gpu_counts:
                header += f" {g:>8} GPUs"
            print(header)
            print("  " + "-" * (15 + 13 * len(gpu_counts)))

            for category in STACK_ORDER:
                row = f"  {labels.get(category, category):<15}"
                for g in gpu_counts:
                    cat_data = type_data[(type_data['num_ranks'] == g) &
                                         (type_data['category'] == category)]
                    if not cat_data.empty:
                        row += f" {cat_data['mean'].values[0]:>8.1f} ms"
                    else:
                        row += f" {'---':>11}"
                print(row)

            # Total row
            row = f"  {'Total':<15}"
            for g in gpu_counts:
                gpu_data = type_data[type_data['num_ranks'] == g]
                total = gpu_data['mean'].sum()
                row += f" {total:>8.1f} ms"
            print(row)


def plot_scaling_time(stats, output_prefix, log_scale=True):
    """Plot execution time vs GPU count."""
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    for ax, ttype in zip(axes, ['type1', 'type2']):
        subset = stats[stats['transform_type'] == ttype]

        for cluster in sorted(subset['cluster'].unique()):
            cluster_data = subset[subset['cluster'] == cluster].sort_values('num_ranks')

            color = CLUSTER_COLORS.get(cluster, '#999999')
            marker = CLUSTER_MARKERS.get(cluster, 'o')
            label = CLUSTER_LABELS.get(cluster, cluster)

            ax.errorbar(
                cluster_data['num_ranks'],
                cluster_data['mean'],
                yerr=cluster_data['std'],
                marker=marker,
                color=color,
                markersize=4,
                linewidth=1,
                capsize=2,
                capthick=0.5,
                label=label
            )

        # Add ideal scaling line
        if not subset.empty:
            ranks = sorted(subset['num_ranks'].unique())
            min_rank = min(ranks)
            # Use mean of all clusters at min_rank for reference
            ref_time = subset[subset['num_ranks'] == min_rank]['mean'].mean()
            ideal_times = [ref_time * min_rank / r for r in ranks]
            ax.plot(ranks, ideal_times, 'k:', linewidth=0.8, alpha=0.5, label='Ideal')

        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Time (ms)')
        ax.set_title(TYPE_LABELS[ttype], fontsize=9)

        if log_scale:
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')

        ax.legend(loc='best', fontsize=6)

        # Set x-ticks to actual GPU counts
        if not subset.empty:
            ranks = sorted(subset['num_ranks'].unique())
            ax.set_xticks(ranks)
            ax.set_xticklabels([str(r) for r in ranks])

    plt.tight_layout()

    plt.savefig(f'{output_prefix}_time.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_time.png', dpi=300, facecolor='white')
    print(f"Saved: {output_prefix}_time.pdf, {output_prefix}_time.png")


def plot_scaling_throughput(stats, output_prefix):
    """Plot throughput vs GPU count."""
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    for ax, ttype in zip(axes, ['type1', 'type2']):
        subset = stats[stats['transform_type'] == ttype]

        for cluster in sorted(subset['cluster'].unique()):
            cluster_data = subset[subset['cluster'] == cluster].sort_values('num_ranks')

            color = CLUSTER_COLORS.get(cluster, '#999999')
            marker = CLUSTER_MARKERS.get(cluster, 'o')
            label = CLUSTER_LABELS.get(cluster, cluster)

            ax.errorbar(
                cluster_data['num_ranks'],
                cluster_data['throughput_mpts'],
                yerr=cluster_data['throughput_std'],
                marker=marker,
                color=color,
                markersize=4,
                linewidth=1,
                capsize=2,
                capthick=0.5,
                label=label
            )

        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Throughput (Mpts/s)')
        ax.set_title(TYPE_LABELS[ttype], fontsize=9)
        ax.set_xscale('log', base=2)

        ax.legend(loc='best', fontsize=6)

        # Set x-ticks
        if not subset.empty:
            ranks = sorted(subset['num_ranks'].unique())
            ax.set_xticks(ranks)
            ax.set_xticklabels([str(r) for r in ranks])

    plt.tight_layout()

    plt.savefig(f'{output_prefix}_throughput.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_throughput.png', dpi=300, facecolor='white')
    print(f"Saved: {output_prefix}_throughput.pdf, {output_prefix}_throughput.png")


def plot_scaling_efficiency(stats, output_prefix):
    """Plot parallel efficiency vs GPU count."""
    setup_style()

    stats = compute_efficiency(stats)

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    for ax, ttype in zip(axes, ['type1', 'type2']):
        subset = stats[stats['transform_type'] == ttype]

        for cluster in sorted(subset['cluster'].unique()):
            cluster_data = subset[subset['cluster'] == cluster].sort_values('num_ranks')

            color = CLUSTER_COLORS.get(cluster, '#999999')
            marker = CLUSTER_MARKERS.get(cluster, 'o')
            label = CLUSTER_LABELS.get(cluster, cluster)

            ax.plot(
                cluster_data['num_ranks'],
                cluster_data['efficiency'],
                marker=marker,
                color=color,
                markersize=4,
                linewidth=1,
                label=label
            )

        # Add 100% reference line
        ax.axhline(y=100, color='k', linestyle=':', linewidth=0.8, alpha=0.5)

        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Parallel Efficiency (%)')
        ax.set_title(TYPE_LABELS[ttype], fontsize=9)
        ax.set_xscale('log', base=2)
        ax.set_ylim(0, 110)

        ax.legend(loc='best', fontsize=6)

        # Set x-ticks
        if not subset.empty:
            ranks = sorted(subset['num_ranks'].unique())
            ax.set_xticks(ranks)
            ax.set_xticklabels([str(r) for r in ranks])

    plt.tight_layout()

    plt.savefig(f'{output_prefix}_efficiency.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_efficiency.png', dpi=300, facecolor='white')
    print(f"Saved: {output_prefix}_efficiency.pdf, {output_prefix}_efficiency.png")


def plot_combined(stats, output_prefix):
    """Plot time and efficiency side by side for both types."""
    setup_style()

    stats = compute_efficiency(stats)

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.5))

    # Top row: execution time
    for ax, ttype in zip(axes[0], ['type1', 'type2']):
        subset = stats[stats['transform_type'] == ttype]

        for cluster in sorted(subset['cluster'].unique()):
            cluster_data = subset[subset['cluster'] == cluster].sort_values('num_ranks')

            color = CLUSTER_COLORS.get(cluster, '#999999')
            marker = CLUSTER_MARKERS.get(cluster, 'o')
            label = CLUSTER_LABELS.get(cluster, cluster)

            ax.errorbar(
                cluster_data['num_ranks'],
                cluster_data['mean'],
                yerr=cluster_data['std'],
                marker=marker,
                color=color,
                markersize=4,
                linewidth=1,
                capsize=2,
                capthick=0.5,
                label=label
            )

        # Ideal scaling line
        if not subset.empty:
            ranks = sorted(subset['num_ranks'].unique())
            min_rank = min(ranks)
            ref_time = subset[subset['num_ranks'] == min_rank]['mean'].mean()
            ideal_times = [ref_time * min_rank / r for r in ranks]
            ax.plot(ranks, ideal_times, 'k:', linewidth=0.8, alpha=0.5, label='Ideal')

        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Time (ms)')
        ax.set_title(TYPE_LABELS[ttype], fontsize=9)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')

        if not subset.empty:
            ranks = sorted(subset['num_ranks'].unique())
            ax.set_xticks(ranks)
            ax.set_xticklabels([str(r) for r in ranks])

        if ttype == 'type1':
            ax.legend(loc='best', fontsize=6)

    # Bottom row: efficiency
    for ax, ttype in zip(axes[1], ['type1', 'type2']):
        subset = stats[stats['transform_type'] == ttype]

        for cluster in sorted(subset['cluster'].unique()):
            cluster_data = subset[subset['cluster'] == cluster].sort_values('num_ranks')

            color = CLUSTER_COLORS.get(cluster, '#999999')
            marker = CLUSTER_MARKERS.get(cluster, 'o')
            label = CLUSTER_LABELS.get(cluster, cluster)

            ax.plot(
                cluster_data['num_ranks'],
                cluster_data['efficiency'],
                marker=marker,
                color=color,
                markersize=4,
                linewidth=1,
                label=label
            )

        ax.axhline(y=100, color='k', linestyle=':', linewidth=0.8, alpha=0.5)

        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Efficiency (%)')
        ax.set_xscale('log', base=2)
        ax.set_ylim(0, 110)

        if not subset.empty:
            ranks = sorted(subset['num_ranks'].unique())
            ax.set_xticks(ranks)
            ax.set_xticklabels([str(r) for r in ranks])

    plt.tight_layout()

    plt.savefig(f'{output_prefix}_combined.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_combined.png', dpi=300, facecolor='white')
    print(f"Saved: {output_prefix}_combined.pdf, {output_prefix}_combined.png")


def print_summary(stats):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SCALING SUMMARY")
    print("=" * 80)

    stats = compute_efficiency(stats)

    for cluster in sorted(stats['cluster'].unique()):
        print(f"\n{CLUSTER_LABELS.get(cluster, cluster)}:")
        print("-" * 60)

        cluster_data = stats[stats['cluster'] == cluster]

        for ttype in ['type1', 'type2']:
            type_data = cluster_data[cluster_data['transform_type'] == ttype]
            if type_data.empty:
                continue

            print(f"\n  {TYPE_LABELS[ttype]}:")
            print(f"  {'GPUs':>6} {'Time (ms)':>12} {'Throughput':>12} {'Efficiency':>10}")
            print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10}")

            for _, row in type_data.sort_values('num_ranks').iterrows():
                print(f"  {row['num_ranks']:>6} {row['mean']:>12.1f} "
                      f"{row['throughput_mpts']:>10.1f} M/s {row['efficiency']:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Plot NUFFT scaling results')
    parser.add_argument('--dirs', '-d', nargs='+', required=True,
                        help='Directories containing benchmark results')
    parser.add_argument('--output', '-o', default='nufft_scaling',
                        help='Output file prefix')
    parser.add_argument('--efficiency', action='store_true',
                        help='Generate efficiency plot')
    parser.add_argument('--throughput', action='store_true',
                        help='Generate throughput plot')
    parser.add_argument('--combined', action='store_true',
                        help='Generate combined time+efficiency plot')
    parser.add_argument('--components', action='store_true',
                        help='Generate component breakdown scaling plot')
    parser.add_argument('--components-pct', action='store_true',
                        help='Generate component percentage breakdown plot')
    parser.add_argument('--all', action='store_true',
                        help='Generate all plot types')
    args = parser.parse_args()

    # Load data from all directories
    all_data = []
    all_component_data = []

    for d in args.dirs:
        print(f"Loading data from {d}...")
        try:
            data = load_cluster_data(d)
            all_data.append(data)
        except ValueError as e:
            print(f"  Warning: {e}")

        comp_data = load_component_data(d)
        if comp_data is not None:
            all_component_data.append(comp_data)
            print(f"  Loaded component data: {len(comp_data)} records")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(combined_df)} timing records")

        # Compute statistics
        stats = compute_statistics(combined_df)

        # Print summary
        print_summary(stats)

        # Generate plots
        if args.all:
            args.efficiency = True
            args.throughput = True
            args.combined = True
            args.components = True
            args.components_pct = True

        # Always generate time plot
        plot_scaling_time(stats, args.output)

        if args.throughput:
            plot_scaling_throughput(stats, args.output)

        if args.efficiency:
            plot_scaling_efficiency(stats, args.output)

        if args.combined:
            plot_combined(stats, args.output)

    # Component breakdown plots
    if all_component_data:
        combined_components = pd.concat(all_component_data, ignore_index=True)
        component_stats = compute_component_statistics(combined_components)

        print_component_summary(component_stats)

        if args.components or args.all:
            # Generate for each cluster
            for cluster in component_stats['cluster'].unique():
                plot_component_scaling(component_stats, args.output, cluster_filter=cluster)
                plot_component_scaling_area(component_stats, args.output, cluster_filter=cluster)

        if args.components_pct or args.all:
            for cluster in component_stats['cluster'].unique():
                plot_component_percentage_scaling(component_stats, args.output, cluster_filter=cluster)


if __name__ == '__main__':
    main()