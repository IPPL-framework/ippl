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
VERSION_COLORS = {
    'PIF': '#0072B2',        # Blue
    'PIFPruned': '#E69F00',  # Orange
}

VERSION_MARKERS = {
    'PIF': 'o',
    'PIFPruned': 's',
}

VERSION_LABELS = {
    'PIF': 'LandauDampingPIF',
    'PIFPruned': 'LandauDampingPIFPruned',
}

# Component colors
COMPONENT_COLORS = {
    'updateParticle': '#0072B2',      # Blue
    'GatherPIFNUFFT': '#E69F00',     # Orange
    'ScatterPIFNUFFT': '#009E73',     # Bluish green
    'Other': '#999999',               # Gray
}

COMPONENT_LABELS = {
    'updateParticle': 'Update Particle',
    'GatherPIFNUFFT': 'Gather (NUFFT)',
    'ScatterPIFNUFFT': 'Scatter (NUFFT)',
    'Other': 'Other',
}

# Stack order (bottom to top)
STACK_ORDER = ['updateParticle', 'GatherPIFNUFFT', 'ScatterPIFNUFFT', 'Other']

# Components we track
TRACKED_COMPONENTS = ['updateParticle', 'GatherPIFNUFFT', 'ScatterPIFNUFFT']


def parse_filename(filepath):
    """
    Parse filename to extract version and GPU count.

    Examples:
        LandauDampingPIF128_0.csv -> ('PIF', 128)
        LandauDampingPIFPruned64_1.csv -> ('PIFPruned', 64)
    """
    filename = Path(filepath).stem

    # Match pattern: LandauDampingPIF[Pruned]<num_gpus>_<run_id>
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

    # Find all CSV files matching pattern
    patterns = [
        'LandauDampingPIF*_*.csv',
        'LandauDampingPIFPruned*_*.csv',
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(str(base_path / pattern)))

    if not files:
        # Try finding any CSV files
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


def compute_statistics(df):
    """Compute mean, std for main timer at each GPU count.

    mainTimer captures the total simulation time across all iterations.
    We average across different runs (files) for the same GPU count.
    """
    # Filter to mainTimer only
    main_df = df[df['timer_name'] == 'mainTimer'].copy()

    # Average across ranks first (should be nearly identical), then across files
    # Group by version, num_gpus, and source_file to get one value per run
    per_run = main_df.groupby(['version', 'num_gpus', 'source_file'])['duration_seconds'].mean().reset_index()

    # Then aggregate across runs
    grouped = per_run.groupby(['version', 'num_gpus'])

    stats = grouped['duration_seconds'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    stats.columns = ['version', 'num_gpus', 'mean', 'std', 'min', 'max', 'count']

    # Convert to milliseconds
    stats['mean_ms'] = stats['mean'] * 1000
    stats['std_ms'] = stats['std'] * 1000 if 'std' in stats.columns else 0

    return stats


def compute_component_statistics(df):
    """Compute total time for each component at each GPU count.

    Note: mainTimer is over all iterations, but component timers are collected
    per iteration. So we sum the component times across all measurement_ids
    to get the total time spent in each component.
    """
    df = df.copy()

    # Filter to tracked components
    component_df = df[df['timer_name'].isin(TRACKED_COMPONENTS)].copy()

    # For each (version, num_gpus, source_file, rank, timer_name),
    # sum across all measurement_ids (iterations)
    # Then average across ranks to get per-GPU-count statistics

    # First, sum across iterations for each rank
    summed = component_df.groupby(['version', 'num_gpus', 'source_file', 'rank', 'timer_name'])['duration_seconds'].sum().reset_index()

    # Then average across ranks (they should be similar, but averaging handles any variation)
    grouped = summed.groupby(['version', 'num_gpus', 'timer_name'])
    stats = grouped['duration_seconds'].agg(['mean', 'std']).reset_index()
    stats.columns = ['version', 'num_gpus', 'component', 'mean', 'std']

    # Convert to milliseconds
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

        # Find baseline (smallest GPU count)
        min_gpus = subset['num_gpus'].min()
        baseline = subset[subset['num_gpus'] == min_gpus]['mean'].values[0]
        baseline_gpus = min_gpus

        # Compute efficiency: (T_base * N_base) / (T_n * N_n)
        for idx in subset.index:
            n_gpus = stats.loc[idx, 'num_gpus']
            time_n = stats.loc[idx, 'mean']
            ideal_time = baseline * baseline_gpus / n_gpus
            stats.loc[idx, 'efficiency'] = ideal_time / time_n * 100

    return stats


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

    # Add ideal scaling line
    if not stats.empty:
        gpu_counts = sorted(stats['num_gpus'].unique())
        min_gpus = min(gpu_counts)
        ref_time = stats[stats['num_gpus'] == min_gpus]['mean_ms'].mean()
        ideal_times = [ref_time * min_gpus / g for g in gpu_counts]
        ax.plot(gpu_counts, ideal_times, 'k:', linewidth=0.8, alpha=0.5, label='Ideal')

    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Strong Scaling: Execution Time')

    if log_scale:
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')

    ax.legend(loc='best')

    # Set x-ticks to actual GPU counts
    if not stats.empty:
        gpu_counts = sorted(stats['num_gpus'].unique())
        ax.set_xticks(gpu_counts)
        ax.set_xticklabels([str(g) for g in gpu_counts])

    plt.tight_layout()

    plt.savefig(f'{output_prefix}_time.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_time.png', dpi=300, facecolor='white')
    print(f"Saved: {output_prefix}_time.pdf, {output_prefix}_time.png")


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

    # Add 100% reference line
    ax.axhline(y=100, color='k', linestyle=':', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Parallel Efficiency (%)')
    ax.set_title('Strong Scaling: Parallel Efficiency')
    ax.set_xscale('log', base=2)
    ax.set_ylim(0, 110)

    ax.legend(loc='best')

    # Set x-ticks
    if not stats.empty:
        gpu_counts = sorted(stats['num_gpus'].unique())
        ax.set_xticks(gpu_counts)
        ax.set_xticklabels([str(g) for g in gpu_counts])

    plt.tight_layout()

    plt.savefig(f'{output_prefix}_efficiency.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_efficiency.png', dpi=300, facecolor='white')
    print(f"Saved: {output_prefix}_efficiency.pdf, {output_prefix}_efficiency.png")


def plot_combined(stats, output_prefix):
    """Plot time and efficiency side by side."""
    setup_style()

    stats = compute_efficiency(stats)

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.8))

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
            markersize=4,
            linewidth=1,
            capsize=2,
            capthick=0.5,
            label=label
        )

    # Ideal scaling
    if not stats.empty:
        gpu_counts = sorted(stats['num_gpus'].unique())
        min_gpus = min(gpu_counts)
        ref_time = stats[stats['num_gpus'] == min_gpus]['mean_ms'].mean()
        ideal_times = [ref_time * min_gpus / g for g in gpu_counts]
        ax.plot(gpu_counts, ideal_times, 'k:', linewidth=0.8, alpha=0.5, label='Ideal')

    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Execution Time')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=6)

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
            markersize=4,
            linewidth=1,
            label=label
        )

    ax.axhline(y=100, color='k', linestyle=':', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title('Parallel Efficiency')
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


def plot_component_scaling(component_stats, main_stats, output_prefix, version_filter=None):
    """Plot stacked bar chart of component times vs GPU count."""
    setup_style()

    if version_filter:
        component_stats = component_stats[component_stats['version'] == version_filter]
        main_stats = main_stats[main_stats['version'] == version_filter]
        version_label = VERSION_LABELS.get(version_filter, version_filter)
    else:
        version_filter = component_stats['version'].iloc[0]
        component_stats = component_stats[component_stats['version'] == version_filter]
        main_stats = main_stats[main_stats['version'] == version_filter]
        version_label = VERSION_LABELS.get(version_filter, version_filter)

    fig, ax = plt.subplots(figsize=(5, 3))

    gpu_counts = sorted(component_stats['num_gpus'].unique())
    n_gpus = len(gpu_counts)
    x = np.arange(n_gpus)
    bar_width = 0.65

    for gpu_idx, gpu_count in enumerate(gpu_counts):
        gpu_data = component_stats[component_stats['num_gpus'] == gpu_count]
        main_data = main_stats[main_stats['num_gpus'] == gpu_count]

        # Get total time
        total_time = main_data['mean_ms'].values[0] if not main_data.empty else 0

        bottom = 0
        tracked_sum = 0

        for component in TRACKED_COMPONENTS:
            comp_data = gpu_data[gpu_data['component'] == component]
            if not comp_data.empty:
                height = comp_data['mean_ms'].values[0]
                tracked_sum += height
                color = COMPONENT_COLORS.get(component, '#999999')
                label = COMPONENT_LABELS.get(component) if gpu_idx == 0 else None

                ax.bar(x[gpu_idx], height, bar_width, bottom=bottom,
                       color=color, edgecolor='black', linewidth=0.3,
                       label=label)
                bottom += height

        # Add "Other" category
        other_time = max(0, total_time - tracked_sum)
        if other_time > 0:
            label = COMPONENT_LABELS.get('Other') if gpu_idx == 0 else None
            ax.bar(x[gpu_idx], other_time, bar_width, bottom=bottom,
                   color=COMPONENT_COLORS['Other'], edgecolor='black', linewidth=0.3,
                   label=label)
            bottom += other_time

        # Total annotation
        ax.annotate(f'{bottom:.0f}',
                    xy=(x[gpu_idx], bottom),
                    xytext=(0, 2), textcoords='offset points',
                    ha='center', va='bottom', fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in gpu_counts])
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Component Breakdown: {version_label}')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=6)

    plt.tight_layout()

    suffix = f'_{version_filter}' if version_filter else ''
    plt.savefig(f'{output_prefix}_components{suffix}.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_components{suffix}.png', dpi=300, facecolor='white')
    print(f"Saved: {output_prefix}_components{suffix}.pdf")


def plot_component_percentage(component_stats, main_stats, output_prefix, version_filter=None):
    """Plot stacked bar chart showing percentage breakdown vs GPU count."""
    setup_style()

    if version_filter:
        component_stats = component_stats[component_stats['version'] == version_filter]
        main_stats = main_stats[main_stats['version'] == version_filter]
        version_label = VERSION_LABELS.get(version_filter, version_filter)
    else:
        version_filter = component_stats['version'].iloc[0]
        component_stats = component_stats[component_stats['version'] == version_filter]
        main_stats = main_stats[main_stats['version'] == version_filter]
        version_label = VERSION_LABELS.get(version_filter, version_filter)

    fig, ax = plt.subplots(figsize=(5, 3))

    gpu_counts = sorted(component_stats['num_gpus'].unique())
    n_gpus = len(gpu_counts)
    x = np.arange(n_gpus)
    bar_width = 0.65

    for gpu_idx, gpu_count in enumerate(gpu_counts):
        gpu_data = component_stats[component_stats['num_gpus'] == gpu_count]
        main_data = main_stats[main_stats['num_gpus'] == gpu_count]

        # Get total time
        total_time = main_data['mean_ms'].values[0] if not main_data.empty else 0
        if total_time == 0:
            continue

        bottom = 0
        tracked_sum = 0

        for component in TRACKED_COMPONENTS:
            comp_data = gpu_data[gpu_data['component'] == component]
            if not comp_data.empty:
                time_ms = comp_data['mean_ms'].values[0]
                tracked_sum += time_ms
                height = time_ms / total_time * 100
                color = COMPONENT_COLORS.get(component, '#999999')
                label = COMPONENT_LABELS.get(component) if gpu_idx == 0 else None

                ax.bar(x[gpu_idx], height, bar_width, bottom=bottom,
                       color=color, edgecolor='black', linewidth=0.3,
                       label=label)

                # Add percentage label if > 10%
                if height > 10:
                    ax.text(x[gpu_idx], bottom + height/2, f'{height:.0f}%',
                            ha='center', va='center', fontsize=5,
                            color='white', fontweight='bold')

                bottom += height

        # Add "Other" category
        other_time = max(0, total_time - tracked_sum)
        if other_time > 0:
            height = other_time / total_time * 100
            label = COMPONENT_LABELS.get('Other') if gpu_idx == 0 else None
            ax.bar(x[gpu_idx], height, bar_width, bottom=bottom,
                   color=COMPONENT_COLORS['Other'], edgecolor='black', linewidth=0.3,
                   label=label)

            if height > 10:
                ax.text(x[gpu_idx], bottom + height/2, f'{height:.0f}%',
                        ha='center', va='center', fontsize=5,
                        color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in gpu_counts])
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'Component Breakdown: {version_label}')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', fontsize=6)

    plt.tight_layout()

    suffix = f'_{version_filter}' if version_filter else ''
    plt.savefig(f'{output_prefix}_components_pct{suffix}.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_components_pct{suffix}.png', dpi=300, facecolor='white')
    print(f"Saved: {output_prefix}_components_pct{suffix}.pdf")


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


def print_component_summary(component_stats, main_stats):
    """Print summary of component breakdown."""
    print("\n" + "=" * 70)
    print("COMPONENT BREAKDOWN SUMMARY")
    print("=" * 70)

    for version in sorted(component_stats['version'].unique()):
        print(f"\n{VERSION_LABELS.get(version, version)}:")
        print("-" * 60)

        ver_comp = component_stats[component_stats['version'] == version]
        ver_main = main_stats[main_stats['version'] == version]

        gpu_counts = sorted(ver_comp['num_gpus'].unique())

        # Header
        header = f"  {'Component':<20}"
        for g in gpu_counts:
            header += f" {g:>8} GPUs"
        print(header)
        print("  " + "-" * (20 + 13 * len(gpu_counts)))

        for component in TRACKED_COMPONENTS:
            row = f"  {COMPONENT_LABELS.get(component, component):<20}"
            for g in gpu_counts:
                comp_data = ver_comp[(ver_comp['num_gpus'] == g) &
                                     (ver_comp['component'] == component)]
                if not comp_data.empty:
                    row += f" {comp_data['mean_ms'].values[0]:>8.1f} ms"
                else:
                    row += f" {'---':>11}"
            print(row)

        # Total row
        row = f"  {'Total':<20}"
        for g in gpu_counts:
            main_data = ver_main[ver_main['num_gpus'] == g]
            if not main_data.empty:
                row += f" {main_data['mean_ms'].values[0]:>8.1f} ms"
            else:
                row += f" {'---':>11}"
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
                        help='Generate component breakdown plot')
    parser.add_argument('--components-pct', action='store_true',
                        help='Generate component percentage breakdown plot')
    parser.add_argument('--all', action='store_true',
                        help='Generate all plot types')
    args = parser.parse_args()

    # Load data from all directories
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

    # Compute statistics
    stats = compute_statistics(combined_df)
    component_stats = compute_component_statistics(combined_df)

    # Print summary
    print_summary(stats)

    if not component_stats.empty:
        print_component_summary(component_stats, stats)

    # Generate plots
    if args.all:
        args.efficiency = True
        args.combined = True
        args.components = True
        args.components_pct = True

    # Always generate time plot
    plot_scaling_time(stats, args.output)

    if args.efficiency:
        plot_scaling_efficiency(stats, args.output)

    if args.combined:
        plot_combined(stats, args.output)

    # Component breakdown plots (per version)
    if args.components and not component_stats.empty:
        for version in component_stats['version'].unique():
            plot_component_scaling(component_stats, stats, args.output, version_filter=version)

    if args.components_pct and not component_stats.empty:
        for version in component_stats['version'].unique():
            plot_component_percentage(component_stats, stats, args.output, version_filter=version)


if __name__ == '__main__':
    main()