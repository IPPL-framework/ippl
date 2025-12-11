#!/usr/bin/env python3
"""
Plot single-GPU NUFFT benchmark results comparing IPPL implementations vs cuFINUFFT.

Expected input format (text files):
Configuration: Grid=256^3, Particles=167772160 (10 per point)
================================================================================
  IPPL OutputFocused | Type 1 | Grid: 256^3 | Particles:  167772160 | Time:    335.935 ms | Throughput:   499.42 Mpts/s
  ...

Usage:
    python plot_nufft_benchmark.py [--input-dir DIR] [--output PREFIX]
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def parse_benchmark_file(filepath):
    """Parse a single benchmark results file."""
    results = []
    current_grid = None
    current_particles = None
    cluster_name = filepath.stem  # Use filename as cluster identifier

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Parse configuration header
            config_match = re.match(
                r'Configuration: Grid=(\d+)\^3, Particles=(\d+)',
                line
            )
            if config_match:
                current_grid = int(config_match.group(1))
                current_particles = int(config_match.group(2))
                continue

            # Parse result line
            # Format: METHOD | Type N | Grid: X^3 | Particles: Y | Time: Z ms | Throughput: W Mpts/s
            result_match = re.match(
                r'^\s*(.+?)\s*\|\s*Type\s+(\d+)\s*\|\s*Grid:\s*(\d+)\^3\s*\|'
                r'\s*Particles:\s*(\d+)\s*\|\s*Time:\s*([\d.]+)\s*ms\s*\|'
                r'\s*Throughput:\s*([\d.]+)\s*Mpts/s',
                line
            )
            if result_match:
                method = result_match.group(1).strip()
                nufft_type = int(result_match.group(2))
                grid_size = int(result_match.group(3))
                particles = int(result_match.group(4))
                time_ms = float(result_match.group(5))
                throughput = float(result_match.group(6))

                results.append({
                    'cluster': cluster_name,
                    'method': method,
                    'type': nufft_type,
                    'grid_size': grid_size,
                    'particles': particles,
                    'time_ms': time_ms,
                    'throughput_mpts': throughput
                })

    return results


def load_all_results(input_dir):
    """Load results from all .txt files in directory."""
    all_results = []
    input_path = Path(input_dir)

    for filepath in input_path.glob('*.out'):
        results = parse_benchmark_file(filepath)
        all_results.extend(results)

    if not all_results:
        # Also try files without extension or other patterns
        for filepath in input_path.iterdir():
            if filepath.is_file() and not filepath.suffix:
                results = parse_benchmark_file(filepath)
                all_results.extend(results)

    return pd.DataFrame(all_results)


def filter_methods(df):
    """Filter out methods we don't want to plot."""
    exclude_methods = ['kokkos_nufft', 'IPPL Native']
    return df[~df['method'].isin(exclude_methods)]


def create_method_labels(method):
    """Create clean labels for plotting."""
    label_map = {
        'IPPL OutputFocused': 'Grid-Parallel',
        'IPPL Tiled': 'Tiled',
        'IPPL Atomic': 'Atomic',
        'IPPL Atomic Sort': 'Atomic-Sort',
        'cuFINUFFT': 'cuFINUFFT'
    }
    return label_map.get(method, method)


def plot_benchmark_results(df, output_prefix='nufft_benchmark'):
    """Create publication-quality benchmark plots."""

    # Set up professional style
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

    # Filter methods
    df = filter_methods(df)

    # Get unique values
    clusters = sorted(df['cluster'].unique())
    grid_sizes = sorted(df['grid_size'].unique())

    # Define colors for methods
    method_colors = {
        'IPPL OutputFocused': '#2ca02c',  # Green
        'IPPL Tiled': '#1f77b4',           # Blue
        'IPPL Atomic': '#ff7f0e',          # Orange
        'IPPL Atomic Sort': '#9467bd',     # Purple
        'cuFINUFFT': '#d62728',            # Red
    }

    # Define method order (for consistent plotting)
    type1_methods = ['IPPL Atomic', 'IPPL Tiled', 'IPPL OutputFocused', 'cuFINUFFT']
    type2_methods = ['IPPL Atomic', 'IPPL Atomic Sort', 'IPPL Tiled', 'cuFINUFFT']

    # Create figure: 2 rows (Type 1, Type 2) x N columns (clusters or grid sizes)
    # Decide layout based on data
    if len(clusters) > 1:
        ncols = len(clusters)
        group_by = 'cluster'
        groups = clusters
    elif len(grid_sizes) > 1:
        ncols = len(grid_sizes)
        group_by = 'grid_size'
        groups = grid_sizes
    else:
        ncols = 1
        group_by = 'cluster'
        groups = clusters if clusters else ['results']

    fig, axes = plt.subplots(2, ncols, figsize=(3.5 * ncols, 4))

    # Handle single column case
    if ncols == 1:
        axes = axes.reshape(2, 1)

    bar_width = 0.7 / max(len(type1_methods), len(type2_methods))

    for col, group in enumerate(groups):
        # Filter data for this group
        if group_by == 'cluster':
            group_df = df[df['cluster'] == group]
            title_suffix = f'({group})'
        else:
            group_df = df[df['grid_size'] == group]
            title_suffix = f'($N={group}^3$)'

        # --- Type 1 (Spreading) ---
        ax1 = axes[0, col]
        type1_df = group_df[group_df['type'] == 1]

        methods_present = [m for m in type1_methods if m in type1_df['method'].values]
        x = np.arange(len(methods_present))

        for i, method in enumerate(methods_present):
            method_data = type1_df[type1_df['method'] == method]
            if not method_data.empty:
                throughput = method_data['throughput_mpts'].values[0]
                color = method_colors.get(method, '#333333')
                ax1.bar(i, throughput, bar_width * 3, color=color, alpha=0.8)

        ax1.set_xticks(x)
        ax1.set_xticklabels([create_method_labels(m) for m in methods_present],
                            rotation=45, ha='right')
        ax1.set_ylabel('Throughput (Mpts/s)')
        ax1.set_title(f'Type-1 (Spreading) {title_suffix}', fontweight='bold', pad=3)
        ax1.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=0.5)

        # --- Type 2 (Interpolation) ---
        ax2 = axes[1, col]
        type2_df = group_df[group_df['type'] == 2]

        methods_present = [m for m in type2_methods if m in type2_df['method'].values]
        x = np.arange(len(methods_present))

        for i, method in enumerate(methods_present):
            method_data = type2_df[type2_df['method'] == method]
            if not method_data.empty:
                throughput = method_data['throughput_mpts'].values[0]
                color = method_colors.get(method, '#333333')
                ax2.bar(i, throughput, bar_width * 3, color=color, alpha=0.8)

        ax2.set_xticks(x)
        ax2.set_xticklabels([create_method_labels(m) for m in methods_present],
                            rotation=45, ha='right')
        ax2.set_ylabel('Throughput (Mpts/s)')
        ax2.set_title(f'Type-2 (Interpolation) {title_suffix}', fontweight='bold', pad=3)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{output_prefix}.pdf', bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"Plots saved to {output_prefix}.png and {output_prefix}.pdf")


def plot_comparison_bars(df, output_prefix='nufft_comparison'):
    """Create grouped bar chart comparing methods across clusters."""

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
        'legend.frameon': False,
        'figure.facecolor': 'white',
    })

    df = filter_methods(df)
    clusters = sorted(df['cluster'].unique())

    # Colors for clusters
    cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    type1_methods = ['IPPL Atomic', 'IPPL Tiled', 'IPPL OutputFocused', 'cuFINUFFT']
    type2_methods = ['IPPL Atomic', 'IPPL Atomic Sort', 'IPPL Tiled', 'cuFINUFFT']

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # --- Type 1 ---
    ax1 = axes[0]
    type1_df = df[df['type'] == 1]
    methods_present = [m for m in type1_methods if m in type1_df['method'].values]

    x = np.arange(len(methods_present))
    width = 0.8 / len(clusters)

    for i, cluster in enumerate(clusters):
        cluster_data = type1_df[type1_df['cluster'] == cluster]
        throughputs = []
        for method in methods_present:
            method_data = cluster_data[cluster_data['method'] == method]
            if not method_data.empty:
                throughputs.append(method_data['throughput_mpts'].values[0])
            else:
                throughputs.append(0)

        offset = (i - len(clusters)/2 + 0.5) * width
        ax1.bar(x + offset, throughputs, width, label=cluster,
                color=cluster_colors[i % len(cluster_colors)], alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels([create_method_labels(m) for m in methods_present],
                        rotation=45, ha='right')
    ax1.set_ylabel('Throughput (Mpts/s)')
    ax1.set_title('Type-1 (Spreading)', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=0.5)

    # --- Type 2 ---
    ax2 = axes[1]
    type2_df = df[df['type'] == 2]
    methods_present = [m for m in type2_methods if m in type2_df['method'].values]

    x = np.arange(len(methods_present))

    for i, cluster in enumerate(clusters):
        cluster_data = type2_df[type2_df['cluster'] == cluster]
        throughputs = []
        for method in methods_present:
            method_data = cluster_data[cluster_data['method'] == method]
            if not method_data.empty:
                throughputs.append(method_data['throughput_mpts'].values[0])
            else:
                throughputs.append(0)

        offset = (i - len(clusters)/2 + 0.5) * width
        ax2.bar(x + offset, throughputs, width, label=cluster,
                color=cluster_colors[i % len(cluster_colors)], alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels([create_method_labels(m) for m in methods_present],
                        rotation=45, ha='right')
    ax2.set_ylabel('Throughput (Mpts/s)')
    ax2.set_title('Type-2 (Interpolation)', fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{output_prefix}.pdf', bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"Plots saved to {output_prefix}.png and {output_prefix}.pdf")


def print_summary_table(df):
    """Print a summary table of results."""
    df = filter_methods(df)

    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    for cluster in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster]
        print(f"\n{cluster}:")
        print("-" * 60)

        for nufft_type in [1, 2]:
            type_df = cluster_df[cluster_df['type'] == nufft_type]
            if type_df.empty:
                continue

            type_name = "Type-1 (Spreading)" if nufft_type == 1 else "Type-2 (Interpolation)"
            print(f"\n  {type_name}:")

            # Sort by throughput descending
            type_df = type_df.sort_values('throughput_mpts', ascending=False)

            best_throughput = type_df['throughput_mpts'].max()

            for _, row in type_df.iterrows():
                speedup = row['throughput_mpts'] / best_throughput
                label = create_method_labels(row['method'])
                print(f"    {label:20s}: {row['throughput_mpts']:8.1f} Mpts/s "
                      f"({row['time_ms']:8.2f} ms) [{speedup:.2f}x]")


def main():
    parser = argparse.ArgumentParser(description='Plot NUFFT benchmark results')
    parser.add_argument('--input-dir', '-i', default='.',
                        help='Directory containing benchmark result files')
    parser.add_argument('--output', '-o', default='nufft_benchmark',
                        help='Output file prefix')
    parser.add_argument('--comparison', '-c', action='store_true',
                        help='Create cross-cluster comparison plot')
    args = parser.parse_args()

    # Load data
    print(f"Loading results from {args.input_dir}...")
    df = load_all_results(args.input_dir)

    if df.empty:
        print("No results found! Check input directory and file format.")
        return

    print(f"Loaded {len(df)} results from {df['cluster'].nunique()} cluster(s)")
    print(f"Grid sizes: {sorted(df['grid_size'].unique())}")
    print(f"Methods: {sorted(df['method'].unique())}")

    # Print summary
    print_summary_table(df)

    # Create plots
    plot_benchmark_results(df, args.output)

    if args.comparison and df['cluster'].nunique() > 1:
        plot_comparison_bars(df, f'{args.output}_comparison')


if __name__ == '__main__':
    main()