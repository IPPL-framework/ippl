#!/usr/bin/env python3
"""
Plot NUFFT kernel benchmark results from CSV files.

Expected CSV columns:
  kernel, operation, distribution, width, tolerance, n_particles, n_grid, rho,
  total_mean_ms, total_stddev_ms, ..., throughput_Mpts_per_s, time_per_pt_ns

Optionally reads tile sweep heatmaps to use optimal tile sizes for Tiled/GridParallel.

Usage:
    python plot_kernel_benchmark.py -i benchmark.csv -o fig_kernels
    python plot_kernel_benchmark.py -i benchmark.csv -o fig_kernels --by-tolerance
    python plot_kernel_benchmark.py -i benchmark.csv -o fig_kernels --tile-sweep tile_sweep_heatmap
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def setup_style():
    """Set up publication-quality matplotlib style."""
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 8,
        'mathtext.fontset': 'stix',

        # Axes
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'axes.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.axisbelow': True,

        # Ticks
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',

        # Legend
        'legend.fontsize': 7,
        'legend.frameon': False,
        'legend.borderpad': 0.3,
        'legend.handlelength': 1.5,
        'legend.handletextpad': 0.4,

        # Figure
        'figure.facecolor': 'white',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,

        # Grid
        'grid.linewidth': 0.3,
        'grid.alpha': 0.5,

        # Lines
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
    })


# Okabe-Ito colorblind-friendly palette
COLORS = {
    # Scatter kernels
    'Atomic': '#E69F00',        # Orange
    'Tiled': '#009E73',         # Bluish green
    'GridParallel': '#0072B2',  # Blue
    # Gather kernels
    'Direct': '#E69F00',        # Orange
    'Sorted': '#56B4E9',        # Sky blue
    'TeamParallel': '#0072B2',  # Blue
    'Native': '#CC79A7',        # Reddish purple
    # External
    'cuFINUFFT': '#D55E00',     # Vermillion
}

MARKERS = {
    'Atomic': 's',
    'Tiled': '^',
    'GridParallel': 'o',
    'Direct': 's',
    'Sorted': 'D',
    'TeamParallel': 'o',
    'Native': 'v',
    'cuFINUFFT': 'X',
}

# Clean labels for plotting
LABELS = {
    'Atomic': 'Atomic',
    'Tiled': 'Tiled',
    'GridParallel': 'Grid-Parallel',
    'Direct': 'Direct',
    'Sorted': 'Sorted',
    'TeamParallel': 'Team-Parallel',
    'Native': 'Native',
    'cuFINUFFT': 'cuFINUFFT',
}


def load_data(filepath):
    """Load and preprocess benchmark CSV."""
    df = pd.read_csv(filepath)

    # Clean up column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Ensure tolerance is numeric
    if 'tolerance' in df.columns:
        df['tolerance'] = pd.to_numeric(df['tolerance'], errors='coerce')

    return df


def load_tile_heatmap(filepath):
    """Load tile sweep heatmap and extract optimal throughput per width."""
    df = pd.read_csv(filepath)

    # Extract widths from column names (width_2, width_3, etc.)
    width_cols = [c for c in df.columns if c.startswith('width_')]

    optimal = {}
    for col in width_cols:
        width = int(col.split('_')[1])
        values = df[col].values
        # Get best throughput (ignoring NaN)
        if not np.all(np.isnan(values)):
            best_idx = np.nanargmax(values)
            optimal[width] = {
                'throughput': values[best_idx],
                'tile_size': df['tile_size'].values[best_idx]
            }

    return optimal


def apply_tile_sweep_data(df, tile_sweep_prefix):
    """Replace Tiled/GridParallel throughput with optimal values from tile sweep."""

    # Load heatmaps
    tiled_file = f"{tile_sweep_prefix}_Tiled.csv"
    output_focused_file = f"{tile_sweep_prefix}_OutputFocused.csv"

    tiled_optimal = {}
    output_focused_optimal = {}

    if Path(tiled_file).exists():
        tiled_optimal = load_tile_heatmap(tiled_file)
        print(f"Loaded Tiled tile sweep from {tiled_file}")
        print(f"  Available widths: {sorted(tiled_optimal.keys())}")
    else:
        print(f"Warning: {tiled_file} not found")

    if Path(output_focused_file).exists():
        output_focused_optimal = load_tile_heatmap(output_focused_file)
        print(f"Loaded GridParallel tile sweep from {output_focused_file}")
        print(f"  Available widths: {sorted(output_focused_optimal.keys())}")
    else:
        print(f"Warning: {output_focused_file} not found")

    # Check widths in benchmark data
    scatter_df = df[df['operation'] == 'scatter']
    benchmark_widths = sorted(scatter_df['width'].unique())
    print(f"Benchmark scatter widths: {benchmark_widths}")

    # Update dataframe - use .copy() to avoid SettingWithCopyWarning
    df = df.copy()

    updates_made = 0
    for idx in df.index:
        row = df.loc[idx]
        if row['operation'] != 'scatter':
            continue

        width = int(row['width'])
        kernel = row['kernel']
        old_tput = row['throughput_Mpts_per_s']

        if kernel == 'Tiled' and width in tiled_optimal:
            opt = tiled_optimal[width]
            df.at[idx, 'throughput_Mpts_per_s'] = opt['throughput']
            n_particles = row['n_particles']
            df.at[idx, 'total_mean_ms'] = n_particles / (opt['throughput'] * 1e6) * 1000
            updates_made += 1
            print(f"  Updated Tiled w={width}: {old_tput:.0f} -> {opt['throughput']:.0f} Mpts/s")

        elif kernel == 'GridParallel' and width in output_focused_optimal:
            opt = output_focused_optimal[width]
            df.at[idx, 'throughput_Mpts_per_s'] = opt['throughput']
            n_particles = row['n_particles']
            df.at[idx, 'total_mean_ms'] = n_particles / (opt['throughput'] * 1e6) * 1000
            updates_made += 1
            print(f"  Updated GridParallel w={width}: {old_tput:.0f} -> {opt['throughput']:.0f} Mpts/s")

    print(f"Updated {updates_made} rows with tile sweep data")

    return df


def deduplicate_data(df):
    """Remove duplicate entries, keeping the one with highest throughput."""
    # Group by key columns and keep max throughput
    key_cols = ['kernel', 'operation', 'width', 'tolerance']

    # Check which key columns exist
    key_cols = [c for c in key_cols if c in df.columns]

    if not key_cols:
        return df

    # Count duplicates
    duplicates = df.duplicated(subset=key_cols, keep=False)
    n_duplicates = duplicates.sum()

    if n_duplicates > 0:
        print(f"Found {n_duplicates} duplicate entries, keeping highest throughput")
        # Keep row with highest throughput for each group
        df = df.sort_values('throughput_Mpts_per_s', ascending=False)
        df = df.drop_duplicates(subset=key_cols, keep='first')
        df = df.sort_values(['operation', 'tolerance', 'kernel'])

    return df


def print_tile_sweep_summary(tile_sweep_prefix):
    """Print optimal tile sizes from sweep data."""
    tiled_file = f"{tile_sweep_prefix}_Tiled.csv"
    output_focused_file = f"{tile_sweep_prefix}_OutputFocused.csv"

    print("\n" + "=" * 60)
    print("OPTIMAL TILE SIZES FROM SWEEP")
    print("=" * 60)

    for name, filepath in [('Tiled', tiled_file), ('Grid-Parallel', output_focused_file)]:
        if not Path(filepath).exists():
            continue

        optimal = load_tile_heatmap(filepath)
        print(f"\n{name}:")
        print(f"  {'Width':>6s} {'Tile':>6s} {'Mpts/s':>10s}")
        print("  " + "-" * 26)
        for w in sorted(optimal.keys()):
            opt = optimal[w]
            print(f"  {w:>6d} {opt['tile_size']:>6d} {opt['throughput']:>10.0f}")


def plot_bar_comparison(df, output_prefix, tolerance=1e-6):
    """Create bar chart comparing kernels at fixed tolerance."""
    setup_style()

    # Filter to specific tolerance
    tol_df = df[np.isclose(df['tolerance'], tolerance, rtol=0.1)]

    if tol_df.empty:
        # Fall back to closest tolerance
        available = df['tolerance'].unique()
        tolerance = min(available, key=lambda x: abs(np.log10(x) - np.log10(tolerance)))
        tol_df = df[np.isclose(df['tolerance'], tolerance, rtol=0.1)]
        print(f"Using tolerance {tolerance:.0e}")

    width = tol_df['width'].iloc[0]

    # Separate scatter and gather
    scatter_df = tol_df[tol_df['operation'] == 'scatter'].copy()
    gather_df = tol_df[tol_df['operation'] == 'gather'].copy()

    # Filter out Native for cleaner comparison
    gather_df = gather_df[gather_df['kernel'] != 'Native']

    # Define order
    scatter_order = ['Atomic', 'Tiled', 'GridParallel']
    gather_order = ['Direct', 'Sorted', 'TeamParallel']

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))

    # --- Scatter (Type-1) ---
    ax = axes[0]
    kernels = [k for k in scatter_order if k in scatter_df['kernel'].values]
    x = np.arange(len(kernels))
    bar_width = 0.6

    throughputs = []
    for kernel in kernels:
        row = scatter_df[scatter_df['kernel'] == kernel]
        throughputs.append(row['throughput_Mpts_per_s'].values[0] if not row.empty else 0)

    bars = ax.bar(x, throughputs, bar_width,
                  color=[COLORS.get(k, '#666666') for k in kernels],
                  edgecolor='black', linewidth=0.4, zorder=3)

    # Value labels
    for bar, val in zip(bars, throughputs):
        ax.annotate(f'{val:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=6, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(k, k) for k in kernels],
                       rotation=35, ha='right', rotation_mode='anchor')
    ax.set_ylabel('Throughput (Mpts/s)')
    ax.set_title(f'Type-1 (spreading), $w={width}$', fontsize=9, pad=6)
    ax.grid(True, axis='y', linestyle='-', linewidth=0.3, alpha=0.5, zorder=0)
    ax.set_ylim(bottom=0, top=max(throughputs) * 1.15)

    # --- Gather (Type-2) ---
    ax = axes[1]
    kernels = [k for k in gather_order if k in gather_df['kernel'].values]
    x = np.arange(len(kernels))

    throughputs = []
    for kernel in kernels:
        row = gather_df[gather_df['kernel'] == kernel]
        throughputs.append(row['throughput_Mpts_per_s'].values[0] if not row.empty else 0)

    bars = ax.bar(x, throughputs, bar_width,
                  color=[COLORS.get(k, '#666666') for k in kernels],
                  edgecolor='black', linewidth=0.4, zorder=3)

    for bar, val in zip(bars, throughputs):
        ax.annotate(f'{val:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=6, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(k, k) for k in kernels],
                       rotation=35, ha='right', rotation_mode='anchor')
    ax.set_ylabel('Throughput (Mpts/s)')
    ax.set_title(f'Type-2 (interpolation), $w={width}$', fontsize=9, pad=6)
    ax.grid(True, axis='y', linestyle='-', linewidth=0.3, alpha=0.5, zorder=0)
    ax.set_ylim(bottom=0, top=max(throughputs) * 1.15)

    plt.tight_layout(w_pad=3.0)
    plt.subplots_adjust(bottom=0.22)

    plt.savefig(f'{output_prefix}_bar.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_bar.png', dpi=300, facecolor='white')
    plt.show()

    print(f"Saved: {output_prefix}_bar.pdf, {output_prefix}_bar.png")


def plot_throughput_vs_tolerance(df, output_prefix):
    """Create line plot of throughput vs tolerance (accuracy)."""
    setup_style()

    # Filter out Native
    df = df[df['kernel'] != 'Native']

    scatter_df = df[df['operation'] == 'scatter']
    gather_df = df[df['operation'] == 'gather']

    scatter_order = ['Atomic', 'Tiled', 'GridParallel']
    #gather_order = ['Direct', 'Sorted', 'TeamParallel']
    gather_order = ['Direct', 'Sorted']

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))

    # --- Scatter ---
    ax = axes[0]
    for kernel in scatter_order:
        kdf = scatter_df[scatter_df['kernel'] == kernel].sort_values('tolerance', ascending=False)
        if kdf.empty:
            continue
        ax.plot(kdf['tolerance'], kdf['throughput_Mpts_per_s'],
                marker=MARKERS.get(kernel, 'o'),
                color=COLORS.get(kernel, '#666666'),
                label=LABELS.get(kernel, kernel),
                markeredgecolor='black', markeredgewidth=0.3)

    ax.set_xscale('log')
    ax.set_xlabel('Tolerance $\\varepsilon$')
    ax.set_ylabel('Throughput (Mpts/s)')
    ax.set_title('Type-1 (spreading)', fontsize=9, pad=6)
    ax.invert_xaxis()
    ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.5, zorder=0)
    ax.legend(loc='best')

    # --- Gather ---
    ax = axes[1]
    for kernel in gather_order:
        kdf = gather_df[gather_df['kernel'] == kernel].sort_values('tolerance', ascending=False)
        if kdf.empty:
            continue
        ax.plot(kdf['tolerance'], kdf['throughput_Mpts_per_s'],
                marker=MARKERS.get(kernel, 'o'),
                color=COLORS.get(kernel, '#666666'),
                label=LABELS.get(kernel, kernel),
                markeredgecolor='black', markeredgewidth=0.3)

    ax.set_xscale('log')
    ax.set_xlabel('Tolerance $\\varepsilon$')
    ax.set_ylabel('Throughput (Mpts/s)')
    ax.set_title('Type-2 (interpolation)', fontsize=9, pad=6)
    ax.invert_xaxis()
    ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.5, zorder=0)
    ax.legend(loc='best')

    plt.tight_layout(w_pad=3.0)

    plt.savefig(f'{output_prefix}_vs_tol.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_vs_tol.png', dpi=300, facecolor='white')
    plt.show()

    print(f"Saved: {output_prefix}_vs_tol.pdf, {output_prefix}_vs_tol.png")


def plot_throughput_vs_width(df, output_prefix):
    """Create line plot of throughput vs kernel width."""
    setup_style()

    df = df[df['kernel'] != 'Native']

    scatter_df = df[df['operation'] == 'scatter']
    gather_df = df[df['operation'] == 'gather']

    scatter_order = ['Atomic', 'Tiled', 'GridParallel']
    gather_order = ['Direct', 'Sorted', 'TeamParallel']

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))

    # --- Scatter ---
    ax = axes[0]
    for kernel in scatter_order:
        kdf = scatter_df[scatter_df['kernel'] == kernel].sort_values('width')
        if kdf.empty:
            continue
        ax.plot(kdf['width'], kdf['throughput_Mpts_per_s'],
                marker=MARKERS.get(kernel, 'o'),
                color=COLORS.get(kernel, '#666666'),
                label=LABELS.get(kernel, kernel),
                markeredgecolor='black', markeredgewidth=0.3)

    ax.set_xlabel('Kernel width $w$')
    ax.set_ylabel('Throughput (Mpts/s)')
    ax.set_title('Type-1 (spreading)', fontsize=9, pad=6)
    ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.5, zorder=0)
    ax.legend(loc='best')
    ax.set_xticks(sorted(scatter_df['width'].unique()))

    # --- Gather ---
    ax = axes[1]
    for kernel in gather_order:
        kdf = gather_df[gather_df['kernel'] == kernel].sort_values('width')
        if kdf.empty:
            continue
        ax.plot(kdf['width'], kdf['throughput_Mpts_per_s'],
                marker=MARKERS.get(kernel, 'o'),
                color=COLORS.get(kernel, '#666666'),
                label=LABELS.get(kernel, kernel),
                markeredgecolor='black', markeredgewidth=0.3)

    ax.set_xlabel('Kernel width $w$')
    ax.set_ylabel('Throughput (Mpts/s)')
    ax.set_title('Type-2 (interpolation)', fontsize=9, pad=6)
    ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.5, zorder=0)
    ax.legend(loc='best')
    ax.set_xticks(sorted(gather_df['width'].unique()))

    plt.tight_layout(w_pad=3.0)

    plt.savefig(f'{output_prefix}_vs_width.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_vs_width.png', dpi=300, facecolor='white')
    plt.show()

    print(f"Saved: {output_prefix}_vs_width.pdf, {output_prefix}_vs_width.png")


def print_summary(df):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("KERNEL BENCHMARK SUMMARY")
    print("=" * 70)

    n_particles = df['n_particles'].iloc[0]
    n_grid = df['n_grid'].iloc[0]
    rho = df['rho'].iloc[0]

    print(f"\nConfiguration: Grid {n_grid}³, {n_particles:,} particles (ρ={rho:.0f})")

    for tol in sorted(df['tolerance'].unique(), reverse=True):
        tol_df = df[np.isclose(df['tolerance'], tol, rtol=0.1)]
        width = tol_df['width'].iloc[0]

        print(f"\n--- Tolerance {tol:.0e} (w={width}) ---")

        for op, op_name in [('scatter', 'Type-1 (spreading)'), ('gather', 'Type-2 (interpolation)')]:
            op_df = tol_df[tol_df['operation'] == op].copy()
            if op_df.empty:
                continue

            op_df = op_df.sort_values('throughput_Mpts_per_s', ascending=False)
            best = op_df['throughput_Mpts_per_s'].max()

            print(f"\n  {op_name}:")
            for _, row in op_df.iterrows():
                rel = row['throughput_Mpts_per_s'] / best
                label = LABELS.get(row['kernel'], row['kernel'])
                print(f"    {label:15s}  {row['throughput_Mpts_per_s']:7.0f} Mpts/s  "
                      f"({row['total_mean_ms']:8.1f} ms)  [{rel:.2f}x]")


def print_latex_table(df, tolerance=1e-6):
    """Print results as LaTeX table."""
    tol_df = df[np.isclose(df['tolerance'], tolerance, rtol=0.1)]

    if tol_df.empty:
        available = df['tolerance'].unique()
        tolerance = min(available, key=lambda x: abs(np.log10(x) - np.log10(tolerance)))
        tol_df = df[np.isclose(df['tolerance'], tolerance, rtol=0.1)]

    width = tol_df['width'].iloc[0]
    n_particles = tol_df['n_particles'].iloc[0]
    n_grid = tol_df['n_grid'].iloc[0]

    print("\n% LaTeX table")
    print("\\begin{table}[t]")
    print("\\centering")
    print(f"\\caption{{Kernel throughput comparison. Grid $N={n_grid}^3$, "
          f"$\\rho={n_particles/n_grid**3:.0f}$, $w={width}$.}}")
    print("\\label{tab:kernel-throughput}")
    print("\\small")
    print("\\begin{tabular}{llrrr}")
    print("\\toprule")
    print("Type & Kernel & Time (ms) & Mpts/s & Speedup \\\\")
    print("\\midrule")

    for op, op_label, order in [
        ('scatter', 'Type-1', ['Atomic', 'Tiled', 'GridParallel']),
        ('gather', 'Type-2', ['Direct', 'Sorted', 'TeamParallel'])
    ]:
        op_df = tol_df[tol_df['operation'] == op]
        baseline = op_df[op_df['kernel'] == order[0]]['throughput_Mpts_per_s'].values
        baseline = baseline[0] if len(baseline) > 0 else 1

        first = True
        for kernel in order:
            row = op_df[op_df['kernel'] == kernel]
            if row.empty:
                continue

            label = LABELS.get(kernel, kernel)
            time_ms = row['total_mean_ms'].values[0]
            tput = row['throughput_Mpts_per_s'].values[0]
            speedup = tput / baseline

            type_col = op_label if first else ""
            first = False

            print(f"{type_col:7s} & {label:15s} & {time_ms:8.1f} & "
                  f"{tput:6.0f} & {speedup:.2f}$\\times$ \\\\")

        if op == 'scatter':
            print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot NUFFT kernel benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_kernel_benchmark.py -i benchmark.csv -o fig_kernels
  python plot_kernel_benchmark.py -i benchmark.csv -o fig_kernels --by-tolerance
  python plot_kernel_benchmark.py -i benchmark.csv -o fig_kernels --by-width
  python plot_kernel_benchmark.py -i benchmark.csv --latex
  python plot_kernel_benchmark.py -i benchmark.csv --tile-sweep tile_sweep_heatmap
        """
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Input CSV file')
    parser.add_argument('--output', '-o', default='kernel_benchmark',
                        help='Output file prefix')
    parser.add_argument('--tolerance', '-t', type=float, default=1e-6,
                        help='Tolerance for bar plot (default: 1e-6)')
    parser.add_argument('--by-tolerance', action='store_true',
                        help='Plot throughput vs tolerance')
    parser.add_argument('--by-width', action='store_true',
                        help='Plot throughput vs kernel width')
    parser.add_argument('--latex', action='store_true',
                        help='Print LaTeX table')
    parser.add_argument('--tile-sweep', type=str, default=None,
                        help='Tile sweep heatmap prefix (expects _Tiled.csv and _OutputFocused.csv)')
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = load_data(args.input)

    print(f"Loaded {len(df)} rows")
    print(f"Kernels: {sorted(df['kernel'].unique())}")
    print(f"Tolerances: {sorted(df['tolerance'].unique(), reverse=True)}")
    print(f"Widths: {sorted(df['width'].unique())}")

    # Deduplicate entries
    df = deduplicate_data(df)
    print(f"After deduplication: {len(df)} rows")

    # Apply tile sweep data if provided
    if args.tile_sweep:
        print_tile_sweep_summary(args.tile_sweep)
        df = apply_tile_sweep_data(df, args.tile_sweep)
        print("\nUsing optimal tile sizes for Tiled/GridParallel scatter kernels")

        # Debug: show updated values for scatter kernels
        scatter_df = df[df['operation'] == 'scatter']
        print("\nUpdated scatter kernel throughputs:")
        for w in sorted(scatter_df['width'].unique()):
            w_df = scatter_df[scatter_df['width'] == w]
            print(f"  w={w}:")
            for _, row in w_df.iterrows():
                print(f"    {row['kernel']:15s}: {row['throughput_Mpts_per_s']:.0f} Mpts/s")

    print_summary(df)

    if args.latex:
        print_latex_table(df, args.tolerance)

    # Generate plots
    plot_bar_comparison(df, args.output, args.tolerance)

    if args.by_tolerance:
        plot_throughput_vs_tolerance(df, args.output)

    if args.by_width:
        plot_throughput_vs_width(df, args.output)


if __name__ == '__main__':
    main()