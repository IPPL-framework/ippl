#!/usr/bin/env python3
"""
Plot NUFFT time breakdown as stacked bars across grid sizes (N).
Separate panels for Type-1 (spreading) and Type-2 (interpolation).

Input format (from IpplTimings::dumpToCSV or combined CSV):
  Option 1 (single run): timer_name,rank,measurement_id,duration_seconds
  Option 2 (combined):   grid,rho,tolerance,timer,run,time_s

Usage:
    python plot_nufft_breakdown.py -i breakdown_combined.csv -o fig_breakdown
    python plot_nufft_breakdown.py -i breakdown_256_rho10.csv -o fig_breakdown --single
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


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
# Consistent colors for corresponding operations in Type-1 and Type-2
COLORS = {
    # Spreading (T1) <-> Interpolation (T2) - main kernel operations
    'Scatter/Gather': '#0072B2',   # Blue
    # FFT - same in both
    'FFT': '#CC79A7',              # Sky blue
    # Halo exchange - same in both
    'Halo': '#56B4E9',             # Pink
    # Deconv (T1) <-> Precorr (T2) - correction operations
    'Deconv/Precorr': '#E69F00',   # Orange
}

# Map timer names to display categories (unified naming)
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

# Unified stack order (bottom to top) - same for both types
STACK_ORDER = ['Scatter/Gather', 'FFT', 'Deconv/Precorr', 'Halo']

# Labels for legend (different per type)
LABELS_T1 = {
    'Scatter/Gather': 'Spreading',
    'FFT': 'FFT',
    'Halo': 'Halo',
    'Deconv/Precorr': 'Deconvolution',
}

LABELS_T2 = {
    'Scatter/Gather': 'Interpolation',
    'FFT': 'FFT',
    'Halo': 'Halo',
    'Deconv/Precorr': 'Precorrection',
}


def load_single_csv(filepath):
    """Load a single benchmark CSV file."""
    df = pd.read_csv(filepath, header=None,
                     names=['timer', 'rank', 'run', 'time_s'])

    # Skip header row if present
    if df.iloc[0]['timer'] == 'timer_name':
        df = df.iloc[1:]

    df['time_s'] = pd.to_numeric(df['time_s'], errors='coerce')
    df['time_ms'] = df['time_s'] * 1000

    return df


def load_combined_csv(filepath):
    """Load combined CSV with multiple grid sizes."""
    df = pd.read_csv(filepath)
    df['time_ms'] = df['time_s'] * 1000
    return df


def aggregate_by_category(df, timer_map, group_cols=None):
    """Aggregate timer data into display categories."""
    if group_cols is None:
        group_cols = []

    df = df.copy()
    df['category'] = df['timer'].map(timer_map)
    df = df.dropna(subset=['category'])

    agg_cols = group_cols + ['category']
    summary = df.groupby(agg_cols)['time_ms'].agg(['mean', 'std']).reset_index()

    return summary


def plot_breakdown_vs_gridsize(df, output_prefix):
    """Create side-by-side stacked bars for Type-1 and Type-2 vs grid size."""
    setup_style()

    grid_sizes = sorted(df['grid'].unique())
    n_grids = len(grid_sizes)

    # Separate Type-1 and Type-2 timers
    type1_timers = list(TIMER_CATEGORIES_T1.keys())
    type2_timers = list(TIMER_CATEGORIES_T2.keys())

    df_t1 = df[df['timer'].isin(type1_timers)].copy()
    df_t2 = df[df['timer'].isin(type2_timers)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    x = np.arange(n_grids)
    bar_width = 0.65

    # Type-1 panel
    ax = axes[0]
    summary_t1 = aggregate_by_category(df_t1, TIMER_CATEGORIES_T1, group_cols=['grid'])

    for grid_idx, grid in enumerate(grid_sizes):
        grid_data = summary_t1[summary_t1['grid'] == grid]
        bottom = 0

        for category in STACK_ORDER:
            cat_data = grid_data[grid_data['category'] == category]
            if not cat_data.empty:
                height = cat_data['mean'].values[0]
                color = COLORS.get(category, '#999999')
                label = LABELS_T1.get(category) if grid_idx == 0 else None

                ax.bar(x[grid_idx], height, bar_width, bottom=bottom,
                       color=color, edgecolor='black', linewidth=0.3,
                       label=label)
                bottom += height

        # Total annotation
        ax.annotate(f'{bottom:.0f}',
                    xy=(x[grid_idx], bottom),
                    xytext=(0, 2), textcoords='offset points',
                    ha='center', va='bottom', fontsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f'${g}^3$' for g in grid_sizes])
    ax.set_xlabel('Grid size $N$')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Type-1 (spreading)', fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', fontsize=6)

    # Type-2 panel
    ax = axes[1]
    summary_t2 = aggregate_by_category(df_t2, TIMER_CATEGORIES_T2, group_cols=['grid'])

    for grid_idx, grid in enumerate(grid_sizes):
        grid_data = summary_t2[summary_t2['grid'] == grid]
        bottom = 0

        for category in STACK_ORDER:
            cat_data = grid_data[grid_data['category'] == category]
            if not cat_data.empty:
                height = cat_data['mean'].values[0]
                color = COLORS.get(category, '#999999')
                label = LABELS_T2.get(category) if grid_idx == 0 else None

                ax.bar(x[grid_idx], height, bar_width, bottom=bottom,
                       color=color, edgecolor='black', linewidth=0.3,
                       label=label)
                bottom += height

        # Total annotation
        ax.annotate(f'{bottom:.0f}',
                    xy=(x[grid_idx], bottom),
                    xytext=(0, 2), textcoords='offset points',
                    ha='center', va='bottom', fontsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f'${g}^3$' for g in grid_sizes])
    ax.set_xlabel('Grid size $N$')
    ax.set_title('Type-2 (interpolation)', fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', fontsize=6)

    plt.tight_layout()

    plt.savefig(f'{output_prefix}.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}.png', dpi=300, facecolor='white')
    print(f"Saved: {output_prefix}.pdf, {output_prefix}.png")


def plot_single_breakdown(df, output_prefix, title=None):
    """Create side-by-side stacked bars for Type-1 and Type-2 (single grid size)."""
    setup_style()

    type1_timers = list(TIMER_CATEGORIES_T1.keys())
    type2_timers = list(TIMER_CATEGORIES_T2.keys())

    df_t1 = df[df['timer'].isin(type1_timers)].copy()
    df_t2 = df[df['timer'].isin(type2_timers)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(4, 2.5), sharey=True)

    # Type-1
    ax = axes[0]
    summary_t1 = aggregate_by_category(df_t1, TIMER_CATEGORIES_T1)

    bottom = 0
    total = 0
    bars_data = []

    for category in STACK_ORDER:
        cat_data = summary_t1[summary_t1['category'] == category]
        if not cat_data.empty:
            height = cat_data['mean'].values[0]
            color = COLORS.get(category, '#999999')
            label = LABELS_T1.get(category)

            ax.bar(0, height, 0.6, bottom=bottom,
                   color=color, edgecolor='black', linewidth=0.3,
                   label=label)
            bars_data.append((category, height, bottom + height/2))
            bottom += height
            total += height

    # Percentage labels
    for category, height, y_center in bars_data:
        pct = 100 * height / total
        if pct > 8:
            ax.text(0, y_center, f'{pct:.0f}%',
                    ha='center', va='center', fontsize=6, color='white',
                    fontweight='bold')

    ax.annotate(f'{total:.0f} ms',
                xy=(0, total),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=7)

    ax.set_xticks([])
    ax.set_ylabel('Time (ms)')
    ax.set_title('Type-1 (spreading)', fontsize=9)
    ax.set_xlim(-0.5, 0.5)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=6)

    # Type-2
    ax = axes[1]
    summary_t2 = aggregate_by_category(df_t2, TIMER_CATEGORIES_T2)

    bottom = 0
    total = 0
    bars_data = []

    for category in STACK_ORDER:
        cat_data = summary_t2[summary_t2['category'] == category]
        if not cat_data.empty:
            height = cat_data['mean'].values[0]
            color = COLORS.get(category, '#999999')
            label = LABELS_T2.get(category)

            ax.bar(0, height, 0.6, bottom=bottom,
                   color=color, edgecolor='black', linewidth=0.3,
                   label=label)
            bars_data.append((category, height, bottom + height/2))
            bottom += height
            total += height

    for category, height, y_center in bars_data:
        pct = 100 * height / total
        if pct > 8:
            ax.text(0, y_center, f'{pct:.0f}%',
                    ha='center', va='center', fontsize=6, color='white',
                    fontweight='bold')

    ax.annotate(f'{total:.0f} ms',
                xy=(0, total),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=7)

    ax.set_xticks([])
    ax.set_title('Type-2 (interpolation)', fontsize=9)
    ax.set_xlim(-0.5, 0.5)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=6)

    plt.tight_layout()

    plt.savefig(f'{output_prefix}.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}.png', dpi=300, facecolor='white')
    print(f"Saved: {output_prefix}.pdf, {output_prefix}.png")


def print_summary_table(df):
    """Print text summary of timing breakdown."""
    print("\n" + "=" * 70)
    print("TIME BREAKDOWN SUMMARY")
    print("=" * 70)

    has_grid = 'grid' in df.columns

    type1_timers = list(TIMER_CATEGORIES_T1.keys())
    type2_timers = list(TIMER_CATEGORIES_T2.keys())

    df_t1 = df[df['timer'].isin(type1_timers)]
    df_t2 = df[df['timer'].isin(type2_timers)]

    if has_grid:
        for grid in sorted(df['grid'].unique()):
            print(f"\nGrid size: {grid}^3")
            print("=" * 50)

            # Type-1
            grid_t1 = df_t1[df_t1['grid'] == grid]
            summary_t1 = aggregate_by_category(grid_t1, TIMER_CATEGORIES_T1)
            total_t1 = summary_t1['mean'].sum()

            print(f"\nType-1 (spreading):")
            print("-" * 40)
            print(f"{'Component':<20} {'Time (ms)':>10} {'%':>8}")
            print("-" * 40)

            for category in STACK_ORDER:
                cat_data = summary_t1[summary_t1['category'] == category]
                if not cat_data.empty:
                    t = cat_data['mean'].values[0]
                    pct = 100 * t / total_t1 if total_t1 > 0 else 0
                    label = LABELS_T1.get(category, category)
                    print(f"{label:<20} {t:>10.1f} {pct:>7.1f}%")
            print(f"{'Total':<20} {total_t1:>10.1f}")

            # Type-2
            grid_t2 = df_t2[df_t2['grid'] == grid]
            summary_t2 = aggregate_by_category(grid_t2, TIMER_CATEGORIES_T2)
            total_t2 = summary_t2['mean'].sum()

            print(f"\nType-2 (interpolation):")
            print("-" * 40)
            print(f"{'Component':<20} {'Time (ms)':>10} {'%':>8}")
            print("-" * 40)

            for category in STACK_ORDER:
                cat_data = summary_t2[summary_t2['category'] == category]
                if not cat_data.empty:
                    t = cat_data['mean'].values[0]
                    pct = 100 * t / total_t2 if total_t2 > 0 else 0
                    label = LABELS_T2.get(category, category)
                    print(f"{label:<20} {t:>10.1f} {pct:>7.1f}%")
            print(f"{'Total':<20} {total_t2:>10.1f}")
    else:
        # Single grid
        summary_t1 = aggregate_by_category(df_t1, TIMER_CATEGORIES_T1)
        total_t1 = summary_t1['mean'].sum()

        print(f"\nType-1 (spreading):")
        print("-" * 40)
        for category in STACK_ORDER:
            cat_data = summary_t1[summary_t1['category'] == category]
            if not cat_data.empty:
                t = cat_data['mean'].values[0]
                pct = 100 * t / total_t1
                label = LABELS_T1.get(category, category)
                print(f"{label:<20} {t:>10.1f} {pct:>7.1f}%")
        print(f"{'Total':<20} {total_t1:>10.1f}")

        summary_t2 = aggregate_by_category(df_t2, TIMER_CATEGORIES_T2)
        total_t2 = summary_t2['mean'].sum()

        print(f"\nType-2 (interpolation):")
        print("-" * 40)
        for category in STACK_ORDER:
            cat_data = summary_t2[summary_t2['category'] == category]
            if not cat_data.empty:
                t = cat_data['mean'].values[0]
                pct = 100 * t / total_t2
                label = LABELS_T2.get(category, category)
                print(f"{label:<20} {t:>10.1f} {pct:>7.1f}%")
        print(f"{'Total':<20} {total_t2:>10.1f}")


def print_latex_table(df):
    """Print LaTeX table."""
    has_grid = 'grid' in df.columns

    type1_timers = list(TIMER_CATEGORIES_T1.keys())
    type2_timers = list(TIMER_CATEGORIES_T2.keys())

    df_t1 = df[df['timer'].isin(type1_timers)]
    df_t2 = df[df['timer'].isin(type2_timers)]

    print("\n% LaTeX table")

    if has_grid:
        grid_sizes = sorted(df['grid'].unique())

        print("\\begin{table}[t]")
        print("\\centering")
        print("\\caption{NUFFT time breakdown (ms) by grid size.}")
        print("\\label{tab:nufft-breakdown}")
        print("\\small")

        # Type-1 table
        print("\\begin{subtable}{\\columnwidth}")
        print("\\centering")
        print("\\caption{Type-1 (spreading)}")

        header = "Component"
        for g in grid_sizes:
            header += f" & ${g}^3$"
        header += " \\\\"

        print("\\begin{tabular}{l" + "r" * len(grid_sizes) + "}")
        print("\\toprule")
        print(header)
        print("\\midrule")

        summary_t1 = aggregate_by_category(df_t1, TIMER_CATEGORIES_T1, group_cols=['grid'])

        for category in STACK_ORDER:
            label = LABELS_T1.get(category, category)
            row = label
            for g in grid_sizes:
                cat_data = summary_t1[(summary_t1['grid'] == g) &
                                      (summary_t1['category'] == category)]
                if not cat_data.empty:
                    row += f" & {cat_data['mean'].values[0]:.1f}"
                else:
                    row += " & ---"
            row += " \\\\"
            print(row)

        print("\\midrule")
        row = "Total"
        for g in grid_sizes:
            g_data = summary_t1[summary_t1['grid'] == g]
            total = g_data['mean'].sum()
            row += f" & {total:.1f}"
        row += " \\\\"
        print(row)

        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{subtable}")

        print("\\vspace{1em}")

        # Type-2 table
        print("\\begin{subtable}{\\columnwidth}")
        print("\\centering")
        print("\\caption{Type-2 (interpolation)}")

        print("\\begin{tabular}{l" + "r" * len(grid_sizes) + "}")
        print("\\toprule")
        print(header)
        print("\\midrule")

        summary_t2 = aggregate_by_category(df_t2, TIMER_CATEGORIES_T2, group_cols=['grid'])

        for category in STACK_ORDER:
            label = LABELS_T2.get(category, category)
            row = label
            for g in grid_sizes:
                cat_data = summary_t2[(summary_t2['grid'] == g) &
                                      (summary_t2['category'] == category)]
                if not cat_data.empty:
                    row += f" & {cat_data['mean'].values[0]:.1f}"
                else:
                    row += " & ---"
            row += " \\\\"
            print(row)

        print("\\midrule")
        row = "Total"
        for g in grid_sizes:
            g_data = summary_t2[summary_t2['grid'] == g]
            total = g_data['mean'].sum()
            row += f" & {total:.1f}"
        row += " \\\\"
        print(row)

        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{subtable}")
        print("\\end{table}")
    else:
        # Single grid case
        summary_t1 = aggregate_by_category(df_t1, TIMER_CATEGORIES_T1)
        summary_t2 = aggregate_by_category(df_t2, TIMER_CATEGORIES_T2)

        print("\\begin{table}[t]")
        print("\\centering")
        print("\\caption{NUFFT time breakdown.}")
        print("\\label{tab:nufft-breakdown}")
        print("\\small")
        print("\\begin{tabular}{lrr|lrr}")
        print("\\toprule")
        print("\\multicolumn{3}{c|}{Type-1} & \\multicolumn{3}{c}{Type-2} \\\\")
        print("Component & ms & \\% & Component & ms & \\% \\\\")
        print("\\midrule")

        total_t1 = summary_t1['mean'].sum()
        total_t2 = summary_t2['mean'].sum()

        rows_t1 = []
        for cat in STACK_ORDER:
            cat_data = summary_t1[summary_t1['category'] == cat]
            if not cat_data.empty:
                t = cat_data['mean'].values[0]
                pct = 100 * t / total_t1
                label = LABELS_T1.get(cat, cat)
                rows_t1.append((label, t, pct))

        rows_t2 = []
        for cat in STACK_ORDER:
            cat_data = summary_t2[summary_t2['category'] == cat]
            if not cat_data.empty:
                t = cat_data['mean'].values[0]
                pct = 100 * t / total_t2
                label = LABELS_T2.get(cat, cat)
                rows_t2.append((label, t, pct))

        max_rows = max(len(rows_t1), len(rows_t2))
        for i in range(max_rows):
            if i < len(rows_t1):
                c1, t1, p1 = rows_t1[i]
                left = f"{c1} & {t1:.1f} & {p1:.0f}"
            else:
                left = "& &"

            if i < len(rows_t2):
                c2, t2, p2 = rows_t2[i]
                right = f"{c2} & {t2:.1f} & {p2:.0f}"
            else:
                right = "& &"

            print(f"{left} & {right} \\\\")

        print("\\midrule")
        print(f"Total & {total_t1:.1f} & 100 & Total & {total_t2:.1f} & 100 \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")


def main():
    parser = argparse.ArgumentParser(description='Plot NUFFT time breakdown')
    parser.add_argument('--input', '-i', required=True,
                        help='Input CSV file (single or combined)')
    parser.add_argument('--output', '-o', default='nufft_breakdown',
                        help='Output file prefix')
    parser.add_argument('--single', action='store_true',
                        help='Input is single-run CSV (not combined)')
    parser.add_argument('--latex', action='store_true',
                        help='Print LaTeX table')
    args = parser.parse_args()

    print(f"Loading {args.input}...")

    if args.single:
        df = load_single_csv(args.input)
    else:
        # Try to detect format
        with open(args.input, 'r') as f:
            header = f.readline().strip()

        if 'grid' in header:
            df = load_combined_csv(args.input)
        else:
            df = load_single_csv(args.input)

    print(f"Loaded {len(df)} timing records")

    print_summary_table(df)

    if args.latex:
        print_latex_table(df)

    # Choose plot type based on data
    has_grid = 'grid' in df.columns

    if has_grid:
        plot_breakdown_vs_gridsize(df, args.output)
    else:
        plot_single_breakdown(df, args.output)


if __name__ == '__main__':
    main()