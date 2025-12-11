#!/usr/bin/env python3
"""
Plot heatmaps of kernel throughput vs tile size and kernel width.

Expected CSV format:
    tile_size,width_2,width_3,width_4,...
    1,487.39,336.31,176.19,...
    2,2509.49,1737.19,863.12,...

Usage:
    python plot_tile_heatmap.py -i tile_sweep_heatmap -o fig_tile_sweep

Expects files: tile_sweep_heatmap_Tiled.csv, tile_sweep_heatmap_OutputFocused.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

        # Figure
        'figure.facecolor': 'white',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })


def load_heatmap_data(filepath):
    """Load heatmap CSV and convert to matrix format."""
    df = pd.read_csv(filepath)

    # Extract tile sizes
    tile_sizes = df['tile_size'].values

    # Extract widths from column names (width_2, width_3, etc.)
    width_cols = [c for c in df.columns if c.startswith('width_')]
    widths = [int(c.split('_')[1]) for c in width_cols]

    # Build matrix (tile_size x width)
    data = df[width_cols].values

    return tile_sizes, widths, data


def plot_single_heatmap(ax, tile_sizes, widths, data, title, cmap='viridis',
                        vmin=None, vmax=None, show_cbar=True):
    """Plot a single heatmap on given axes."""

    # Mask NaN values
    masked_data = np.ma.masked_invalid(data)

    # Create heatmap
    im = ax.imshow(masked_data, aspect='auto', cmap=cmap,
                   vmin=vmin, vmax=vmax, origin='lower')

    # Set ticks
    ax.set_xticks(np.arange(len(widths)))
    ax.set_xticklabels(widths)
    ax.set_yticks(np.arange(len(tile_sizes)))
    ax.set_yticklabels(tile_sizes)

    ax.set_xlabel('Kernel width $w$')
    ax.set_ylabel('Tile size')
    ax.set_title(title, fontsize=9, pad=6)

    # Add value annotations
    for i in range(len(tile_sizes)):
        for j in range(len(widths)):
            val = data[i, j]
            if np.isnan(val):
                continue
            # Choose text color based on background
            text_color = 'white' if val < (vmax - vmin) * 0.5 + vmin else 'black'
            if val >= 1000:
                text = f'{val/1000:.1f}k'
            else:
                text = f'{val:.0f}'
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=6, color=text_color)

    return im


def plot_dual_heatmap(tiled_file, output_focused_file, output_prefix):
    """Create side-by-side heatmaps for Tiled and OutputFocused methods."""
    setup_style()

    # Load data
    tile_sizes_t, widths_t, data_t = load_heatmap_data(tiled_file)
    tile_sizes_o, widths_o, data_o = load_heatmap_data(output_focused_file)

    # Use common scale
    all_data = np.concatenate([data_t.flatten(), data_o.flatten()])
    vmin = 0
    vmax = np.nanmax(all_data)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))

    # Plot Tiled
    im1 = plot_single_heatmap(axes[0], tile_sizes_t, widths_t, data_t,
                              'Tiled', vmin=vmin, vmax=vmax, show_cbar=False)

    # Plot OutputFocused (Grid-Parallel)
    im2 = plot_single_heatmap(axes[1], tile_sizes_o, widths_o, data_o,
                              'Grid-Parallel', vmin=vmin, vmax=vmax, show_cbar=False)

    # Add shared colorbar
    fig.subplots_adjust(right=0.88, wspace=0.25)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('Throughput (Mpts/s)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.savefig(f'{output_prefix}.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}.png', dpi=300, facecolor='white')
    plt.show()

    print(f"Saved: {output_prefix}.pdf, {output_prefix}.png")


def plot_single_file_heatmap(filepath, output_prefix, title=None):
    """Plot heatmap from a single file."""
    setup_style()

    tile_sizes, widths, data = load_heatmap_data(filepath)

    if title is None:
        title = Path(filepath).stem

    fig, ax = plt.subplots(figsize=(4, 3))

    vmin = 0
    vmax = np.nanmax(data)

    im = plot_single_heatmap(ax, tile_sizes, widths, data, title,
                             vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label('Throughput (Mpts/s)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()

    plt.savefig(f'{output_prefix}.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}.png', dpi=300, facecolor='white')
    plt.show()

    print(f"Saved: {output_prefix}.pdf, {output_prefix}.png")


def print_optimal_tiles(tiled_file, output_focused_file):
    """Print optimal tile size for each width."""
    print("\n" + "=" * 60)
    print("OPTIMAL TILE SIZES")
    print("=" * 60)

    for name, filepath in [('Tiled', tiled_file), ('Grid-Parallel', output_focused_file)]:
        tile_sizes, widths, data = load_heatmap_data(filepath)

        print(f"\n{name}:")
        print(f"{'Width':>8s} {'Opt Tile':>10s} {'Throughput':>12s}")
        print("-" * 32)

        for j, w in enumerate(widths):
            col = data[:, j]
            if np.all(np.isnan(col)):
                continue
            best_idx = np.nanargmax(col)
            best_tile = tile_sizes[best_idx]
            best_tput = col[best_idx]
            print(f"{w:>8d} {best_tile:>10d} {best_tput:>12.0f}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot tile size sweep heatmaps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot dual heatmap (expects _Tiled.csv and _OutputFocused.csv)
  python plot_tile_heatmap.py -i tile_sweep_heatmap -o fig_tile_sweep
  
  # Plot single file
  python plot_tile_heatmap.py --single tile_sweep_heatmap_Tiled.csv -o fig_tiled
        """
    )
    parser.add_argument('--input', '-i', default='tile_sweep_heatmap',
                        help='Input file prefix (without _Tiled.csv/_OutputFocused.csv)')
    parser.add_argument('--output', '-o', default='tile_heatmap',
                        help='Output file prefix')
    parser.add_argument('--single', type=str, default=None,
                        help='Plot single file instead of dual')
    args = parser.parse_args()

    if args.single:
        plot_single_file_heatmap(args.single, args.output)
    else:
        tiled_file = f"{args.input}_Tiled.csv"
        output_focused_file = f"{args.input}_OutputFocused.csv"

        # Check files exist
        if not Path(tiled_file).exists():
            print(f"Error: {tiled_file} not found")
            return
        if not Path(output_focused_file).exists():
            print(f"Error: {output_focused_file} not found")
            return

        print_optimal_tiles(tiled_file, output_focused_file)
        plot_dual_heatmap(tiled_file, output_focused_file, args.output)


if __name__ == '__main__':
    main()