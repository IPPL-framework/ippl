#!/usr/bin/env python3
"""Render FEL HDF5 visualization output as ffmpeg-like debug frames."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


STEP_RE = re.compile(r"^step_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render FEL HDF5 output to PNG frames. The view matches the old "
            "ffmpeg visualization: lab-frame Poynting magnitude on the x-z "
            "mid-y plane with particles overlaid in green."
        )
    )
    parser.add_argument("input", type=Path, help="FEL HDF5 file, e.g. fel_output_4.h5")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("fel_hdf5_frames"),
        help="Directory for rendered PNG frames",
    )
    parser.add_argument(
        "--movie",
        type=Path,
        default=None,
        help="Optional MP4/GIF output assembled from rendered frames",
    )
    parser.add_argument("--fps", type=int, default=30, help="Movie frame rate")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0e-5,
        help="Multiplier applied before clipping into the turbo colormap",
    )
    parser.add_argument(
        "--max-particles",
        type=int,
        default=20000,
        help="Maximum particles drawn per frame; use 0 to draw none",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="PNG resolution control",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Render only the first N available HDF5 steps",
    )
    return parser.parse_args()


def step_number(name: str) -> int:
    match = STEP_RE.match(name)
    if not match:
        raise ValueError(f"Not a step group: {name}")
    return int(match.group(1))


def collect_steps(h5: h5py.File, limit: int | None) -> list[str]:
    steps = sorted((name for name in h5 if STEP_RE.match(name)), key=step_number)
    if limit is not None:
        steps = steps[:limit]
    return steps


def particle_sample(particles: np.ndarray, max_particles: int) -> np.ndarray:
    if max_particles <= 0 or particles.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if len(particles) <= max_particles:
        return particles
    idx = np.linspace(0, len(particles) - 1, max_particles, dtype=np.int64)
    return particles[idx]


def render_step(
    group: h5py.Group,
    step_name: str,
    output_path: Path,
    scale: float,
    max_particles: int,
    dpi: int,
) -> None:
    poynting = np.asarray(group["poynting_magnitude"])
    particles = np.asarray(group["particle_xz_m"])
    particles = particle_sample(particles, max_particles)

    extents = np.asarray(group.attrs["extents_m"], dtype=np.float64)
    time = float(group.attrs["time"])
    iteration = int(group.attrs["iteration"])

    normalized = np.clip(poynting * scale, 0.0, 1.0)
    extent = [0.0, extents[2], -0.5 * extents[0], 0.5 * extents[0]]

    aspect = max(extents[2] / max(extents[0], np.finfo(float).eps), 1.0)
    fig_width = min(max(6.0 * aspect, 8.0), 18.0)
    fig_height = 6.0
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    ax.imshow(
        normalized,
        origin="lower",
        extent=extent,
        cmap="turbo",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )

    if len(particles) > 0:
        ax.scatter(
            particles[:, 1],
            particles[:, 0],
            s=0.15,
            c="#00ff66",
            alpha=0.7,
            linewidths=0,
            rasterized=True,
        )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("z [m]")
    ax.set_ylabel("x [m]")
    ax.set_title(f"{step_name}: iteration {iteration}, t={time:.6g}")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_movie(frame_paths: list[Path], movie_path: Path, fps: int) -> None:
    movie_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(movie_path, fps=fps) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[Path] = []
    with h5py.File(args.input, "r") as h5:
        steps = collect_steps(h5, args.limit)
        if not steps:
            raise RuntimeError(f"No step_* groups found in {args.input}")
        for idx, step_name in enumerate(steps):
            frame_path = args.output_dir / f"frame_{idx:05d}_{step_name}.png"
            render_step(
                h5[step_name],
                step_name,
                frame_path,
                args.scale,
                args.max_particles,
                args.dpi,
            )
            rendered.append(frame_path)
            print(frame_path)

    if args.movie is not None:
        write_movie(rendered, args.movie, args.fps)
        print(args.movie)


if __name__ == "__main__":
    main()
