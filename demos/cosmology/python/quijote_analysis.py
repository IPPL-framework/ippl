#!/usr/bin/env python3
## @file quijote_analysis.py
# @brief Bounded-particle-memory real-space spectra for canonical Quijote comparisons.
# @ingroup cosmology_python
# @details For nonzero modes delta_n=sum_p exp(-2*pi*i*n.x/L)/Np.
# Interlaced CIC combines grids at 0 and half a cell after correcting their
# physical-origin phase; coefficients are divided by prod_d sinc(n_d/N)^2.
# Shell auto power is L^3 <|delta_n|^2>, cross power L^3 Re<delta_A delta_B*>.
# R2C multiplicities count the full lattice. Correlation always uses raw auto
# power; optional Poisson subtraction changes reported autos, never cross power.
# Interlacing/window correction do not eliminate all aliasing. The analysis
# mesh is independent of the force mesh; no convergence claim is inferred.
"""Chunked particle deposition; serial mesh with threaded SciPy FFTs.

Particle buffers are bounded by chunk_size. Mesh/FFT arrays remain in RAM and
are explicitly budgeted; this is not a distributed FFT implementation.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import fft


## @brief Hash a file with an 8 MiB stream buffer.
# @param path Input or output filesystem path; the calling contract determines freshness and format.
# @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
def sha256(path):
    """Hash a file with an 8 MiB stream buffer."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


## @brief Resolve canonical files, a converter JSON, or one snapshots.csv epoch.
# @param specifications Canonical paths/globs, a converter JSON manifest, or snapshots.csv.
# @param epoch Optional exact snapshot name selecting one row of snapshots.csv.
# @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
def resolve_sources(specifications, epoch=None):
    """Resolve canonical files, a converter JSON, or one snapshots.csv epoch.

    Returns paths and a declared scale factor (if a snapshot manifest was used).
    Manifest readers resolve relative paths against the manifest directory.
    """
    paths = [Path(item) for item in specifications]
    expected_a = None
    if len(paths) == 1 and paths[0].suffix == ".json":
        manifest_path = paths[0].resolve(strict=True)
        manifest = json.loads(manifest_path.read_text())
        output = manifest["output"]
        path = Path(output["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        if path.stat().st_size != output["bytes"] or sha256(path) != output["sha256"]:
            raise ValueError("Converted input does not match its provenance manifest")
        paths = [path]
    elif len(paths) == 1 and paths[0].name == "snapshots.csv":
        manifest_path = paths[0].resolve(strict=True)
        with manifest_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if epoch is None and len(rows) != 1:
            raise ValueError("Specify an epoch when snapshots.csv contains multiple outputs")
        selected = [row for row in rows if epoch is None or row["name"] == epoch]
        if len(selected) != 1:
            raise ValueError("Snapshot epoch is missing or duplicated")
        row = selected[0]
        if row["format"] != "binary" or int(row["ranks"]) < 1:
            raise ValueError("Only binary snapshot manifests are supported")
        name = row["name"]
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("Invalid snapshot epoch name")
        expected_a = float(row["a"])
        paths = [manifest_path.parent / f"particles_{name}_rank{rank}.bin"
                 for rank in range(int(row["ranks"]))]
    elif epoch is not None:
        raise ValueError("An epoch selector requires snapshots.csv")
    expanded = []
    for path in paths:
        matches = sorted(glob.glob(str(path)))
        if not matches:
            raise FileNotFoundError(path)
        expanded.extend(Path(match).resolve(strict=True) for match in matches)
    if len(set(expanded)) != len(expanded):
        raise ValueError("Duplicate particle input file")
    return expanded, expected_a


## @brief Canonical binary state with bounded particle chunks and exact ID auditing.
class ParticleSource:
    """Validated canonical binary source with repeatable mmap chunk iteration."""

    ## @brief Initialize the documented source geometry and chunked particle access.
    # @param paths Distinct canonical IPPLPS01 files covering one complete particle state.
    # @param chunk_size Positive maximum particle records materialized per chunk.
    # @param expected_a Optional exact comparison scale factor, checked against the binary header.
    def __init__(self, paths, chunk_size=262144, expected_a=None):
        import quijote_io
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        ## @brief Distinct resolved canonical shard paths for this state.
        self.paths = [Path(path).resolve(strict=True) for path in paths]
        if not self.paths or len(set(self.paths)) != len(self.paths):
            raise ValueError("Require distinct particle files")
        ## @brief Maximum records in each particle buffer.
        self.chunk_size = chunk_size
        ## @brief Validated per-shard cosmology/count headers.
        self.headers = [quijote_io.read_header(path) for path in self.paths]
        ## @brief Common cosmology and global-count header.
        self.header = self.headers[0]
        ## @brief Global number of equal-mass particles.
        self.count = int(self.header["total_count"])
        ## @brief Periodic comoving box side in Mpc/h.
        self.box = float(self.header["box_mpc_h"])
        for header in self.headers:
            for key in ("total_count", "a", "box_mpc_h", "omega_m", "omega_lambda", "hubble", "particle_mass_msun_h"):
                if header[key] != self.header[key]:
                    raise ValueError(f"Inconsistent shard header: {key}")
        if sum(header["file_count"] for header in self.headers) != self.count:
            raise ValueError("Shard counts do not cover the declared particle count")
        if expected_a is not None and not math.isclose(self.header["a"], expected_a, rel_tol=1e-12, abs_tol=0):
            raise ValueError("Snapshot header disagrees with manifest epoch")
        ## @brief Exact file hashes, sizes and headers recorded after input validation.
        self.provenance = None

    ## @brief Yield at most chunk_size records with no full-particle materialization.
    # @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
    def records(self):
        """Yield at most chunk_size records with no full-particle materialization."""
        import quijote_io
        for path, header in zip(self.paths, self.headers):
            if not header["file_count"]:
                continue
            data = np.memmap(path, dtype=quijote_io.RecordDtype, mode="r",
                             offset=quijote_io.HeaderSize, shape=(header["file_count"],))
            for first in range(0, len(data), self.chunk_size):
                records = data[first:first + self.chunk_size]
                if (not np.isfinite(records["position"]).all() or not np.isfinite(records["momentum"]).all()
                        or np.any(records["position"] < 0) or np.any(records["position"] >= self.box)
                        or np.any(records["id"] >= self.count)):
                    raise ValueError("Invalid canonical phase space or particle ID")
                if header["flags"] & 1 and not np.array_equal(records["id"], np.arange(first, first + len(records))):
                    raise ValueError("Canonical ordered IDs do not match their record index")
                yield records
            del data

    ## @brief Real positions or s_los=x_los+p_los/(a^2 E), periodically wrapped.
    # @param line_of_sight None for real space; 0,1,2 for x,y,z redshift-space analysis.
    # @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
    def positions(self, line_of_sight=None):
        """Real positions or s_los=x_los+p_los/(a^2 E), periodically wrapped."""
        if line_of_sight not in (None, 0, 1, 2):
            raise ValueError("Line of sight must be 0, 1, 2 or None")
        for records in self.records():
            positions = records["position"]
            if line_of_sight is not None:
                a = self.header["a"]
                expansion = math.sqrt(self.header["omega_m"] / a**3 + self.header["omega_lambda"])
                positions = positions.copy()
                positions[:, line_of_sight] += records["momentum"][:, line_of_sight] / (a*a*expansion)
                positions[:, line_of_sight] %= self.box
            yield positions

    ## @brief Check IDs across all shards exactly; memory N bytes plus chunk buffers.
    # @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
    def validate(self):
        """Check IDs across all shards exactly; memory N bytes plus chunk buffers.

        Hash before and after validation to reject concurrent input modification.
        The returned provenance describes the measured inputs, including headers.
        """
        before = [sha256(path) for path in self.paths]
        seen = np.zeros(self.count, dtype=np.bool_)
        count = 0
        for records in self.records():
            ids = records["id"]
            if np.any(ids >= self.count) or len(np.unique(ids)) != len(ids) or np.any(seen[ids]):
                raise ValueError("Particle IDs are out of range or duplicated across shards")
            seen[ids] = True
            count += len(ids)
        if count != self.count or not seen.all():
            raise ValueError("Incomplete particle ID coverage")
        del seen
        after = [sha256(path) for path in self.paths]
        if before != after:
            raise ValueError("Particle files changed during validation")
        ## @brief Exact file hashes, sizes and headers recorded after input validation.
        self.provenance = [{"path": str(path), "bytes": path.stat().st_size,
                            "sha256": digest, "header": header}
                           for path, digest, header in zip(self.paths, after, self.headers)]
        return self.provenance


## @brief Conservative mesh/FFT/chunk/ID-audit allowance, excluding filesystem cache.
# @param grid Even independent analysis-grid side; unrelated to the simulation force mesh.
# @param chunk_size Positive maximum particle records materialized per chunk.
# @param particles Global particle count used to budget the exact ID-coverage audit.
# @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
def analysis_memory_bytes(grid, chunk_size, particles=0):
    """Conservative mesh/FFT/chunk/ID-audit allowance, excluding filesystem cache.

    Six complex half-grids plus four real grids allow both fields, interlacing,
    FFT workspace and output copies. This is a preflight estimate, not measured
    peak RSS; library allocator behavior may vary across supported platforms.
    """
    if grid < 4 or grid % 2 or chunk_size < 1 or particles < 0:
        raise ValueError("Require even grid >=4 and positive chunk_size")
    return int(6 * 16 * grid**2 * (grid // 2 + 1) + 4 * 8 * grid**3
               + 256 * chunk_size + 2 * particles + 128 * 1024**2)


## @brief Evaluate  geometry.
# @param box Positive periodic comoving box side in Mpc/h.
# @param grid Even independent analysis-grid side; unrelated to the simulation force mesh.
# @param count Expected total equal-mass particle count.
# @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
def _geometry(box, grid, count):
    if not math.isfinite(box) or box <= 0 or grid < 4 or grid % 2 or count < 1:
        raise ValueError("Require positive box/count and even analysis grid >=4")


## @brief Validate physical shell geometry before any expensive FFT allocation.
# @param edges Strictly increasing shell edges in h/Mpc.
# @param grid Even independent analysis-grid side.
# @param box Positive periodic comoving box side in Mpc/h.
# @return Finite float64 shell-edge array within the analysis Nyquist limit.
def _validated_edges(edges, grid, box):
    edges = np.asarray(edges, dtype=np.float64)
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or edges[0] < 0 or not np.all(np.diff(edges) > 0)
            or edges[-1] > np.pi * grid / box * (1 + 1e-13)):
        raise ValueError("Require increasing physical k edges within analysis Nyquist")
    return edges


## @brief Deposit equal masses on periodic nodes (j+shift)*L/grid.
# @param position_chunks Iterator of finite (Nchunk,3) position arrays in Mpc/h.
# @param box Positive periodic comoving box side in Mpc/h.
# @param grid Even independent analysis-grid side; unrelated to the simulation force mesh.
# @param count Expected total equal-mass particle count.
# @param shift Analysis-grid origin offset in units of one mesh cell.
# @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
def deposit_cic(position_chunks, box, grid, count, shift=0.0):
    """Deposit equal masses on periodic nodes (j+shift)*L/grid.

    np.add.at uses only chunk-sized indices/weights; unlike bincount(minlength),
    no additional full mesh is allocated on every particle chunk.
    """
    _geometry(box, grid, count)
    density = np.zeros((grid, grid, grid), dtype=np.float64)
    flat = density.ravel()
    actual = 0
    for positions in position_chunks:
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
            raise ValueError("Particle positions must be finite Nx3 chunks")
        actual += len(positions)
        coordinate = (np.remainder(positions, box) * (grid / box) - shift) % grid
        lower = np.floor(coordinate).astype(np.int64)
        fraction = coordinate - lower
        for dx in (0, 1):
            ix = (lower[:, 0] + dx) % grid
            wx = fraction[:, 0] if dx else 1 - fraction[:, 0]
            for dy in (0, 1):
                iy = (lower[:, 1] + dy) % grid
                wy = fraction[:, 1] if dy else 1 - fraction[:, 1]
                for dz in (0, 1):
                    iz = (lower[:, 2] + dz) % grid
                    wz = fraction[:, 2] if dz else 1 - fraction[:, 2]
                    np.add.at(flat, (ix * grid + iy) * grid + iz, wx * wy * wz)
    if actual != count:
        raise ValueError("Particle chunk count disagrees with source header")
    mass_error = abs(float(density.sum(dtype=np.float64)) / count - 1)
    if mass_error > 1e-11:
        raise ValueError("Analysis CIC mass conservation failed")
    density *= grid**3 / count
    density -= 1
    return density, mass_error


## @brief Return CIC-window-corrected interlaced R2C coefficients and mass audit.
# @param source Particle source with repeatable bounded-chunk position iteration.
# @param grid Even independent analysis-grid side; unrelated to the simulation force mesh.
# @param workers Positive number of SciPy FFT worker threads.
# @param line_of_sight None for real space; 0,1,2 for x,y,z redshift-space analysis.
# @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
def fourier_field(source, grid, workers=1, line_of_sight=None):
    """Return CIC-window-corrected interlaced R2C coefficients and mass audit."""
    if workers < 1:
        raise ValueError("FFT workers must be positive")
    fields = None
    mass_errors = []
    axis = np.fft.fftfreq(grid) * grid
    last = np.arange(grid // 2 + 1)
    for shift in (0., .5):
        positions = source.positions() if line_of_sight is None else source.positions(line_of_sight)
        density, error = deposit_cic(positions, source.box, grid, source.count, shift)
        mass_errors.append(error)
        transformed = fft.rfftn(density, workers=workers, overwrite_x=True)
        del density
        transformed /= grid**3
        if fields is None:
            fields = transformed
        else:
            for index, nx in enumerate(axis):
                phase = np.exp(-2j * np.pi * shift * (nx + axis[:, None] + last[None, :]) / grid)
                fields[index] += transformed[index] * phase
            del transformed
            fields *= .5
    for index, nx in enumerate(axis):
        window = np.sinc(nx / grid)**2 * np.sinc(axis[:, None] / grid)**2 * np.sinc(last[None, :] / grid)**2
        fields[index] /= window
    fields[0, 0, 0] = 0
    return fields, mass_errors


## @brief Bounded-chunk exact particle sum, intended only for a few oracle modes.
# @param position_chunks Iterator of finite (Nchunk,3) position arrays in Mpc/h.
# @param box Positive periodic comoving box side in Mpc/h.
# @param modes Integer (Nmodes,3) Fourier wavevectors for exact particle sums.
# @param count Expected total equal-mass particle count.
# @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
def direct_coefficients(position_chunks, box, modes, count):
    """Bounded-chunk exact particle sum, intended only for a few oracle modes."""
    modes = np.asarray(modes)
    if modes.ndim != 2 or modes.shape[1] != 3 or not np.equal(modes, np.rint(modes)).all():
        raise ValueError("Require integer Mx3 modes")
    result = np.zeros(len(modes), dtype=np.complex128)
    actual = 0
    for positions in position_chunks:
        actual += len(positions)
        for index, mode in enumerate(modes):
            result[index] += np.exp(-2j * np.pi * (positions @ mode) / box).sum()
    if actual != count:
        raise ValueError("Particle count mismatch in direct mode oracle")
    return result / count


## @brief Bin two phase-aligned R2C fields without allocating full-grid mode arrays.
# @param left First field/source; spectra ratios use the right field as reference.
# @param right Second field/source; reference denominator for reported power ratios.
# @param box Positive periodic comoving box side in Mpc/h.
# @param edges Strictly increasing physical shell edges in h/Mpc, within analysis Nyquist.
# @param left_count Equal-mass particle count of the first field.
# @param right_count Equal-mass particle count of the reference field.
# @param shot_noise raw retains both auto powers; poisson subtracts L cubed divided by particle count.
# @param line_of_sight None for real space; 0,1,2 for x,y,z redshift-space analysis.
# @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
def shell_spectra(left, right, box, edges, left_count, right_count, shot_noise="raw", line_of_sight=None):
    """Bin two phase-aligned R2C fields without allocating full-grid mode arrays."""
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("Fourier fields must share the same R2C geometry")
    grid = left.shape[0]
    _geometry(box, grid, left_count)
    if left.shape != (grid, grid, grid // 2 + 1) or right_count < 1:
        raise ValueError("Invalid R2C geometry or particle count")
    edges = _validated_edges(edges, grid, box)
    if shot_noise not in ("raw", "poisson"):
        raise ValueError("shot_noise must be raw or poisson")
    if line_of_sight not in (None, 0, 1, 2):
        raise ValueError("Invalid line of sight")
    bins = len(edges) - 1
    sums = np.zeros((9, bins), dtype=np.float64)
    axis = np.fft.fftfreq(grid) * grid
    last = np.arange(grid // 2 + 1)
    weight = np.full((grid, grid // 2 + 1), 2., dtype=np.float64)
    weight[:, 0] = weight[:, -1] = 1.
    for index, nx in enumerate(axis):
        radius = np.sqrt(nx * nx + axis[:, None]**2 + last[None, :]**2) * (2 * np.pi / box)
        which = np.searchsorted(edges, radius, side="right") - 1
        selected = (radius > 0) & (which >= 0) & (which < bins)
        ids = which[selected]
        w = weight[selected]
        a, b = left[index][selected], right[index][selected]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Nonfinite Fourier coefficients")
        powers = (w * abs(a)**2, w * abs(b)**2)
        values = [w, w * radius[selected], *powers, w * (a * np.conj(b)).real]
        if line_of_sight is not None:
            component = (np.full(radius.shape, nx) if line_of_sight == 0 else
                         np.broadcast_to(axis[:, None], radius.shape) if line_of_sight == 1 else
                         np.broadcast_to(last[None, :], radius.shape))
            mu = component[selected] * (2 * np.pi / box) / radius[selected]
            l2, l4 = .5 * (3*mu**2 - 1), (35*mu**4 - 30*mu**2 + 3) / 8
            values += [5 * powers[0] * l2, 5 * powers[1] * l2,
                       9 * powers[0] * l4, 9 * powers[1] * l4]
        for target, values in zip(sums, values):
            target += np.bincount(ids, weights=values, minlength=bins)
    rows = []
    for index in range(bins):
        modes = int(round(sums[0, index]))
        if not modes:
            continue
        pa, pb, cross = box**3 * sums[2:5, index] / modes
        noise_a, noise_b = box**3 / left_count, box**3 / right_count
        measured_a = pa - noise_a if shot_noise == "poisson" else pa
        measured_b = pb - noise_b if shot_noise == "poisson" else pb
        rows.append({"k_lower": float(edges[index]), "k_upper": float(edges[index + 1]),
                     "k_mean": float(sums[1, index] / modes), "modes": modes,
                     "left_raw": float(pa), "right_raw": float(pb), "cross_raw": float(cross),
                     "left_poisson_noise": noise_a, "right_poisson_noise": noise_b,
                     "left_power": float(measured_a), "right_power": float(measured_b),
                     "ratio": float(measured_a / measured_b) if measured_b > 0 and measured_a > 0 else None,
                     "correlation_raw": float(cross / math.sqrt(pa * pb)) if pa > 0 and pb > 0 else None})
        if line_of_sight is not None:
            p2a, p2b, p4a, p4b = box**3 * sums[5:, index] / modes
            rows[-1].update(left_p2_raw=float(p2a), right_p2_raw=float(p2b),
                            left_p4_raw=float(p4a), right_p4_raw=float(p4b))
    return rows


## @brief Validate, measure both fields identically, and return an auditable report.
# @param left First field/source; spectra ratios use the right field as reference.
# @param right Second field/source; reference denominator for reported power ratios.
# @param grid Even independent analysis-grid side; unrelated to the simulation force mesh.
# @param edges Strictly increasing physical shell edges in h/Mpc, within analysis Nyquist.
# @param workers Positive number of SciPy FFT worker threads.
# @param memory_limit_bytes Optional declared memory cap in bytes checked before field allocation.
# @param shot_noise raw retains both auto powers; poisson subtracts L cubed divided by particle count.
# @param line_of_sight None for real space; 0,1,2 for x,y,z redshift-space analysis.
# @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
def compare(left, right, *, grid, edges, workers=1, memory_limit_bytes=None, shot_noise="raw", line_of_sight=None):
    """Validate, measure both fields identically, and return an auditable report."""
    for key in ("total_count", "a", "box_mpc_h", "omega_m", "omega_lambda", "hubble", "particle_mass_msun_h"):
        if not math.isclose(left.header[key], right.header[key], rel_tol=1e-10, abs_tol=0):
            raise ValueError(f"Comparison inputs differ in {key}")
    estimate = analysis_memory_bytes(grid, max(left.chunk_size, right.chunk_size), max(left.count, right.count))
    if memory_limit_bytes is not None and estimate > memory_limit_bytes:
        raise MemoryError(f"Analysis estimate {estimate / 1024**3:.2f} GiB exceeds memory budget")
    edges = _validated_edges(edges, grid, left.box)
    if workers < 1 or shot_noise not in ("raw", "poisson") or line_of_sight not in (None, 0, 1, 2):
        raise ValueError("Invalid FFT workers, shot-noise convention or line of sight")
    for source in (left, right):
        source.validate()
    a, amass = fourier_field(left, grid, workers, line_of_sight)
    b, bmass = fourier_field(right, grid, workers, line_of_sight)
    rows = shell_spectra(a, b, left.box, edges, left.count, right.count, shot_noise, line_of_sight)
    del a, b
    for source in (left, right):
        if any(sha256(item["path"]) != item["sha256"] for item in source.provenance):
            raise ValueError("Particle files changed while computing spectra")
    return {"schema": "ippl-quijote-spectra-v1", "left": left.provenance, "right": right.provenance,
            "a": left.header["a"], "box_mpc_h": left.box, "analysis_grid": grid,
            "fft_workers": workers, "estimated_memory_bytes": estimate,
            "k_edges_h_per_mpc": list(map(float, edges)), "mass_relative_errors": [amass, bmass],
            "estimator": "interlaced CIC; physical-origin phase; CIC window corrected; R2C multiplicity",
            "power_units": "(Mpc/h)^3", "shot_noise": shot_noise,
            "line_of_sight": line_of_sight,
            "multipoles": "(2ell+1)*L^3 mean(|delta|^2 Legendre_ell(mu)); raw ell=2,4 when LOS selected",
            "cross_convention": "raw; no cross shot-noise subtraction; correlation uses raw autos",
            "qualification": "measurement only; force/time/analysis convergence is not assumed; residual aliasing remains",
            "rows": rows}
