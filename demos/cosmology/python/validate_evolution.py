#!/usr/bin/env python3
## @file validate_evolution.py
# @brief Matched imported-particle evolution: IPPL versus pinned native plain-PM FastPM.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
# @see cosmology_model cosmology_numerics cosmology_spectra
"""Matched imported-particle evolution: IPPL versus pinned native plain-PM FastPM.

This is a small, deterministic local comparison, not a halo/CDM-statistics or
exascale qualification. Native Nyquist/float32 differences are retained. The
planar analytic map is used to construct initial data, never as truth after
shell crossing. All engineering budgets below precede comparison execution.
"""
from __future__ import annotations

from source_paths import source_path

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
from functools import lru_cache
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile

import numpy as np
import pandas as pd

from validate_linear import growth_reference
from runtime_metadata import validate_runtime_metadata


## @var Parameters
# @brief Named Parameters protocol/schema value; the source initializer records its exact contents.
Parameters = {"particle_grid": 32, "box_size": 168.75, "omega_m": .31,
              "a_initial": .02, "a_final": .2, "checkpoints": 8}
## @var Limits
# @brief Named Limits protocol/schema value; the source initializer records its exact contents.
Limits = {
    "initial_position_eps_box": 128., "initial_momentum": "exact shared float32 values",
    "native_time_factor_relative": 1e-8, "schedule_relative": 2e-13,
    "rank_ippl_position_cells": 1e-10, "rank_ippl_momentum_relative": 1e-10,
    "rank_fastpm_position_cells": 5e-5, "rank_fastpm_momentum_relative": 5e-5,
    "planar_cross_position_cells": 1e-3, "planar_cross_momentum_relative": 1e-3,
    "resolved_power_fraction": {16: .05, 32: .02, 64: .01},
    "resolved_complex_relative": {16: .05, 32: .02, 64: .01},
    "resolved_correlation_minimum": {16: .995, 32: .999, 64: .9995},
    "time_postcross_ratio_minimum": 1.5, "time_precross_ratio_range": [2., 6.],
    "time_finest_position_cells": .002, "time_finest_momentum_relative": .002,
    "time_precision_position_cells": 1e-6, "time_precision_momentum_relative": 1e-5,
    "time_precision_complex_relative": 1e-5,
    "mesh_finest_pair_resolved_power_fraction": .05,
    "net_momentum_ippl_relative": 1e-10, "net_momentum_fastpm_relative": 5e-5,
    "old_mass_diagnostic_limit_recorded_not_overridden": 2e-12,
}
## @var FastPMCommit
# @brief Named FastPMCommit protocol/schema value; the source initializer records its exact contents.
FastPMCommit = "15b6c4fd7502a81d99dd13f54fcc9cfa44be1331"
## @var Columns
# @brief Named Columns protocol/schema value; the source initializer records its exact contents.
Columns = ["id", "x", "y", "z", "px", "py", "pz"]


## @brief Compute the streaming SHA256 digest of the file bytes.
# @see cosmology_tools
#
# @param path Input/output filesystem path; freshness and hash requirements are defined by this routine.
# @return Lowercase hexadecimal SHA256 of the exact retained bytes.
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


## @brief Return sqrt(mean(sum(vector_components squared))); distinguish it from componentwise RMS.
# @see cosmology_tools
#
# @param array Numerical array in the units and shape declared by this metric's caller.
# @return Vector RMS in the input array's units.
def vector_rms(array):
    return float(np.sqrt(np.mean(np.sum(np.asarray(array)**2, axis=-1))))


## @brief Return the minimum-image coordinate difference in the declared periodic box.
# @see cosmology_tools
#
# @param left First ID-matched state or Fourier array; the metric specifies denominator and normalization.
# @param right Second ID-matched state or Fourier array; the metric specifies denominator and normalization.
# @param box Positive periodic comoving box side in Mpc/h.
# @return Minimum-image coordinate array in the same comoving length unit.
def periodic_difference(left, right, box):
    delta = np.asarray(left) - np.asarray(right)
    return delta - box * np.rint(delta / box)


## @brief Independent smooth growing-mode displacement; one shared rounded p input.
# @see cosmology_tools
#
# @param kind Fixture selector from the module's declared supported cases.
# @param particle_grid Particle lattice size NP per dimension; expected particle count is NP^3.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def make_fixture(kind, particle_grid=32):
    """Independent smooth growing-mode displacement; one shared rounded p input."""
    n, box, omega = particle_grid, Parameters["box_size"], Parameters["omega_m"]
    ai, af = Parameters["a_initial"], Parameters["a_final"]
    ids = np.arange(n**3, dtype=np.uint64)
    q = (np.column_stack((ids % n, ids // n % n, ids // n**2)) + .5) * box / n
    if kind == "pancake":
        modes = [([1, 0, 0], 1.5, .17)]
    elif kind == "coupled3d":
        entries = [([1, 0, 0], .9, .17), ([0, 1, 0], .8, 1.13),
                   ([0, 0, 1], .7, 2.07), ([1, 1, 0], .6, .73),
                   ([1, 0, 1], .5, 1.71), ([0, 1, 1], .4, 2.51),
                   ([1, 1, 1], .3, .39), ([2, 1, 0], .25, 1.29),
                   ([0, 1, 2], .2, 2.91)]
        # Prescribed linear density RMS is 1 at af=.2. This is a synthetic coupled
        # benchmark, NOT a sigma8-normalized Gaussian LCDM realization.
        normalization = math.sqrt(sum(amplitude**2 for _, amplitude, _ in entries) / 2)
        modes = [(mode, amplitude / normalization, phase) for mode, amplitude, phase in entries]
    else:
        raise ValueError("Unknown fixture")
    growth_i, rate_i = growth_reference(ai, omega)
    growth_f, _ = growth_reference(af, omega)
    displacement = np.zeros_like(q)
    initial_gradient = np.tile(np.eye(3), (len(q), 1, 1))
    for mode, amplitude, phase in modes:
        wave = np.asarray(mode) * (2 * np.pi / box)
        initial_amplitude = amplitude * growth_i / growth_f
        angle = q @ wave + phase
        displacement -= initial_amplitude * np.sin(angle)[:, None] * wave / (wave @ wave)
        initial_gradient -= (initial_amplitude * np.cos(angle)[:, None, None]
                             * np.outer(wave, wave) / (wave @ wave))
    raw_p = ai**2 * math.sqrt(omega / ai**3 + 1 - omega) * rate_i * displacement
    momentum = raw_p.astype(np.float32).astype(np.float64)
    relative_rounding = vector_rms(momentum - raw_p) / vector_rms(raw_p)
    if relative_rounding > np.finfo(np.float32).eps or np.linalg.eigvalsh(initial_gradient).min() <= 0:
        raise ValueError("Invalid initial fixture or momentum quantization")
    frame = pd.DataFrame(np.column_stack(((q + displacement) % box, momentum)), columns=Columns[1:])
    frame.insert(0, "id", ids)
    frame["mass"] = 1
    description = {"fixture": kind, "particle_grid": n, "modes": modes,
                   "linear_final_density_rms": math.sqrt(sum(a*a for _, a, _ in modes) / 2),
                   "initial_minimum_map_eigenvalue": float(np.linalg.eigvalsh(initial_gradient).min()),
                   "momentum_quantization_relative_rms": relative_rounding,
                   "momentum_contract": "rounded once to float32, identical exact values supplied to both codes",
                   "scope": "synthetic deterministic growing-mode fixture; not cosmological halo statistics"}
    return frame, description


## @brief Evaluate the resolved modes helper in the documented module workflow.
# @see cosmology_tools
#
# @param kind Fixture selector from the module's declared supported cases.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def resolved_modes(kind):
    if kind == "pancake":
        return np.array([[k, 0, 0] for k in range(1, 5)], dtype=int)
    if kind != "coupled3d":
        raise ValueError("Unknown fixture")
    modes = []
    for mode in itertools.product(range(-4, 5), repeat=3):
        if not 0 < sum(v*v for v in mode) <= 16:
            continue
        if next(v for v in mode if v) > 0:
            modes.append(mode)
    return np.asarray(modes, dtype=int)


## @brief Direct particle delta_hat, unique +/- pairs; no mesh/window correction.
# @see cosmology_tools
#
# @param positions Finite particle position array of shape (Nparticles,3), in comoving Mpc/h.
# @param box Positive periodic comoving box side in Mpc/h.
# @param modes Signed integer Fourier-mode array with three Cartesian components per mode.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def density_modes(positions, box, modes):
    """Direct particle delta_hat, unique +/- pairs; no mesh/window correction."""
    positions, modes = np.asarray(positions), np.asarray(modes)
    if (positions.ndim != 2 or positions.shape[1] != 3 or not len(positions)
            or not np.isfinite(positions).all() or not np.isfinite(box) or box <= 0
            or modes.ndim != 2 or modes.shape[1] != 3 or not len(modes)
            or not np.isfinite(modes).all() or not np.equal(modes, np.rint(modes)).all()):
        raise ValueError("Density requires finite Nx3 positions, positive box and integer Mx3 modes")
    result = []
    for offset in range(0, len(modes), 16):
        phase = (2 * np.pi / box) * (np.asarray(positions) @ modes[offset:offset+16].T)
        result.extend(np.exp(-1j * phase).mean(axis=0))
    return np.asarray(result)


## @brief Compare two complex Fourier arrays with power, complex residual and correlation metrics.
# @see cosmology_tools
#
# @param left First ID-matched state or Fourier array; the metric specifies denominator and normalization.
# @param right Second ID-matched state or Fourier array; the metric specifies denominator and normalization.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def density_comparison(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if (left.ndim != 1 or left.shape != right.shape or not left.size
            or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError("Density modes must be finite nonempty matching vectors")
    power_left, power_right = float(np.vdot(left, left).real), float(np.vdot(right, right).real)
    absolute = float(np.linalg.norm(left - right))
    if min(power_left, power_right) <= 1e-28:
        return {"power_left": power_left, "power_right": power_right, "absolute_difference": absolute,
                "power_ratio": None, "complex_relative": None, "correlation": None,
                "normalization_defined": False}
    product_norms = math.sqrt(power_left * power_right)
    return {"power_left": power_left, "power_right": power_right,
            "absolute_difference": absolute, "power_ratio": power_left / power_right,
            "complex_relative": absolute / math.sqrt(product_norms),
            "correlation": float(np.vdot(right, left).real / product_norms),
            "normalization_defined": True}


## @brief Reject malformed, incomplete or nonfinite retained particle state.
# @see cosmology_tools
#
# @param frame Particle DataFrame with the exact columns/ID/finite-state contract checked by this routine.
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def validate_snapshot(frame):
    if (len(frame) == 0 or not set(Columns).issubset(frame.columns)
            or not np.array_equal(frame.id.to_numpy(), np.arange(len(frame)))
            or not np.isfinite(frame[Columns].to_numpy(dtype=float)).all()):
        raise ValueError("Snapshot requires complete sorted unique IDs and finite phase space")


## @brief Measure ID-matched periodic position and canonical-momentum differences with explicit normalizations.
# @see cosmology_tools
#
# @param left First ID-matched state or Fourier array; the metric specifies denominator and normalization.
# @param right Second ID-matched state or Fourier array; the metric specifies denominator and normalization.
# @param box Positive periodic comoving box side in Mpc/h.
# @param mesh Force-mesh size per dimension for normalizing particle displacements to cell widths.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def phase_space_metrics(left, right, box, mesh):
    validate_snapshot(left)
    validate_snapshot(right)
    if not np.isfinite(box) or box <= 0 or mesh <= 0:
        raise ValueError("Invalid phase-space comparison geometry")
    if not np.array_equal(left.id.to_numpy(), right.id.to_numpy()):
        raise ValueError("Particle IDs must match and be sorted")
    dx = periodic_difference(left[["x", "y", "z"]].to_numpy(), right[["x", "y", "z"]].to_numpy(), box)
    dp = left[["px", "py", "pz"]].to_numpy() - right[["px", "py", "pz"]].to_numpy()
    reference = vector_rms(right[["px", "py", "pz"]].to_numpy())
    return {"position_rms": vector_rms(dx), "position_cells": vector_rms(dx) / (box / mesh),
            "momentum_rms": vector_rms(dp), "momentum_relative": vector_rms(dp) / reference if reference else None,
            "reference_momentum_rms": reference}


## @brief Characterize ID-labelled planar sheet ordering; do not extend a pre-crossing analytical truth past crossing.
# @see cosmology_tools
#
# @param snapshot ID-labelled retained particle state; its epoch and units follow the originating run metadata.
# @param particle_grid Particle lattice size NP per dimension; expected particle count is NP^3.
# @param box Positive periodic comoving box side in Mpc/h.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def planar_ordering(snapshot, particle_grid, box):
    validate_snapshot(snapshot)
    n = particle_grid
    if not isinstance(n, (int, np.integer)) or n < 2 or len(snapshot) != n**3 or not np.isfinite(box) or box <= 0:
        raise ValueError("Invalid planar geometry")
    qx = (np.arange(n) + .5) * box / n
    # IDs are x-fast. These fixtures preserve plane symmetry; this is a 1D
    # neighbour-order diagnostic, NOT a general three-dimensional caustic finder.
    displacement = periodic_difference(snapshot.x.to_numpy().reshape(n, n, n), qx, box)
    jacobian = 1 + (np.roll(displacement, -1, axis=2) - displacement) / (box / n)
    return {"minimum_sampled_jacobian": float(jacobian.min()),
            "negative_interval_fraction": float(np.mean(jacobian < 0)),
            "transverse_row_displacement_spread": float(np.ptp(displacement.reshape(-1, n), axis=0).max())}


## @brief Assess successive-solution differences above the declared numerical floor and order budget.
# @see cosmology_tools
#
# @param coarse_difference Norm of the coarse-to-middle successive solution difference.
# @param fine_difference Norm of the middle-to-fine successive solution difference.
# @param floor Declared numerical reporting/order floor; not a substituted positive measured value.
# @param lower Declared lower order/range limit or first integration endpoint according to this routine.
# @param upper Declared upper order/range limit or second endpoint; optional where documented.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def refinement(coarse_difference, fine_difference, floor, lower, upper=None):
    if (not np.isfinite([floor, lower]).all() or min(floor, lower) < 0
            or (upper is not None and (not np.isfinite(upper) or upper < lower))):
        raise ValueError("Invalid refinement floor or ratio bounds")
    if not np.isfinite([coarse_difference, fine_difference]).all() or min(coarse_difference, fine_difference) < 0:
        raise ValueError("Invalid refinement difference")
    if max(coarse_difference, fine_difference) <= floor:
        return {"status": "precision_limited", "passed": True, "ratio": None, "measured_order": None}
    if fine_difference == 0:
        return {"status": "fine_difference_zero", "passed": True, "ratio": None, "measured_order": None}
    ratio = coarse_difference / fine_difference
    return {"status": "measured", "passed": bool(ratio >= lower and (upper is None or ratio <= upper)),
            "ratio": ratio, "measured_order": math.log2(ratio) if ratio > 0 else None}


## @brief Check the recorded native kick/drift factor schedule against the declared endpoints and step count.
# @see cosmology_tools
#
# @param data Finite numerical or serialized record data in the routine's explicit schema.
# @param steps Positive PM step count; endpoints are uniform in log(a) for the imported drivers.
# @param ai Positive initial dimensionless scale factor.
# @param af Positive final dimensionless scale factor, ordered after ai for forward evolution.
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def validate_factor_schedule(data, steps, ai, af):
    expected_columns = ["step", "a0", "ah", "a1", "drift", "canonical_kick0", "canonical_kick1"]
    if (list(data.columns) != expected_columns or len(data) != steps
            or not np.array_equal(data.step, np.arange(steps))
            or not np.isfinite(data.to_numpy()).all()
            or not (data[expected_columns[1:]].to_numpy() > 0).all()):
        raise ValueError("Incomplete, nonfinite or wrong native factor table")
    endpoints = ai * np.exp(np.arange(steps + 1) * np.log(af / ai) / steps)
    expected = np.column_stack((endpoints[:-1], np.sqrt(endpoints[:-1] * endpoints[1:]), endpoints[1:]))
    if np.max(abs(data[["a0", "ah", "a1"]].to_numpy() / expected - 1)) > Limits["schedule_relative"]:
        raise ValueError("Native factor intervals do not match requested schedule")


## @brief Immutable fixture/resolution/schedule descriptor used to name and select validation runs.
# @see cosmology_tools
@dataclass(frozen=True)
class Case:
    ## @var fixture
    # @brief Shared initial-condition data or its explicit case descriptor; it must remain consistent across compared solvers.
    fixture: str
    ## @var mesh
    # @brief Force-mesh size per dimension for normalizing particle displacements to cell widths.
    mesh: int
    ## @var steps
    # @brief Positive PM step count; endpoints are uniform in log(a) for the imported drivers.
    steps: int
    ## @var ranks
    # @brief Positive MPI rank count; all expected snapshot shards must exist.
    ranks: int = 1

    ## @brief Return the stable protocol case name used for artifact identity.
    # @see cosmology_tools
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
    @property
    def name(self):
        return f"{self.fixture}_m{self.mesh}_t{self.steps}_r{self.ranks}"


## @brief Evaluate the cases helper in the documented module workflow.
# @see cosmology_tools
#
# @param quick Select reduced engineering coverage; not a replacement for the full qualification matrix.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def cases(quick=False):
    result = []
    for kind in ("pancake", "coupled3d"):
        if quick:
            result.extend(Case(kind, 16, 32, rank) for rank in (1, 2))
        else:
            result.extend(Case(kind, 32, steps) for steps in (64, 128, 256))
            result.extend(Case(kind, mesh, 256) for mesh in (16, 64))
            result.extend(Case(kind, 32, 128, rank) for rank in (2, 3, 4))
    return result


## @brief Read and verify snapshot.
# @see cosmology_tools
#
# @param directory Artifact directory following this module's ownership/freshness contract.
# @param checkpoint Synchronized saved epoch index, with zero denoting imported initial state.
# @param ranks Positive MPI rank count; all expected snapshot shards must exist.
# @param particle_grid Particle lattice size NP per dimension; expected particle count is NP^3.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
@lru_cache(maxsize=8)
def read_snapshot(directory, checkpoint, ranks, particle_grid):
    directory = Path(directory)
    paths = sorted(directory.glob(f"particles_checkpoint{checkpoint:04d}_rank*.csv*"))
    expected = {f"particles_checkpoint{checkpoint:04d}_rank{rank}.csv" for rank in range(ranks)}
    if len(paths) != ranks or {path.name.removesuffix(".gz") for path in paths} != expected:
        raise ValueError(f"{directory}: wrong snapshot rank file set")
    frames = [pd.read_csv(path, dtype={"id": np.uint64}, float_precision="round_trip") for path in paths]
    if any(list(frame.columns) != Columns for frame in frames):
        raise ValueError("Unexpected snapshot columns")
    frame = pd.concat([part for part in frames if len(part)], ignore_index=True).sort_values("id").reset_index(drop=True)
    validate_snapshot(frame)
    if len(frame) != particle_grid**3:
        raise ValueError("Wrong snapshot particle count")
    return frame


## @brief Lossless compression of ONLY this completed run's generated snapshots.
# @see cosmology_tools
#
# @param directory Artifact directory following this module's ownership/freshness contract.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def archive_snapshots(directory):
    """Lossless compression of ONLY this completed run's generated snapshots.

    Retain CSV content hashes as well as compressed hashes; original bytes can
    be recovered by decompression. No user input or prior-run artifact is touched.
    """
    records = []
    for path in sorted(directory.glob("particles_checkpoint*_rank*.csv")):
        before, size = sha256(path), path.stat().st_size
        compressed = path.with_suffix(path.suffix + ".gz")
        with path.open("rb") as source, compressed.open("xb") as target:
            with gzip.GzipFile(filename="", fileobj=target, mode="wb", compresslevel=6, mtime=0) as zipped:
                shutil.copyfileobj(source, zipped)
        with gzip.open(compressed, "rb") as source:
            recovered = hashlib.sha256(source.read()).hexdigest()
        if recovered != before:
            raise RuntimeError("Snapshot compression verification failed; original retained")
        records.append({"path": str(compressed), "sha256": sha256(compressed),
                        "csv_sha256": before, "csv_bytes": size, "compressed_bytes": compressed.stat().st_size})
        path.unlink()
    return records


## @brief Launch the documented module workflow.
# @see cosmology_tools
#
# @param command Argument-vector command; no shell interpolation is required by the Python launcher.
# @param log_path Path retaining combined subprocess output and failure evidence.
# @param timeout Finite positive subprocess timeout in seconds.
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def launch(command, log_path, timeout):
    environment = dict(os.environ, OMP_NUM_THREADS="1", OMP_PROC_BIND="false",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, command)


## @brief Controller retaining commands, common inputs, fixed acceptance gates and native comparison results.
# @see cosmology_tools
class Campaign:
    ## @brief Initialize the instance with the explicit protocol and artifact ownership passed by the caller.
    # @see cosmology_tools
    #
    # @param args Parsed command-line options; see main/--help and the module workflow contract.
    def __init__(self, args):
        ## @var args
        # @brief Parsed command-line options; see main/--help and the module workflow contract.
        self.args = args
        ## @var executables
        # @brief Retained executables state owned by this instance; see the initialization and workflow contract.
        self.executables = {code: Path(getattr(args, code + "_exe")).resolve() for code in ("ippl", "fastpm")}
        ## @var root
        # @brief Campaign or artifact root following this module's ownership contract.
        self.root = args.output_dir.resolve() if args.output_dir else Path(tempfile.mkdtemp(
            prefix="matched-evolution-", dir=self.executables["ippl"].parent))
        self.root.mkdir(parents=True, exist_ok=True)
        if any(self.root.iterdir()):
            raise ValueError("Output directory must be new or empty")
        ## @var parameters
        # @brief Named protocol or cosmology parameters; unsupported keys are rejected by the calling validator.
        self.parameters = dict(Parameters, particle_grid=16 if args.quick else 32)
        source = Path(__file__).resolve().parent
        sources = [source_path(name) for name in (
            "source_paths.py",
            "validate_evolution.py", "validate_linear.py", "CosmologySimulation.h", "CosmologyPhysics.h",
            "ExecutionMetadata.h", "runtime_metadata.py",
            "CosmologyConfig.h", "tests/CompareCosmologyEvolution.cpp", "reference/FastPMEvolution.c",
            "reference/build_fastpm_evolution.sh")]
        sources.extend(self.executables.values())
        if args.fastpm_manifest:
            sources.append(args.fastpm_manifest.resolve())
        ## @var hashes
        # @brief Absolute source/artifact paths mapped to expected SHA256 values; changed bytes invalidate provenance.
        self.hashes = {str(path): sha256(path) for path in sources}
        ## @var fixtures
        # @brief Retained fixtures state owned by this instance; see the initialization and workflow contract.
        ## @var outputs
        # @brief Retained outputs state owned by this instance; see the initialization and workflow contract.
        ## @var spectra
        # @brief Recorded Fourier-shell measurements and their declared estimator conventions.
        self.fixtures, self.outputs, self.spectra = {}, {}, {}
        ## @var report
        # @brief Structured campaign/audit report; recorded failures are not retroactively changed.
        self.report = {"schema": "ippl-fastpm-evolution-v1", "started_utc": datetime.now(timezone.utc).isoformat(),
                       "parameters": self.parameters, "limits": Limits, "quick": args.quick,
                       "hashes_before": self.hashes, "runs": [], "checks": [], "comparisons": [],
                       "time_refinement": [], "mesh_refinement": [], "fixtures": {}, "accepted_baseline":
                       "User accepted prior frozen/pancake discrepancies; original limits/results unchanged",
                       "limitations": ["Synthetic smooth ICs, not a Gaussian LCDM halo-statistics qualification",
                           "Native Nyquist and float32 behavior preserved; raw3D trajectories characterized only",
                           "Fixed particle sampling; finer mesh is not a Vlasov continuum limit",
                           "No analytic post-shell-crossing reference", "Local MPI1–4, one host thread, backend recorded; no scaling claim"],
                       "passed": False, "complete": False}
        self.save()

    ## @brief Persist the complete current report without discarding recorded failed checks.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def save(self):
        self.report["failed_checks"] = [check for check in self.report["checks"] if not check["passed"]]
        (self.root / "results.json").write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n")

    ## @brief Record the named predeclared check and its diagnostic values.
    # @see cosmology_tools
    #
    # @param name Stable artifact/run/check identifier as defined by the caller.
    # @param passed Boolean result of the predeclared check, not of merely completing execution.
    # @param values Recorded diagnostic values in the metric/schema defined by the caller.
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def check(self, name, passed, **values):
        self.report["checks"].append({"name": name, "passed": bool(passed), **values})

    ## @brief Record whether a measured value stays within its explicit upper budget.
    # @see cosmology_tools
    #
    # @param name Stable artifact/run/check identifier as defined by the caller.
    # @param value Measured or serialized scalar in the declared metric/schema; no normalization is inferred.
    # @param limit Predeclared acceptance limit in the reported metric's units.
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def upper(self, name, value, limit):
        self.check(name, value is not None and np.isfinite(value) and value <= limit,
                   value=value, limit=limit)

    ## @brief Prepare the documented module workflow.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def prepare(self):
        for kind in ("pancake", "coupled3d"):
            frame, description = make_fixture(kind, self.parameters["particle_grid"])
            path = self.root / (kind + "-initial.csv")
            frame.sample(frac=1, random_state=77821).to_csv(path, index=False, float_format="%.17g")
            self.hashes[str(path)] = sha256(path)
            self.fixtures[kind] = frame
            self.report["fixtures"][kind] = {**description, "path": str(path), "sha256": self.hashes[str(path)],
                                              "resolved_modes": resolved_modes(kind).tolist()}
        self.save()

    ## @brief Evaluate the native factors helper in the documented module workflow.
    # @see cosmology_tools
    #
    # @param output Output path; use a fresh destination where the workflow rejects existing evidence.
    # @param name Stable artifact/run/check identifier as defined by the caller.
    # @param steps Positive PM step count; endpoints are uniform in log(a) for the imported drivers.
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
    def native_factors(self, output, name, steps):
        data = pd.read_csv(output / "factors.csv", float_precision="round_trip")
        validate_factor_schedule(data, steps, self.parameters["a_initial"], self.parameters["a_final"])
        nodes, weights = np.polynomial.legendre.leggauss(64)
        omega = self.parameters["omega_m"]
        def integral(lower, upper, power):
            lo, hi = np.log(np.asarray(lower)), np.log(np.asarray(upper))
            loga = (lo + hi)[:, None] / 2 + (hi - lo)[:, None] * nodes / 2
            a = np.exp(loga)
            return (hi - lo) * np.sum(weights / (a**power * np.sqrt(omega / a**3 + 1 - omega)), axis=1) / 2
        for field, lo, hi, power in (("drift", "a0", "a1", 2), ("canonical_kick0", "a0", "ah", 1),
                                      ("canonical_kick1", "ah", "a1", 1)):
            expected = integral(data[lo], data[hi], power)
            relative = float(np.max(np.abs(data[field] / expected - 1)))
            self.upper(name + "/native_" + field, relative, Limits["native_time_factor_relative"])

    ## @brief Execute case.
    # @see cosmology_tools
    #
    # @param case Immutable case specification identifying the fixture, resolutions and schedule.
    # @param code Solver identifier (IPPL, native FastPM or GADGET) selected by the protocol.
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def run_case(self, case, code):
        p = self.parameters
        name = case.name + "_" + code
        output = self.root / name
        input_path = self.report["fixtures"][case.fixture]["path"]
        command = [self.args.mpiexec, *self.args.mpi_arg, self.args.numproc_flag, str(case.ranks),
                   str(self.executables[code]), str(p["particle_grid"]), str(case.mesh), str(p["box_size"]),
                   str(p["omega_m"]), str(p["a_initial"]), str(p["a_final"]), str(case.steps),
                   str(p["checkpoints"]), input_path, str(output)]
        launch(command, self.root / (name + ".log"), self.args.timeout)
        archived = archive_snapshots(output)
        metadata = dict(line.split("=", 1) for line in (output / "metadata.txt").read_text().splitlines() if "=" in line)
        execution = validate_runtime_metadata(metadata, case.ranks, code=code)
        table = pd.read_csv(output / "checkpoints.csv", float_precision="round_trip")
        contract = {"n_particles_grid": p["particle_grid"], "n_grid": case.mesh, "n_steps": case.steps,
                    "n_checkpoints": p["checkpoints"], "ranks": case.ranks,
                    "box_size": p["box_size"], "omega_m": p["omega_m"],
                    "a_initial": p["a_initial"], "a_final": p["a_final"]}
        if any(float(metadata[key]) != value for key, value in contract.items()):
            raise ValueError(f"{name}: execution metadata mismatch")
        if code == "fastpm":
            native = {"upstream_commit": FastPMCommit, "force_type": "FASTPM_FORCE_PM",
                      "integrator": "fastpm_solver_evolve", "momentum_precision_bits": "32"}
            if any(metadata.get(key) != value for key, value in native.items()):
                raise ValueError("Native reference differs from approved contract")
            self.native_factors(output, name, case.steps)
        expected_steps = np.arange(p["checkpoints"] + 1) * (case.steps // p["checkpoints"])
        expected_a = p["a_initial"] * np.exp(np.arange(p["checkpoints"] + 1)
                                              * np.log(p["a_final"] / p["a_initial"]) / p["checkpoints"])
        if (not np.array_equal(table.checkpoint, np.arange(p["checkpoints"] + 1))
                or not np.array_equal(table.step, expected_steps) or not np.isfinite(table.to_numpy()).all()):
            raise ValueError("Incomplete, nonfinite or wrong checkpoint schedule")
        self.upper(name + "/schedule", float(np.max(abs(table.a / expected_a - 1))), Limits["schedule_relative"])
        if code == "fastpm":
            expansion = np.sqrt(p["omega_m"] / table.a**3 + 1 - p["omega_m"])
            self.upper(name + "/background", float(np.max(abs(table.E / expansion - 1))), 1e-12)
        record = {"name": name, **case.__dict__, "code": code, "command": command, "output": str(output),
                  "input_sha256": self.hashes[input_path], "metadata": metadata, "execution": execution,
                  "snapshots": archived, "checkpoints": table.to_dict(orient="records"),
                  "diagnostic_note": "IPPL mass_error is a CIC mesh sum; FastPM mass_error is particle-count mass. Not interchangeable."}
        self.report["runs"].append(record)
        self.outputs[(case, code)] = output
        original = self.fixtures[case.fixture]
        initial_mean = original[["px", "py", "pz"]].to_numpy().mean(axis=0)
        record["imported_mean_momentum"] = initial_mean.tolist()
        modes = resolved_modes(case.fixture)
        record["density_modes"], record["observations"] = [], []
        for checkpoint, a in enumerate(expected_a):
            snapshot = read_snapshot(str(output), checkpoint, case.ranks, p["particle_grid"])
            positions = snapshot[["x", "y", "z"]].to_numpy()
            momentum = snapshot[["px", "py", "pz"]].to_numpy()
            self.check(f"{name}/{checkpoint}/periodic_finite_complete", bool((positions >= 0).all() and (positions < p["box_size"]).all()))
            if checkpoint == 0:
                dx = float(np.max(abs(periodic_difference(positions, original[["x", "y", "z"]].to_numpy(), p["box_size"]))))
                self.upper(name + "/initial_position", dx, Limits["initial_position_eps_box"] * np.finfo(float).eps * p["box_size"])
                self.check(name + "/identical_imported_momentum", np.array_equal(momentum, original[["px", "py", "pz"]].to_numpy()))
            net = float(np.linalg.norm(momentum.mean(axis=0) - initial_mean)) / vector_rms(momentum)
            self.upper(f"{name}/{checkpoint}/net_momentum", net, Limits[f"net_momentum_{code}_relative"])
            coefficient = density_modes(positions, p["box_size"], modes)
            self.spectra[(case, code, checkpoint)] = coefficient
            record["density_modes"].append({"checkpoint": checkpoint, "a": float(a),
                                            "real": coefficient.real.tolist(), "imag": coefficient.imag.tolist()})
            observation = {"checkpoint": checkpoint, "a": float(a), "momentum_rms": vector_rms(momentum)}
            if case.fixture == "pancake":
                observation.update(planar_ordering(snapshot, p["particle_grid"], p["box_size"]))
                if checkpoint == p["checkpoints"]:
                    self.check(name + "/actual_sheet_crossing", observation["minimum_sampled_jacobian"] < -.01,
                               minimum_sampled_jacobian=observation["minimum_sampled_jacobian"])
            record["observations"].append(observation)
        if code == "ippl":
            record["mass_diagnostic_above_prior_strict_limit"] = (float(metadata["maximum_mass_error"])
                > Limits["old_mass_diagnostic_limit_recorded_not_overridden"])
        print(name, flush=True)
        self.save()

    ## @brief Compare codes.
    # @see cosmology_tools
    #
    # @param case Immutable case specification identifying the fixture, resolutions and schedule.
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def compare_codes(self, case):
        p = self.parameters
        for checkpoint in range(p["checkpoints"] + 1):
            left = read_snapshot(str(self.outputs[(case, "ippl")]), checkpoint, case.ranks, p["particle_grid"])
            right = read_snapshot(str(self.outputs[(case, "fastpm")]), checkpoint, case.ranks, p["particle_grid"])
            phase = phase_space_metrics(left, right, p["box_size"], case.mesh)
            density = density_comparison(self.spectra[(case, "ippl", checkpoint)], self.spectra[(case, "fastpm", checkpoint)])
            label = f"{case.name}/{checkpoint}/cross"
            if case.fixture == "pancake":
                self.upper(label + "/position_cells", phase["position_cells"], Limits["planar_cross_position_cells"])
                self.upper(label + "/momentum_relative", phase["momentum_relative"], Limits["planar_cross_momentum_relative"])
            self.check(label + "/nonzero_resolved_signal", density["normalization_defined"])
            if density["normalization_defined"]:
                self.upper(label + "/resolved_power", abs(density["power_ratio"] - 1), Limits["resolved_power_fraction"][case.mesh])
                self.upper(label + "/resolved_complex", density["complex_relative"], Limits["resolved_complex_relative"][case.mesh])
                self.check(label + "/resolved_correlation", density["correlation"] >= Limits["resolved_correlation_minimum"][case.mesh],
                           value=density["correlation"], minimum=Limits["resolved_correlation_minimum"][case.mesh])
            shells = []
            radius = np.linalg.norm(resolved_modes(case.fixture), axis=1)
            for lo, hi in zip((.5, 1.5, 2.5), (1.5, 2.5, 4.5)):
                mask = (radius >= lo) & (radius < hi)
                shells.append({"lower": lo, "upper": hi, "pairs": int(mask.sum()), **density_comparison(
                    self.spectra[(case, "ippl", checkpoint)][mask], self.spectra[(case, "fastpm", checkpoint)][mask])})
            self.report["comparisons"].append({"case": case.name, "checkpoint": checkpoint, "phase_space": phase,
                                                "resolved_density": density, "shells_characterization_only": shells,
                                                "raw_3d_trajectory_is_gated": False})
        self.save()

    ## @brief Compare ranks.
    # @see cosmology_tools
    #
    # @param case Immutable case specification identifying the fixture, resolutions and schedule.
    # @param code Solver identifier (IPPL, native FastPM or GADGET) selected by the protocol.
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def compare_ranks(self, case, code):
        base = Case(case.fixture, case.mesh, case.steps, 1)
        p = self.parameters
        for checkpoint in range(p["checkpoints"] + 1):
            left = read_snapshot(str(self.outputs[(case, code)]), checkpoint, case.ranks, p["particle_grid"])
            right = read_snapshot(str(self.outputs[(base, code)]), checkpoint, 1, p["particle_grid"])
            result = phase_space_metrics(left, right, p["box_size"], case.mesh)
            label = f"{case.name}/{code}/{checkpoint}/rank"
            floor = Limits["initial_position_eps_box"] * np.finfo(float).eps * case.mesh
            self.upper(label + "/position_cells", result["position_cells"], Limits[f"rank_{code}_position_cells"] + floor)
            self.upper(label + "/momentum_relative", result["momentum_relative"], Limits[f"rank_{code}_momentum_relative"])

    ## @brief Evaluate the convergence helper in the documented module workflow.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def convergence(self):
        p = self.parameters
        for kind in ("pancake", "coupled3d"):
            for code in ("ippl", "fastpm"):
                for checkpoint in (4, 8):
                    group = [Case(kind, 32, steps) for steps in (64, 128, 256)]
                    snapshots = [read_snapshot(str(self.outputs[(case, code)]), checkpoint, 1, p["particle_grid"]) for case in group]
                    differences = [phase_space_metrics(a, b, p["box_size"], 32) for a, b in zip(snapshots[:-1], snapshots[1:])]
                    density = [density_comparison(self.spectra[(a, code, checkpoint)], self.spectra[(b, code, checkpoint)])
                               for a, b in zip(group[:-1], group[1:])]
                    label = f"{kind}/{code}/{checkpoint}/time"
                    row = {"name": label, "phase_space_differences": differences, "density_differences": density, "orders": {}}
                    lower, upper = Limits["time_precross_ratio_range"] if checkpoint == 4 else (Limits["time_postcross_ratio_minimum"], None)
                    for quantity in ("position_cells", "momentum_relative", "complex_relative"):
                        values = [d[quantity] for d in (density if quantity == "complex_relative" else differences)]
                        floor = Limits["time_precision_" + quantity]
                        result = refinement(*values, floor, lower, upper)
                        row["orders"][quantity] = result
                        self.check(label + "/" + quantity, result["passed"], **{k: v for k, v in result.items() if k != "passed"},
                                   coarse_difference=values[0], fine_difference=values[1], precision_floor=floor)
                    if checkpoint == 8:
                        self.upper(label + "/finest_position", differences[-1]["position_cells"], Limits["time_finest_position_cells"])
                        self.upper(label + "/finest_momentum", differences[-1]["momentum_relative"], Limits["time_finest_momentum_relative"])
                    self.report["time_refinement"].append(row)
                for lo, hi in ((16, 32), (32, 64)):
                    coarse, fine = Case(kind, lo, 256), Case(kind, hi, 256)
                    density = density_comparison(self.spectra[(coarse, code, 8)], self.spectra[(fine, code, 8)])
                    row = {"fixture": kind, "code": code, "meshes": [lo, hi], "resolved_density": density,
                           "interpretation": "fixed particles; not a continuum Vlasov convergence test"}
                    if lo == 32:
                        self.upper(f"{kind}/{code}/mesh_finest_power", abs(density["power_ratio"] - 1),
                                   Limits["mesh_finest_pair_resolved_power_fraction"])
                    self.report["mesh_refinement"].append(row)
        self.save()

    ## @brief Execute the documented module workflow.
    # @see cosmology_tools
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
    def run(self):
        self.prepare()
        print(f"Evidence: {self.root}", flush=True)
        for case in cases(self.args.quick):
            for code in ("ippl", "fastpm"):
                self.run_case(case, code)
                if case.ranks > 1:
                    self.compare_ranks(case, code)
            self.compare_codes(case)
        if not self.args.quick:
            self.convergence()
        after = {path: sha256(path) for path in self.hashes}
        self.report["hashes_after"] = after
        self.check("all sources, inputs and executables unchanged during campaign", after == self.hashes)
        self.report["complete"] = True
        self.report["passed"] = all(check["passed"] for check in self.report["checks"])
        self.report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        self.save()
        print(f"{len(self.report['runs'])} runs, {len(self.report['checks'])} checks, "
              f"{len(self.report['failed_checks'])} failed: {self.root / 'results.json'}", flush=True)
        for failure in self.report["failed_checks"]:
            print(json.dumps(failure))
        return 0 if self.report["passed"] else 1


## @brief Parse the documented command-line interface and execute its selected workflow, retaining nonzero failures.
# @see cosmology_tools
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ippl-exe", type=Path, required=True)
    parser.add_argument("--fastpm-exe", type=Path, required=True)
    parser.add_argument("--fastpm-manifest", type=Path)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--numproc-flag", default="-n")
    parser.add_argument("--mpi-arg", action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    campaign = Campaign(args)
    try:
        return campaign.run()
    except Exception as error:
        campaign.report["execution_error"] = str(error)
        campaign.save()
        raise


## @cond CLI_DISPATCH
if __name__ == "__main__":
    raise SystemExit(main())
## @endcond
