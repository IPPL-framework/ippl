#!/usr/bin/env python3
## @file validate_native_ic.py
# @brief Compare native 1LPT states with immutable imported fixtures and MPI/GPU variants.
# @ingroup cosmology_python
# @details Canonical records store x and p=a^2 dx/d(H0 t), both in Mpc/h.
# For the cell-centred, x-fast lattice q, Psi=(x-q)_periodic/D(a).
# The forward-normalized DFT satisfies delta(k)=-i k.Psi(k), after removing
# exp(i*pi*(nx+ny+nz)/N), the half-cell sampling phase. The independent target
# is mode_gaussian(seed,n)*sqrt(P_BBKS(k)/L^3), without realization rescaling.
# Particle differences use minimum-image x residuals and ID matching. Raw
# particle density modes use delta(k)=Nparticle^-1 sum exp(-i k.x); no Poisson
# term is subtracted. Optional shell spectra reuse the frozen campaign
# estimator, identically for reference and candidates. This tool never evolves
# particles or changes production tolerances and cannot certify billion-particle
# memory safety from small validation cases.
"""Audit already-produced native runs on Merlin; preserve references and evidence.

Examples (inside a Merlin allocation):
  python validate_native_ic.py --grid 64 --reference /data/.../ics/n64.bin \
    --candidate /data/.../native64-r1 --candidate /data/.../native64-r2 \
    --epoch initial --output /data/.../validation64-initial
  python validate_native_ic.py --grid 64 --reference /data/.../runs/n64 \
    --candidate /data/.../native64-r1 --epoch final \
    --power-source /data/.../ok-check/source --output /data/.../validation64-final

Initial acceptance inherits the existing independent Gaussian coefficient
budget (1e-8). Exact half-cell indexing, periodicity, cutoff and momentum
precision are additionally checked. Final differences are descriptive unless
--limits supplies explicit budgets and their justification before execution.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np

import gaussian_fixture as gaussian
from quijote_analysis import ParticleSource, resolve_sources
from validate_linear import GaussianCoefficientTolerance, bbks_spectrum, growth_reference

## @brief Stable report schema; all numerical tolerances are retained in each result.
Schema = "ippl-native-ic-validation-v1"
## @brief Selected signed modes include axis, oblique and k approximately 1 probes.
ProbeModes = ((1, 0, 0), (0, 2, 0), (0, 0, 3), (2, -1, 0),
              (3, 2, -1), (8, 0, 0), (16, 0, 0), (24, 0, 0), (26, 0, 0))


## @brief Reject a violated validation contract without changing its tolerance.
# @param condition Boolean invariant that must hold for the declared contract.
# @param message Failure explanation retained in the validation report.
# @return None; raises ValueError when the invariant fails.
def require(condition, message):
    if not condition:
        raise ValueError(message)


## @brief Hash exact bytes in bounded host buffers.
# @param path Existing file whose exact bytes are hashed in bounded buffers.
# @return Lowercase SHA256 digest of the retained bytes.
def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


## @brief Describe immutable input/output bytes; particle headers are audited separately.
# @param path Existing regular artifact file to resolve and describe.
# @return Absolute path, byte count and SHA256 provenance mapping.
def product(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


## @brief Publish new finite JSON and refuse to overwrite previous validation evidence.
# @param path New report path; an existing file is never replaced.
# @param data JSON-compatible report with finite numerical values.
# @return None; publishes the report or raises an I/O/serialization error.
def saveJson(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


## @brief Parse finite key=value runtime metadata without interpreting strings as code.
# @param path Existing production metadata.txt file containing unique key=value lines.
# @return Mapping of metadata keys to their unchanged string values.
def readMetadata(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            require(key not in values, f"Duplicate runtime metadata key {key}")
            values[key] = value
    return values


## @brief Return dx-L*rint(dx/L), preserving the signed minimum-image residual.
# @param left First position array in comoving Mpc/h.
# @param right ID-matched reference position array in comoving Mpc/h.
# @param box Positive periodic box side in comoving Mpc/h.
# @return Signed minimum-image differences in comoving Mpc/h.
def minimumImage(left, right, box):
    delta = np.asarray(left) - np.asarray(right)
    return delta - box * np.rint(delta / box)


## @brief Vector RMS sqrt(mean(sum(v_i^2))); not the componentwise RMS.
# @param values Finite array of particle vectors with components on the final axis.
# @return Root mean squared vector magnitude in the input units.
def vectorRms(values):
    return float(np.sqrt(np.mean(np.sum(np.asarray(values)**2, axis=1))))


## @brief Construct the ID-indexed cell-centred lattice with x varying fastest.
# @param grid Positive integer particle-lattice side length.
# @param box Positive comoving box side in Mpc/h.
# @return Grid-cubed by three positions in x-fast uint64-ID order.
def lattice(grid, box):
    ids = np.arange(grid**3, dtype=np.uint64)
    return (np.column_stack((ids % grid, ids // grid % grid, ids // (grid * grid))) + .5) * (box / grid)


## @brief Audit full ID coverage and load bounded validation-sized states in ID order.
# @details No all-particle load is implicit: maxParticles must cover N before
# allocation. Each canonical shard is checked for finite phase space, ID range,
# duplicate IDs within/across chunks, exact count and consistent cosmology.
# @param specification Canonical IC file, snapshots.csv file, or run directory.
# @param epoch Exact initial or final epoch selector for a snapshot manifest.
# @param grid Expected particle-lattice side, giving the exact global count grid cubed.
# @param maxParticles Explicit host-memory ceiling on particles loaded into arrays.
# @return ID-ordered phase space, validated header and immutable input provenance.
def loadState(specification, epoch, grid, maxParticles):
    specification = Path(specification).resolve(strict=True)
    if specification.is_dir():
        specification = specification / "snapshots.csv"
    selector = epoch if specification.name == "snapshots.csv" else None
    paths, scale = resolve_sources([specification], selector)
    source = ParticleSource(paths, expected_a=scale)
    require(source.count == grid**3, "Particle count does not match grid cubed")
    require(source.count <= maxParticles, "Validation particle-memory cap exceeded")
    if epoch == "initial":
        require(math.isclose(source.header["a"], .01, rel_tol=1e-12), "Expected z=99 IC")
    else:
        require(math.isclose(source.header["a"], 1., rel_tol=1e-12), "Expected z=0 final snapshot")
    inputs = source.validate()
    phase = np.empty((source.count, 6), dtype=np.float64)
    for records in source.records():
        phase[records["id"], :3] = records["position"]
        phase[records["id"], 3:] = records["momentum"]
    return {"phase": phase, "header": source.header, "inputs": inputs,
            "specification": product(specification), "run_directory": str(specification.parent)
            if specification.name == "snapshots.csv" else None}


## @brief Compute ID-matched periodic position and momentum residuals without normalization fitting.
# @param candidate Audited candidate state returned by loadState.
# @param reference Audited saved or alternate-layout reference state.
# @param grid Particle-lattice side defining the cell length L/grid.
# @return Periodic position and momentum residual metrics with declared normalizations.
def compareStates(candidate, reference, grid):
    left, right = candidate["phase"], reference["phase"]
    require(left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all(), "Invalid phase-space shapes/values")
    for key in ("total_count", "a", "box_mpc_h", "omega_m", "omega_lambda", "hubble", "particle_mass_msun_h"):
        require(math.isclose(float(candidate["header"][key]), float(reference["header"][key]), rel_tol=2e-14, abs_tol=0),
                f"Header mismatch: {key}")
    box = reference["header"]["box_mpc_h"]
    dx = minimumImage(left[:, :3], right[:, :3], box)
    dp = left[:, 3:] - right[:, 3:]
    momentumRms = vectorRms(right[:, 3:])
    momentumMax = float(np.abs(right[:, 3:]).max())
    return {"position_rms_mpc_h": vectorRms(dx), "position_rms_cells": vectorRms(dx) / (box / grid),
            "position_linf_mpc_h": float(np.abs(dx).max()), "position_linf_box": float(np.abs(dx).max()) / box,
            "momentum_rms_mpc_h": vectorRms(dp), "momentum_linf_mpc_h": float(np.abs(dp).max()),
            "momentum_relative_rms": vectorRms(dp) / momentumRms if momentumRms else None,
            "momentum_relative_linf": float(np.abs(dp).max()) / momentumMax if momentumMax else None,
            "reference_momentum_rms_mpc_h": momentumRms, "reference_momentum_linf_mpc_h": momentumMax,
            "identical_momentum_components": int(np.count_nonzero(left[:, 3:] == right[:, 3:])),
            "momentum_components": int(left[:, 3:].size)}


## @brief Reconstruct all sampled linear density coefficients from native 1LPT particles.
# @details The half-cell phase is removed only after the forward FFT; components
# are physical xyz, while arrays are laid out zyx for x-fast particle IDs.
# All coefficients, including DC and out-of-band modes, are checked. The
# existing 1e-8 independent Gaussian budget includes BBKS quadrature and growth
# reference differences; it is not relaxed in response to a measured result.
# @param state Audited native or saved initial canonical state at redshift 99.
# @param grid Particle-lattice and displacement-FFT side length.
# @param seed Fixed unsigned realization seed; no seed search is performed.
# @param cutoff Positive spherical integer-mode ceiling strictly below Nyquist.
# @param momentumPrecision Declared double or once-rounded float32 momentum convention.
# @return Deterministic coefficient, cutoff, DC and momentum checks plus descriptive shell statistics.
def initialAudit(state, grid, seed, cutoff, momentumPrecision):
    require(0 < cutoff < grid // 2, "Initial oracle requires a positive cutoff below Nyquist")
    phase = state["phase"]
    header = state["header"]
    box, a, omega = header["box_mpc_h"], header["a"], header["omega_m"]
    parameters = dict(gaussian.DefaultCosmology)
    for key, expected in (("box_mpc_h", parameters["box_size"]), ("omega_m", parameters["Omega_m"]),
                          ("hubble", parameters["hubble"])):
        require(math.isclose(header[key], expected, rel_tol=1e-14), "This saved-fixture oracle requires the declared ok-check cosmology")
    q = lattice(grid, box)
    displacement = minimumImage(phase[:, :3], q, box)
    require(float(np.abs(displacement).max()) < box / 4,
            "Displacement too large for unambiguous IC minimum-image recovery")
    growth, rate = growth_reference(a, omega)
    expansion = math.sqrt(omega / a**3 + 1 - omega)
    predicted = a*a*expansion*rate*displacement
    if momentumPrecision == "float32":
        require(np.array_equal(phase[:, 3:], phase[:, 3:].astype(np.float32).astype(np.float64)),
                "Declared float32 momentum is not exactly float32 representable")
        predicted = predicted.astype(np.float32).astype(np.float64)
    momentumDenominator = vectorRms(predicted)
    require(momentumDenominator > 0, "Cannot normalize a zero momentum field")
    momentumError = vectorRms(phase[:, 3:] - predicted) / momentumDenominator
    roundingBound = np.finfo(np.float32).eps if momentumPrecision == "float32" else 0.
    momentumLimit = GaussianCoefficientTolerance + roundingBound
    meanDisplacement = displacement.mean(axis=0)
    displacementRms = vectorRms(displacement)
    require(displacementRms > 0, "Cannot normalize a zero displacement field")
    dcRelative = float(np.linalg.norm(meanDisplacement)) / displacementRms

    signed = np.rint(np.fft.fftfreq(grid) * grid).astype(np.int32)
    iz, iy, ix = signed[:, None, None], signed[None, :, None], signed[None, None, :]
    reconstructed = np.zeros((grid, grid, grid), dtype=np.complex128)
    for component, integers in enumerate((ix, iy, iz)):
        componentFft = np.fft.fftn((displacement[:, component] / growth).reshape(grid, grid, grid), norm="forward")
        reconstructed -= 1j * (2 * math.pi / box) * integers * componentFft
    reconstructed *= np.exp(-1j * math.pi * (ix + iy + iz) / grid)
    modes = np.asarray([mode for mode in itertools.product(range(-cutoff, cutoff + 1), repeat=3)
                        if 0 < sum(value * value for value in mode) <= cutoff**2], dtype=np.int32)
    wave = 2 * math.pi / box * np.linalg.norm(modes, axis=1)
    power = bbks_spectrum(wave, parameters)
    coefficients = np.asarray([gaussian.mode_gaussian(seed, mode) for mode in modes]) * np.sqrt(power / box**3)
    indices = tuple(modes[:, component] % grid for component in (2, 1, 0))
    recovered = reconstructed[indices]
    norm = float(np.linalg.norm(coefficients))
    require(norm > 0, "Zero target coefficient norm")
    residual = recovered - coefficients
    relativeL2 = float(np.linalg.norm(residual)) / norm
    relativeLinf = float(np.abs(residual).max()) / float(np.abs(coefficients).max())
    reconstructed[indices] = 0
    outsideRelativeL2 = float(np.linalg.norm(reconstructed.ravel())) / norm
    coefficientHash = hashlib.sha256(gaussian.RngDomain + modes.astype("<i4").tobytes()
                                     + coefficients.astype("<c16").tobytes()).hexdigest()
    # Summing both conjugates gives identical per-pair power. Statistical
    # variability is reported, while the deterministic coefficient test gates.
    normalized = box**3 * np.abs(recovered)**2 / power
    independentModes = len(modes) // 2
    sphere = np.floor(np.linalg.norm(modes, axis=1)).astype(int)
    shells = []
    for shell in np.unique(sphere):
        selected = sphere == shell
        shells.append({"shell": int(shell), "full_mode_count": int(selected.sum()),
                       "mean_k_h_mpc": float(wave[selected].mean()),
                       "p_recovered_z0_linear_mpc_h_cubed": float((box**3 * np.abs(recovered[selected])**2).mean()),
                       "p_realization_target_z0_linear_mpc_h_cubed": float((box**3 * np.abs(coefficients[selected])**2).mean()),
                       "p_ensemble_target_z0_linear_mpc_h_cubed": float(power[selected].mean()),
                       "gaussian_fractional_stddev": float(math.sqrt(2 / selected.sum()))})
    return {"passed": bool(relativeL2 <= GaussianCoefficientTolerance and relativeLinf <= GaussianCoefficientTolerance
                           and dcRelative <= GaussianCoefficientTolerance
                           and outsideRelativeL2 <= GaussianCoefficientTolerance and momentumError <= momentumLimit),
            "coefficient_relative_l2": relativeL2, "coefficient_relative_linf": relativeLinf,
            "outside_band_relative_l2": outsideRelativeL2, "coefficient_limit": GaussianCoefficientTolerance,
            "coefficient_sha256": coefficientHash, "coefficient_count_full": len(modes),
            "gaussian_mean_power_over_ensemble": float(normalized.mean()),
            "gaussian_mean_power_fractional_stddev": 1 / math.sqrt(independentModes),
            "gaussian_statistics_are_descriptive": True, "momentum_relative_rms": momentumError,
            "momentum_limit": momentumLimit, "momentum_rounding_bound": float(roundingBound),
            "momentum_precision": momentumPrecision, "growth_D": growth, "growth_f": rate,
            "displacement_rms_mpc_h": displacementRms, "displacement_dc_relative": dcRelative,
            "displacement_linf_mpc_h": float(np.abs(displacement).max()),
            "mean_displacement_mpc_h": meanDisplacement.tolist(),
            "spectrum": shells,
            "limits_basis": "Existing validate_linear.GaussianCoefficientTolerance; optional two-rounding float32 epsilon for momentum only"}


## @brief Compare low-cost exact particle density probes; no mesh, shot subtraction or fitting.
# @param candidate Audited candidate canonical phase-space state.
# @param reference Audited ID-matched reference state in the same periodic box.
# @param grid Particle-lattice side used to exclude unresolved probe modes.
# @return Selected direct complex density modes and raw-power ratios without shot subtraction.
def densityProbes(candidate, reference, grid):
    box = reference["header"]["box_mpc_h"]
    modes = [mode for mode in ProbeModes if max(abs(value) for value in mode) < grid // 2]
    rows = []
    for mode in modes:
        wave = np.asarray(mode) * (2 * math.pi / box)
        values = []
        for state in (reference, candidate):
            total = 0j
            for offset in range(0, len(state["phase"]), 65536):
                total += np.exp(-1j * (state["phase"][offset:offset+65536, :3] @ wave)).sum()
            values.append(total / len(state["phase"]))
        before, after = values
        denominator = abs(before)
        rows.append({"mode": list(mode), "k_h_mpc": float(np.linalg.norm(wave)),
                     "reference_re": float(before.real), "reference_im": float(before.imag),
                     "candidate_re": float(after.real), "candidate_im": float(after.imag),
                     "absolute_complex_error": float(abs(after - before)),
                     "relative_complex_error": float(abs(after - before) / denominator) if denominator > 1e-15 else None,
                     "raw_power_ratio": float(abs(after)**2 / abs(before)**2) if denominator > 1e-15 else None})
    return rows


## @brief Audit the native preflight artifact according to its explicit schema.
# @details Only native candidates require this receipt; older imported
# reference fixtures need not have an artifact that did not exist at the time.
# @param runDirectory Native run directory containing IC checks, initial spectrum and runtime metadata.
# @param grid Expected native particle and force grid side.
# @param seed Expected fixed mode_hash_v1 realization seed.
# @param cutoff Expected native spherical integer-mode cutoff.
# @param momentumPrecision Expected double or float32 initial momentum declaration.
# @return Verified native acceptance artifact descriptors and unchanged report contents.
def preflightAudit(runDirectory, grid, seed, cutoff, momentumPrecision):
    directory = Path(runDirectory)
    path = directory / "ic_check.json"
    value = json.loads(path.read_text())
    require(value.get("schema") == "ippl-native-ic-check-v1", "Unsupported native IC preflight schema")
    require(value.get("passed") is True, "Native IC preflight did not pass")
    require(value.get("native_1lpt") is True and value.get("ic_mode") == "gaussian", "Preflight did not audit native Gaussian 1LPT")
    require(value.get("failures") == [], "Native IC preflight contains failures")
    checks = value.get("checks")
    require(isinstance(checks, dict) and checks, "Native IC preflight lacks checks")
    for name in ("particle_count", "finite_wrapped_phase_space", "unit_pm_weights", "physical_particle_mass",
                 "expected_id_signatures", "native_lagrangian_lattice", "native_momentum_relation",
                 "native_finite_modes", "native_dc_zero", "native_cutoff_and_nyquist_zero",
                 "native_inverse_reality", "native_declared_mode_coefficients", "native_selected_mode_recovery"):
        check = checks.get(name)
        require(check is True or (isinstance(check, dict) and check.get("passed") is True),
                f"Native IC check not passed: {name}")
    config = value["config"]
    for key, expected in (("np", grid), ("seed", seed), ("ic_mode_cutoff", cutoff),
                          ("ic_rng", "mode_hash_v1"), ("ic_momentum_precision", momentumPrecision)):
        require(config[key] == expected, f"Native IC check config mismatch: {key}")
    require(math.isclose(config["a_initial"], .01, rel_tol=1e-12), "Preflight epoch mismatch")
    require(math.isclose(config["box_mpc_h"], gaussian.DefaultCosmology["box_size"], rel_tol=1e-14), "Preflight box mismatch")
    require(isinstance(value.get("selected_modes"), list) and value["selected_modes"], "Missing mode-recovery diagnostics")
    for mode in value["selected_modes"]:
        require(mode.get("passed") is True and mode.get("declared_passed") is True, "Mode-recovery/declared-coefficient diagnostic failed")
        for key in ("expected_initial_real", "expected_initial_imaginary", "recovered_real", "recovered_imaginary", "absolute_error", "tolerance",
                    "declared_initial_real", "declared_initial_imaginary", "declared_absolute_error", "declared_tolerance"):
            require(math.isfinite(float(mode[key])), "Nonfinite mode diagnostic")
        require(0 <= mode["absolute_error"] <= mode["tolerance"], "Inconsistent mode acceptance record")
        require(0 <= mode["declared_absolute_error"] <= mode["declared_tolerance"], "Inconsistent declared-coefficient acceptance record")
    metadata = readMetadata(directory / "metadata.txt")
    for key, expected in (("np", grid), ("seed", seed), ("ic_mode_cutoff", cutoff)):
        require(int(metadata[key]) == expected, f"Unexpected native metadata {key}")
    require(metadata["ic_mode"] == "gaussian" and metadata["ic_rng"] == "mode_hash_v1", "Wrong native IC/RNG mode")
    require(metadata["ic_momentum_precision"] == momentumPrecision, "Wrong native momentum precision")
    if "ranks" in config:
        require(int(metadata["ranks"]) == int(config["ranks"]) >= 1, "Preflight/runtime MPI rank mismatch")
    pkPath = directory / "pk_initial.csv"
    require(pkPath.is_file() and pkPath.stat().st_size > 0, "Missing native initial spectrum artifact")
    with pkPath.open(newline="") as stream:
        rows = list(csv.DictReader(line for line in stream if not line.lstrip().startswith("#")))
    require(rows, "Empty native Gaussian initial spectrum")
    previousK, previousShell = -1., -1
    for row in rows:
        require(all(math.isfinite(float(number)) for number in row.values()), "Nonfinite native initial spectrum")
        for key in ("shell_index", "modes_full", "modes_independent"):
            require(float(row[key]) == int(row[key]) and int(row[key]) > 0, "Invalid native spectrum integer count")
        require(float(row["k_mean_h_mpc"]) > previousK and int(row["shell_index"]) > previousShell,
                "Nonmonotonic native initial spectrum")
        require(float(row["p_linear_initial_mpc_h_cubed"]) >= 0 and float(row["p_target_initial_mpc_h_cubed"]) > 0,
                "Invalid native initial linear power")
        require(float(row["gaussian_fractional_sigma"]) > 0, "Missing Gaussian finite-mode uncertainty")
        previousK, previousShell = float(row["k_mean_h_mpc"]), int(row["shell_index"])
    return {"ic_check": product(path), "value": value, "metadata": product(directory / "metadata.txt"),
            "metadata_values": metadata, "pk_initial": product(pkPath)}


## @brief Check explicitly declared metric maxima; no inferred or post-hoc limits.
# @param metrics Named measured scalar errors; undefined values may be None.
# @param limits Explicit nonnegative finite maxima for known metric names.
# @return Per-metric values, declared maxima and boolean acceptance results.
def applyLimits(metrics, limits):
    checks = {}
    for key, maximum in limits.items():
        require(key in metrics, f"Unknown metric limit {key}")
        require(not isinstance(maximum, bool) and isinstance(maximum, (int, float)) and math.isfinite(maximum) and maximum >= 0, "Invalid metric limit")
        value = metrics[key]
        checks[key] = {"value": value, "maximum": float(maximum),
                       "passed": bool(value is not None and math.isfinite(value) and value <= maximum)}
    return checks


## @brief Reuse the frozen Pylians estimator for an optional matched raw P(k) comparison.
# @param candidate Audited candidate canonical particle state.
# @param reference Audited saved reference state at the same epoch.
# @param grid Particle and force mesh side shared by the compared simulations.
# @param analysisGrid Common independent CIC/FFT analysis mesh side.
# @param sourceDirectory Frozen spectrum_tables.py and matching estimator dependency directory.
# @return Matched raw shell spectra, deposition diagnostics and estimator provenance.
def comparePower(candidate, reference, grid, analysisGrid, sourceDirectory):
    sys.path.insert(0, str(sourceDirectory))
    from spectrum_tables import depositDensity, spectrumRows
    for name in ("spectrum_tables", "quijote_analysis", "quijote_io"):
        require(digest(sys.modules[name].__file__) == digest(sourceDirectory / f"{name}.py"),
                f"Loaded {name} differs from selected frozen estimator source")
    box = reference["header"]["box_mpc_h"]
    tables = []
    for state in (reference, candidate):
        chunks = (state["phase"][offset:offset+262144, :3] for offset in range(0, len(state["phase"]), 262144))
        delta, deposition = depositDensity(chunks, box, analysisGrid, grid**3)
        tables.append((spectrumRows(delta, box, 1, forceGrid=grid), deposition))
        del delta
    rows = []
    require(len(tables[0][0]) == len(tables[1][0]), "Spectrum shell counts differ")
    for left, right in zip(tables[0][0], tables[1][0]):
        for key in ("shell_index", "k_mean_h_mpc", "k_low_h_mpc", "k_high_h_mpc", "modes_independent"):
            require(left[key] == right[key], "Spectrum shell geometry differs")
        raw = left["p_raw_mpc_h_cubed"]
        rows.append({"shell_index": left["shell_index"], "k_mean_h_mpc": left["k_mean_h_mpc"],
                     "eligible_geometric": bool(left["within_force_nyquist"] and left["within_analysis_nyquist"]),
                     "reference_raw_power": raw, "candidate_raw_power": right["p_raw_mpc_h_cubed"],
                     "raw_power_ratio": right["p_raw_mpc_h_cubed"] / raw if raw > 0 else None})
    return {"analysis_grid": analysisGrid, "source": {name: product(sourceDirectory / name)
            for name in ("spectrum_tables.py", "quijote_analysis.py", "quijote_io.py")},
            "reference_deposition": tables[0][1], "candidate_deposition": tables[1][1], "rows": rows}


## @brief Run reference/candidate comparisons and retain failure evidence without evolving anything.
# @param args Parsed CLI namespace selecting saved states, candidates, output paths and explicit budgets.
# @return None; writes a new validation report and raises on a failed structural or acceptance check.
def run(args):
    require(platform.node().startswith("merlin-") and os.environ.get("SLURM_JOB_ID"),
            "Scientific validation must execute in a Merlin Slurm allocation")
    require(args.grid >= 4 and args.grid % 2 == 0 and 0 < args.cutoff < args.grid // 2, "Invalid grid/cutoff")
    require(args.max_particles >= args.grid**3, "Particle-memory cap is below requested grid")
    require(args.power_grid >= 4 and args.power_grid % 2 == 0 and args.power_grid <= 512,
            "This small-state validator allows analysis grids 4..512, even")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": Schema, "state": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "runtime": {"host": platform.node(), "slurm_job": os.environ["SLURM_JOB_ID"],
                          "python": sys.version, "command": sys.argv, "source": product(__file__)},
              "settings": {"grid": args.grid, "epoch": args.epoch, "seed": args.seed, "cutoff": args.cutoff,
                           "momentum_precision": args.momentum_precision, "max_particles": args.max_particles},
              "candidates": [], "scope": "Small-state native IC/regression evidence, not billion-particle capacity qualification"}
    try:
        reference = loadState(args.reference, args.epoch, args.grid, args.max_particles)
        report["reference"] = {key: value for key, value in reference.items() if key != "phase"}
        if args.epoch == "initial":
            report["reference_initial_audit"] = initialAudit(reference, args.grid, args.seed, args.cutoff, args.momentum_precision)
            require(report["reference_initial_audit"]["passed"], "Independent oracle rejected the saved IC reference")
        explicitLimits = None
        if args.limits:
            explicitLimits = json.loads(args.limits.read_text())
            require(isinstance(explicitLimits.get("justification"), str) and explicitLimits["justification"].strip(),
                    "Acceptance budgets require a nonempty justification")
            require(isinstance(explicitLimits.get("metrics"), dict), "Acceptance budgets require metrics mapping")
            report["limits"] = {"file": product(args.limits), "value": explicitLimits}
        labels = args.label or [Path(path).parent.name if Path(path).name == "snapshots.csv" else Path(path).name
                                for path in args.candidate]
        require(len(labels) == len(args.candidate) and len(set(labels)) == len(labels), "Require one unique label per candidate")
        for label, path in zip(labels, args.candidate):
            candidate = loadState(path, args.epoch, args.grid, args.max_particles)
            require(candidate["run_directory"] is not None, "Native candidate must have snapshots.csv metadata")
            item = {"label": label, **{key: value for key, value in candidate.items() if key != "phase"}}
            report["candidates"].append(item)
            item["preflight"] = preflightAudit(candidate["run_directory"], args.grid, args.seed, args.cutoff, args.momentum_precision)
            item["particle_metrics"] = compareStates(candidate, reference, args.grid)
            item["density_probes"] = densityProbes(candidate, reference, args.grid)
            if args.epoch == "initial":
                item["initial_audit"] = initialAudit(candidate, args.grid, args.seed, args.cutoff, args.momentum_precision)
                require(item["initial_audit"]["coefficient_sha256"] == report["reference_initial_audit"]["coefficient_sha256"],
                        "Expected Fourier coefficient contract differs across grids/ranks")
                require(item["initial_audit"]["passed"], f"Independent initial oracle rejected {label}")
                baseline = report["reference_initial_audit"]
                floatingPosition = 128 * np.finfo(np.float64).eps * reference["header"]["box_mpc_h"]
                roundingBound = np.finfo(np.float32).eps if args.momentum_precision == "float32" else 0.
                inheritedLimits = {"position_rms_mpc_h": floatingPosition + GaussianCoefficientTolerance * baseline["displacement_rms_mpc_h"],
                                   "position_linf_mpc_h": floatingPosition + GaussianCoefficientTolerance * baseline["displacement_linf_mpc_h"],
                                   "momentum_relative_rms": GaussianCoefficientTolerance + roundingBound,
                                   "momentum_relative_linf": GaussianCoefficientTolerance + roundingBound}
                item["inherited_initial_checks"] = applyLimits(item["particle_metrics"], inheritedLimits)
                require(all(check["passed"] for check in item["inherited_initial_checks"].values()), f"Initial particle comparison rejected {label}")
            if explicitLimits:
                item["explicit_checks"] = applyLimits(item["particle_metrics"], explicitLimits["metrics"])
                require(all(check["passed"] for check in item["explicit_checks"].values()), f"Explicit metric budget rejected {label}")
            if args.power_source:
                item["power_comparison"] = comparePower(candidate, reference, args.grid, args.power_grid, args.power_source.resolve(strict=True))
            for descriptor in candidate["inputs"]:
                require(digest(descriptor["path"]) == descriptor["sha256"], "Candidate particles changed during validation")
            require(product(candidate["specification"]["path"]) == candidate["specification"], "Candidate manifest changed")
            item["state"] = "passed" if args.epoch == "initial" or explicitLimits else "measured_without_acceptance_budget"
            del candidate
        for descriptor in reference["inputs"]:
            require(digest(descriptor["path"]) == descriptor["sha256"], "Reference particles changed during validation")
        report["state"] = "passed" if args.epoch == "initial" or explicitLimits else "measured_without_acceptance_budget"
    except Exception as error:
        report["state"] = "failed"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        saveJson(args.output / "validation.json", report)
    print(json.dumps({"state": report["state"], "report": str(args.output / "validation.json")}))


## @brief Parse an analysis-only workflow; this command never starts a simulation.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--epoch", choices=("initial", "final"), default="initial")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cutoff", type=int, default=24)
    parser.add_argument("--momentum-precision", choices=("double", "float32"), default="float32")
    parser.add_argument("--max-particles", type=int, default=128**3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limits", type=Path)
    parser.add_argument("--power-source", type=Path)
    parser.add_argument("--power-grid", type=int, default=256)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
