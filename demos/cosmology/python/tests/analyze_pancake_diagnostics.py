#!/usr/bin/env python3
## @file analyze_pancake_diagnostics.py
# @brief Read-only follow-up of saved pancake mass and local-gradient diagnostics.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
# @see cosmology_model cosmology_numerics cosmology_spectra
# Tests assert declared invariants using isolated fixtures; no scientific tolerances are relaxed by documentation.
"""Read-only follow-up of saved pancake mass and local-gradient diagnostics.

This does not rerun simulations, modify the campaign or override failed gates.
It redeposits initial/final snapshots independently and compares summation
methods.  The maximum-error intermediate epoch is NOT reconstructed, because
the production campaign saved only endpoint particle snapshots.

Example:
    python analyze_pancake_diagnostics.py --campaign path/to/pancake-validation \
        --output-dir path/to/new-pancake-diagnostics
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_pancake import analytic_solution, periodic_difference, rms_vector, sha256


## @brief Read and verify snapshot.
# @see cosmology_tools
#
# @param directory Artifact directory following this module's ownership/freshness contract.
# @param epoch Declared synchronized output scale factor or checkpoint label.
# @param ranks Positive MPI rank count; all expected snapshot shards must exist.
# @param count Expected number of particles/elements, or byte-count context explicitly declared by the routine.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def read_snapshot(directory: Path, epoch: str, ranks: int, count: int) -> tuple[pd.DataFrame, list[Path]]:
    paths = [directory/f"particles_{epoch}_rank{rank}.csv" for rank in range(ranks)]
    frames = [pd.read_csv(path, dtype={"id": np.uint64}) for path in paths]
    frame = pd.concat(frames, ignore_index=True).sort_values("id").reset_index(drop=True)
    if not np.array_equal(frame.id.to_numpy(), np.arange(count, dtype=np.uint64)):
        raise ValueError("Incomplete, duplicated or unsorted particle IDs")
    if not np.isfinite(frame[["x", "y", "z", "px", "py", "pz"]].to_numpy()).all():
        raise ValueError("Nonfinite saved phase space")
    return frame, paths


## @brief Independent unit-mass, cell-centred periodic CIC with NumPy add.at.
# @see cosmology_tools
#
# @param positions Finite particle position array of shape (Nparticles,3), in comoving Mpc/h.
# @param n Mesh/lattice size per Cartesian dimension in this routine's integer-grid convention.
# @param box Positive periodic comoving box side in Mpc/h.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def redeposition_sums(positions: np.ndarray, n: int, box: float) -> dict:
    """Independent unit-mass, cell-centred periodic CIC with NumPy add.at."""
    total = n**3
    coordinate = positions/(box/n)-0.5
    lower = np.floor(coordinate).astype(np.int64)
    fraction = coordinate-lower
    density = np.zeros(total)
    weightSum = np.zeros(len(positions))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                shift = np.asarray([dx, dy, dz])
                index = (lower+shift) % n
                weight = np.prod(np.where(shift[None, :], fraction, 1-fraction), axis=1)
                np.add.at(density, index[:, 0]+n*(index[:, 1]+n*index[:, 2]), weight)
                weightSum += weight
    if not np.isfinite(density).all() or (density < 0).any():
        raise ValueError("Invalid independently deposited mesh mass")
    # These deliberately different accumulation orders expose sensitivity of
    # the scalar diagnostic, without changing any deposited cell value.
    totals = {"numpy_pairwise": float(density.sum()), "math_fsum": math.fsum(density),
              "numpy_longdouble": float(density.sum(dtype=np.longdouble)),
              "naive_serial_xfast": float(np.cumsum(density)[-1]),
              "naive_serial_zfast": float(np.cumsum(density.reshape(n, n, n)
                                                    .transpose(2, 1, 0).ravel())[-1])}
    # Nonnegative-sum forward-error bound, not a substitute acceptance budget.
    unitRoundoff = np.finfo(float).eps/2
    gamma = (total-1)*unitRoundoff/(1-(total-1)*unitRoundoff)
    return {"expected_total_mass": total, "independent_total_mass": totals,
            "independent_relative_errors": {name: value/total-1 for name, value in totals.items()},
            "maximum_individual_particle_weight_sum_error": float(abs(weightSum-1).max()),
            "numpy_longdouble_epsilon": float(np.finfo(np.longdouble).eps),
            "binary64_unit_roundoff": unitRoundoff,
            "naive_nonnegative_sum_gamma_n_bound": gamma,
            "bound_is_not_an_acceptance_tolerance": True}


## @brief Axis-x finite-difference Jacobian and L-infinity trajectory errors.
# @see cosmology_tools
#
# @param snapshot ID-labelled retained particle state; its epoch and units follow the originating run metadata.
# @param parameters Named protocol or cosmology parameters; unsupported keys are rejected by the calling validator.
# @param a Dimensionless scale factor; must satisfy the calling background/epoch contract.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def local_profile(snapshot: pd.DataFrame, parameters: dict, a: float) -> dict:
    """Axis-x finite-difference Jacobian and L-infinity trajectory errors.

    Compare the sampled numerical Jacobian to the SAME finite difference of
    the analytic particle map.  This removes the trivial differencing bias
    from the reported local-gradient error.
    """
    if [parameters[f"mode_{axis}"] for axis in "xyz"] != [1, 0, 0]:
        raise ValueError("The local profile currently supports the x-axis fundamental only")
    n, box = parameters["np"], parameters["box_size"]
    reference = analytic_solution(snapshot.id.to_numpy(), parameters, a)
    positions = snapshot[["x", "y", "z"]].to_numpy()
    momentum = snapshot[["px", "py", "pz"]].to_numpy()
    displacement = periodic_difference(positions, reference["q"], box)
    positionError = periodic_difference(positions, reference["positions"], box)
    momentumError = momentum-reference["momentum"]
    profile = displacement[:, 0].reshape(n, n, n)
    # The production axis solution should be independent of transverse cells;
    # retain the deviation as evidence rather than silently averaging it away.
    transverseVariation = float(abs(profile-profile[0, 0][None, None, :]).max())
    profile = profile[0, 0]
    exactProfile = reference["displacement"][:n, 0]
    jacobian = 1+(np.roll(profile, -1)-profile)/(box/n)
    exactJacobian = 1+(np.roll(exactProfile, -1)-exactProfile)/(box/n)
    error = jacobian-exactJacobian
    positionMax = float(np.linalg.norm(positionError, axis=1).max())
    momentumMax = float(np.linalg.norm(momentumError, axis=1).max())
    amplitude = reference["amplitude"]
    waveLengthScale = np.linalg.norm(reference["wave"])
    continuumPositionPeak = amplitude/waveLengthScale
    continuumMomentumPeak = math.sqrt(2)*rms_vector(reference["momentum"])
    return {
        "n": n, "deformation_amplitude": amplitude,
        "minimum_sampled_jacobian": float(jacobian.min()),
        "minimum_exact_sampled_jacobian": float(exactJacobian.min()),
        "minimum_continuum_jacobian": 1-amplitude,
        "jacobian_max_absolute_error": float(abs(error).max()),
        "jacobian_rms_error": float(np.sqrt(np.mean(error*error))),
        "cell_count_abs_jacobian_error_above_0_05": int(np.count_nonzero(abs(error) > .05)),
        "cell_fraction_abs_jacobian_error_above_0_05": float(np.mean(abs(error) > .05)),
        "threshold_0_05_is_descriptive_not_a_gate": True,
        "transverse_profile_max_variation": transverseVariation,
        "position_max_absolute_error": positionMax,
        "momentum_max_absolute_error": momentumMax,
        "position_max_relative_to_continuum_peak_displacement": positionMax/continuumPositionPeak,
        "momentum_max_relative_to_continuum_peak_momentum": momentumMax/continuumMomentumPeak,
        "lagrangian_q_over_L": ((np.arange(n)+.5)/n).tolist(),
        "lagrangian_interval_midpoint_over_L": ((np.arange(n)+1)/n).tolist(),
        "numerical_displacement_x": profile.tolist(),
        "exact_displacement_x": exactProfile.tolist(),
        "numerical_jacobian": jacobian.tolist(), "exact_sampled_jacobian": exactJacobian.tolist(),
        "jacobian_error": error.tolist(),
    }


## @brief Analyze the documented module workflow.
# @see cosmology_tools
#
# @param campaign Retained campaign object/report with declared configuration, source hashes and run states.
# @param outputDirectory Fresh destination for retained analysis/figure artifacts.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def analyze(campaign: Path, outputDirectory: Path) -> Path:
    campaign = campaign.resolve()
    resultPath = campaign/"results.json"
    results = json.loads(resultPath.read_text())
    if results.get("schema") != "ippl-pancake-validation-v1":
        raise ValueError("Expected a pancake-validation-v1 campaign")
    outputDirectory = outputDirectory.resolve()
    outputDirectory.mkdir(parents=True, exist_ok=False)
    inputs = {str(resultPath): sha256(resultPath)}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema": "ippl-pancake-diagnostic-audit-v1", "campaign": str(campaign),
        "original_campaign_passed": results["passed"],
        "original_failed_gates": [check for check in results["checks"] if not check["passed"]],
        "does_not_override_campaign_gates": True,
        "analysis_source_sha256": sha256(Path(__file__).resolve()),
        "analysis_helper_sha256": sha256(Path(__file__).resolve().parents[1]/"validate_pancake.py"),
        "analysis_growth_helper_sha256": sha256(Path(__file__).resolve().parents[1]/"validate_linear.py"),
        "mass_diagnostics": {}, "local_profiles": {},
        "interpretation_limits": [
            "Independent endpoint redeposition cannot prove the mass error at an unsaved intermediate epoch",
            "math.fsum accurately sums the independently deposited binary64 cells; not exact-real arithmetic",
            "The naive-summation forward-error bound is explanatory, not a replacement acceptance gate",
            "A finite positive sampled planar Jacobian is not a general 3D caustic test",
            "Convergent trajectory L2 errors do not establish local-density or gradient convergence",
            "Fixed-cell-width gradient defects are consistent with grid/lattice locking, not proof of its cause",
            "Separate particle/mesh resolution and IC-phase studies are needed to establish the origin",
        ],
    }
    runs = {run["name"]: run for run in results["runs"]}
    for amplitude in (5, 8):
        for n in (16, 32, 64):
            name = f"axis_a{amplitude}_n{n}"
            if name not in runs:
                continue
            run = runs[name]
            parameters = run["parameters"]
            directory = campaign/name/"output"
            frame, paths = read_snapshot(directory, "final", run["ranks"], n**3)
            inputs.update({str(path): sha256(path) for path in paths})
            summary["local_profiles"][name] = local_profile(frame, parameters, 1/(1+parameters["z_fi"]))
            if n != 64:
                continue
            diagnosticPath = directory/"diagnostics.csv"
            inputs[str(diagnosticPath)] = sha256(diagnosticPath)
            diagnostic = pd.read_csv(diagnosticPath)
            peak = diagnostic.loc[diagnostic.mass_error.idxmax()]
            evidence = {"maximum_reported_mass_error": float(peak.mass_error),
                        "maximum_error_step": int(peak.step), "maximum_error_scale_factor": float(peak.a),
                        "maximum_reported_absolute_mass_residual": float(peak.mass_error*n**3),
                        "peak_epoch_particles_available": False, "endpoints": {}}
            for epoch in ("initial", "final"):
                snapshot, paths = read_snapshot(directory, epoch, run["ranks"], n**3)
                inputs.update({str(path): sha256(path) for path in paths})
                values = redeposition_sums(snapshot[["x", "y", "z"]].to_numpy(), n, parameters["box_size"])
                values["reported_mass_error"] = float(diagnostic.mass_error.iloc[0 if epoch == "initial" else -1])
                evidence["endpoints"][epoch] = values
            summary["mass_diagnostics"][name] = evidence
    summary["input_hashes_before"] = inputs
    summary["input_hashes_after"] = {path: sha256(Path(path)) for path in inputs}
    summary["inputs_unchanged"] = summary["input_hashes_after"] == inputs
    if not summary["inputs_unchanged"]:
        raise RuntimeError("Campaign evidence changed while it was being analyzed")
    outputPath = outputDirectory/"diagnostics.json"
    outputPath.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return outputPath


## @brief Parse the documented command-line interface and execute its selected workflow, retaining nonzero failures.
# @see cosmology_tools
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory; never overwrite saved evidence")
    arguments = parser.parse_args()
    try:
        print(analyze(arguments.campaign, arguments.output_dir))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Pancake diagnostic audit failed: {error}", file=sys.stderr)
        return 2


## @cond CLI_DISPATCH
if __name__ == "__main__":
    sys.exit(main())
## @endcond
