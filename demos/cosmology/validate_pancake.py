#!/usr/bin/env python3
r"""Validate finite-amplitude planar collapse before shell crossing.

The production sine IC and production KDK/CIC/FFT evolution are compared by
particle ID with the exact continuum plane-symmetric growing solution.  The
two final deformation amplitudes, 0.5 and 0.8, have continuum minimum
Jacobians 0.5 and 0.2 (peak density contrasts 1 and 4).  This is not an
analytic oracle after shell crossing, and is not a general nonlinear-CDM test.

Full coverage: N=16,32,64; four timestep counts; three axis orientations and
one oblique orientation; MPI ranks 1--4.  Particle and mesh resolution remain
coupled.  The current IC has no phase parameter: arbitrary subcell translation
is tested by the separate frozen-force comparison, not claimed here.

Engineering error budgets and convergence gates are fixed below, before any
production run.  They are acceptance criteria, not assumed error estimates.
Inputs, logs, diagnostics, snapshots and executable/source hashes are retained
in a fresh results directory.  Dependencies: numpy and pandas.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

from validate_linear import (Case, Validation, fourier_modes, growth_reference,
                             phase_space_difference, self_test as growth_self_test)


Parameters = {"box_size": 168.75, "z_in": 49.0, "z_fi": 9.0,
              "hubble": 0.675, "Omega_m": 0.31, "Omega_bar": 0.0487,
              "Sigma_8": 0.82, "n_s": 0.965, "TFFlag": 4,
              "ic_mode": "sine", "mode_x": 1, "mode_y": 0, "mode_z": 0,
              "diagnostics_every": 1, "write_particles": 1}
Tolerances = {
    "initial_trajectory_relative": 1.0e-9,
    "net_momentum_relative": 1.0e-10,
    "displacement_dc_relative": 1.0e-10,
    "transverse_relative": 1.0e-9,
    "mpi_phase_space_relative": 1.0e-10,
    "position_roundoff_eps_box": 128.0,
    "time_difference_ratio": [2.8, 5.5],
    "mesh_error_ratio_minimum": 2.0,
    "n64_final_relative_error": {
        "0.5": {"displacement": 0.01, "momentum": 0.02},
        "0.8": {"displacement": 0.02, "momentum": 0.04},
    },
    "budget_scaling": "(64/N)^2 * squared integer mode norm",
    "relative_mass": 2.0e-12,
    "growth_reference_relative": 1.0e-8,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rms_vector(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(values*values, axis=1))))


def periodic_difference(left: np.ndarray, right: np.ndarray, box: float) -> np.ndarray:
    difference = left-right
    return difference-box*np.rint(difference/box)


def lattice(ids: np.ndarray, n: int, box: float) -> np.ndarray:
    """Cell-centred, x-fast global-ID contract; independent of rank ordering."""
    ids = np.asarray(ids)
    return (np.column_stack((ids % n, (ids//n) % n, ids//(n*n)))+0.5)*(box/n)


def initial_amplitude(finalAmplitude: float, parameters: dict) -> float:
    if not 0.0 < finalAmplitude < 1.0:
        raise ValueError("The analytic pancake oracle requires 0 < A_final < 1")
    ai = 1.0/(1.0+parameters["z_in"])
    af = 1.0/(1.0+parameters["z_fi"])
    return finalAmplitude*(growth_reference(ai, parameters["Omega_m"])[0]
                           / growth_reference(af, parameters["Omega_m"])[0])


def analytic_solution(ids: np.ndarray, parameters: dict, a: float) -> dict:
    """Exact plane-symmetric, pressureless solution, only while A(a)<1.

    x=q+s, s=-A(a) sin(k.q) k/k^2, p=a^2 E f s; tau=H0*t.
    Unlike a linear Eulerian-density comparison, this retains the nonlinear
    mass mapping and is exact at finite displacement before trajectories cross.
    """
    box, n, omega = parameters["box_size"], parameters["np"], parameters["Omega_m"]
    ai = 1.0/(1.0+parameters["z_in"])
    growth, rate = growth_reference(a, omega)
    amplitude = parameters["amplitude"]*growth/growth_reference(ai, omega)[0]
    if not 0.0 < amplitude < 1.0:
        raise ValueError("The analytic pancake oracle requires 0 < A(a) < 1; no post-crossing oracle")
    mode = np.asarray([parameters[f"mode_{axis}"] for axis in "xyz"], dtype=float)
    if not np.any(mode) or np.any(np.abs(mode) >= n/2):
        raise ValueError("A nonzero, strictly sub-Nyquist integer mode is required")
    wave = 2.0*np.pi*mode/box
    q = lattice(ids, n, box)
    displacement = -amplitude*np.sin(q@wave)[:, None]*wave/np.dot(wave, wave)
    expansion = math.sqrt(omega/a**3+1.0-omega)
    momentum = a*a*expansion*rate*displacement
    positions = (q+displacement) % box
    return {"q": q, "positions": positions, "displacement": displacement,
            "momentum": momentum, "amplitude": amplitude,
            "minimum_continuum_jacobian": 1.0-amplitude,
            "continuum_peak_density_contrast": amplitude/(1.0-amplitude),
            "wave": wave}


def continuum_density_modes(amplitude: float, harmonics=(1, 2, 3, 4)) -> np.ndarray:
    """Exact Eulerian 2*delta_hat_n = 2*J_n(n*A), by periodic q quadrature.

    The Lagrangian deformation amplitude A is NOT the Eulerian fundamental
    at finite amplitude: 2*J_1(A)=A-A^3/8+..., with generated higher harmonics.
    This diagnostic is not a second independent trajectory acceptance gate.
    """
    if not 0.0 <= amplitude < 1.0:
        raise ValueError("Density oracle requires pre-shell-crossing amplitude")
    phase = 2*np.pi*(np.arange(4096)+0.5)/4096
    mapped = phase-amplitude*np.sin(phase)
    return np.asarray([2*np.exp(-1j*int(harmonic)*mapped).mean() for harmonic in harmonics])


def trajectory_metrics(snapshot: pd.DataFrame, parameters: dict, a: float) -> dict:
    n, box = parameters["np"], parameters["box_size"]
    ids = snapshot["id"].to_numpy()
    if not np.array_equal(ids, np.arange(n**3, dtype=np.uint64)):
        raise ValueError("Snapshot IDs must be unique, complete and sorted")
    positions = snapshot[["x", "y", "z"]].to_numpy()
    momentum = snapshot[["px", "py", "pz"]].to_numpy()
    if (not np.isfinite(positions).all() or not np.isfinite(momentum).all()
            or (positions < 0).any() or (positions >= box).any()):
        raise ValueError("Snapshot must contain finite canonical phase space in the periodic box")
    reference = analytic_solution(ids, parameters, a)
    displacement = periodic_difference(positions, reference["q"], box)
    positionError = periodic_difference(positions, reference["positions"], box)
    momentumError = momentum-reference["momentum"]
    positionScale = rms_vector(reference["displacement"])
    momentumScale = rms_vector(reference["momentum"])
    unit = reference["wave"]/np.linalg.norm(reference["wave"])
    displacementTransverse = displacement-(displacement@unit)[:, None]*unit
    momentumTransverse = momentum-(momentum@unit)[:, None]*unit
    # The current suite uses one-dimensional axes and (1,1,0).  Under the
    # separately checked planar symmetry, this finite difference measures
    # neighbour ordering along a nonzero mode axis.  It is not a general
    # three-dimensional caustic detector.
    firstAxis = int(np.flatnonzero(unit)[0])
    modeComponent = parameters[f"mode_{'xyz'[firstAxis]}"]
    phaseDisplacement = (displacement@reference["wave"]).reshape(n, n, n)
    phaseStep = 2*np.pi*modeComponent/n
    jacobian = 1.0+(np.roll(phaseDisplacement, -1, axis=2-firstAxis)
                    -phaseDisplacement)/phaseStep
    return {
        "amplitude": reference["amplitude"],
        "minimum_continuum_jacobian": reference["minimum_continuum_jacobian"],
        "continuum_peak_density_contrast": reference["continuum_peak_density_contrast"],
        "minimum_sampled_planar_jacobian": float(jacobian.min()),
        "displacement_relative_error": rms_vector(positionError)/positionScale,
        "momentum_relative_error": rms_vector(momentumError)/momentumScale,
        "position_rms_error": rms_vector(positionError),
        "position_max_error": float(np.linalg.norm(positionError, axis=1).max()),
        "momentum_rms_error": rms_vector(momentumError),
        "momentum_max_error": float(np.linalg.norm(momentumError, axis=1).max()),
        "reference_displacement_rms": positionScale,
        "reference_momentum_rms": momentumScale,
        "transverse_displacement_relative": rms_vector(displacementTransverse)/positionScale,
        "transverse_momentum_relative": rms_vector(momentumTransverse)/momentumScale,
        "displacement_dc_relative": float(np.linalg.norm(displacement.mean(axis=0)))/positionScale,
        "net_momentum_relative": float(np.linalg.norm(momentum.mean(axis=0)))/momentumScale,
    }


def error_budgets(n: int, finalAmplitude: float, mode: tuple[int, int, int]) -> dict:
    base = Tolerances["n64_final_relative_error"][f"{finalAmplitude:g}"]
    factor = (64.0/n)**2*sum(component*component for component in mode)
    return {name: factor*value for name, value in base.items()}


class PancakeValidation(Validation):
    """Reuse only the launcher/integrity/background helpers, not linear gates."""
    def __init__(self, arguments: argparse.Namespace):
        if arguments.work_dir is None:
            arguments.work_dir = Path(tempfile.mkdtemp(prefix="pancake-validation-", dir=Path.cwd()))
        super().__init__(arguments)
        sourceDirectory = Path(__file__).resolve().parent
        sourcePaths = [sourceDirectory/name for name in (
            "Cosmology.cpp", "CosmologyConfig.h", "CosmologyPhysics.h", "CosmologySimulation.h",
            "validate_linear.py", "validate_pancake.py", "tests/test_validate_pancake.py")]
        self.provenancePaths = [arguments.exe.resolve(), *sourcePaths]
        self.hashes = {str(path): sha256(path) for path in self.provenancePaths}
        self.metrics: dict[str, dict] = {}
        self.results.update({
            "schema": "ippl-pancake-validation-v1", "tolerances": Tolerances,
            "hashes_before": self.hashes,
            "protocol": {
                "parameters": Parameters, "final_amplitudes": [0.5, 0.8],
                "mesh_sizes": [16, 32, 64], "mesh_study_steps": 128,
                "time_study_steps": [16, 32, 64, 128], "ranks": [1, 2, 3, 4],
                "orientations": [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]],
                "oracle": "Exact continuum planar growing solution before shell crossing",
                "time_study": "Successive solution differences at fixed N=32, A_final=.5; expected second order",
                "mesh_study": "Continuum trajectory errors at fixed nt=128; expect O(h^2), require each halving to reduce error by at least 2",
                "budget_status": "Predeclared engineering acceptance budgets, not assumed accuracy",
                "limitations": ["No post-shell-crossing analytic oracle", "No random-CDM nonlinear qualification",
                                "Particle and force-mesh resolutions are coupled", "Only cell-centred IC phase",
                                "CPU MPI ranks 1--4, one OpenMP thread", "Quick mode omits convergence studies"],
            },
            "trajectory_metrics": self.metrics,
        })
        self.save()

    def run_pancake(self, name: str, *, n=32, steps=128, finalAmplitude=0.5,
                     mode=(1, 0, 0), ranks=1) -> Case | None:
        parameters = {**Parameters, "np": n, "nt": steps,
                      **{f"mode_{axis}": value for axis, value in zip("xyz", mode)}}
        parameters["amplitude"] = initial_amplitude(finalAmplitude, parameters)
        case = self.run(name, ranks=ranks, **parameters)
        if case is None:
            return None
        try:
            metrics = {}
            for epoch, redshift in (("initial", parameters["z_in"]), ("final", parameters["z_fi"])):
                values = trajectory_metrics(case.snapshot(epoch), parameters, 1.0/(1.0+redshift))
                metrics[epoch] = values
                for key, tolerance in (("transverse_displacement_relative", Tolerances["transverse_relative"]),
                                       ("transverse_momentum_relative", Tolerances["transverse_relative"]),
                                       ("displacement_dc_relative", Tolerances["displacement_dc_relative"]),
                                       ("net_momentum_relative", Tolerances["net_momentum_relative"])):
                    self.check(f"{name}: {epoch} {key}", values[key] <= tolerance,
                               error=values[key], tolerance=tolerance)
                self.check(f"{name}: {epoch} sampled planar ordering",
                           values["minimum_sampled_planar_jacobian"] > 0,
                           minimum_sampled_jacobian=values["minimum_sampled_planar_jacobian"],
                           minimum_continuum_jacobian=values["minimum_continuum_jacobian"])
                if epoch == "initial":
                    for quantity in ("displacement", "momentum"):
                        error = values[f"{quantity}_relative_error"]
                        self.check(f"{name}: exact initial {quantity}",
                                   error <= Tolerances["initial_trajectory_relative"],
                                   relative_error=error, tolerance=Tolerances["initial_trajectory_relative"])
            budgets = error_budgets(n, finalAmplitude, mode)
            for quantity in ("displacement", "momentum"):
                error = metrics["final"][f"{quantity}_relative_error"]
                self.check(f"{name}: final analytic {quantity}", error <= budgets[quantity],
                           relative_error=error, tolerance=budgets[quantity])
            modes = np.asarray(mode)[None, :]*np.arange(1, 5)[:, None]
            measured = fourier_modes(case.snapshot("final"), parameters["box_size"], modes)
            expected = continuum_density_modes(finalAmplitude)
            metrics["eulerian_density_harmonics"] = {
                "note": "Diagnostic only: 2*delta_hat_n, not the Lagrangian deformation amplitude",
                "harmonics": [1, 2, 3, 4], "measured_real": measured.real.tolist(),
                "measured_imag": measured.imag.tolist(), "continuum_real": expected.real.tolist(),
            }
            self.metrics[name] = metrics
            self.save()
            return case
        except (ValueError, OSError) as error:
            self.check(name+": trajectory analysis", False, error=str(error))
            return None

    def compare_ranks(self, reference: Case | None, other: Case | None) -> None:
        if reference is None or other is None:
            self.check("MPI comparison inputs available", False)
            return
        for epoch in ("initial", "final"):
            position, momentum = phase_space_difference(reference.snapshot(epoch), other.snapshot(epoch),
                                                       reference.parameters["box_size"])
            # Shared helper reports per-component RMS; use vector RMS here.
            position *= math.sqrt(3)
            momentum *= math.sqrt(3)
            values = self.metrics[reference.name][epoch]
            positionTolerance = max(Tolerances["mpi_phase_space_relative"]*values["reference_displacement_rms"],
                                    Tolerances["position_roundoff_eps_box"]*np.finfo(float).eps
                                    *reference.parameters["box_size"])
            momentumTolerance = (Tolerances["mpi_phase_space_relative"]
                                 *values["reference_momentum_rms"])
            self.check(f"{other.name}: {epoch} MPI phase space", bool(
                position <= positionTolerance and momentum <= momentumTolerance),
                position_rms_difference=position, momentum_rms_difference=momentum,
                position_tolerance=positionTolerance, momentum_tolerance=momentumTolerance)

    def check_mesh(self, cases: list[Case | None], finalAmplitude: float) -> None:
        if any(case is None for case in cases):
            self.check(f"A={finalAmplitude}: mesh inputs available", False)
            return
        for coarse, fine in zip(cases[:-1], cases[1:]):
            for quantity in ("displacement", "momentum"):
                coarseError = self.metrics[coarse.name]["final"][f"{quantity}_relative_error"]
                fineError = self.metrics[fine.name]["final"][f"{quantity}_relative_error"]
                ratio = coarseError/fineError if fineError > 0 else math.inf
                self.check(f"A={finalAmplitude}: mesh {quantity} N{coarse.parameters['np']} to N{fine.parameters['np']}",
                           math.isfinite(ratio) and ratio >= Tolerances["mesh_error_ratio_minimum"],
                           coarse_relative_error=coarseError, fine_relative_error=fineError,
                           error_ratio=ratio, measured_order=math.log2(ratio) if ratio > 0 else None,
                           minimum_ratio=Tolerances["mesh_error_ratio_minimum"], expected_asymptotic_order=2)

    def check_time(self, cases: list[Case | None]) -> None:
        if any(case is None for case in cases):
            self.check("time convergence inputs available", False)
            return
        differences = [phase_space_difference(coarse.snapshot("final"), fine.snapshot("final"),
                                             coarse.parameters["box_size"])
                       for coarse, fine in zip(cases[:-1], cases[1:])]
        for index in range(len(differences)-1):
            for component, quantity in enumerate(("position", "momentum")):
                coarse, fine = differences[index][component], differences[index+1][component]
                ratio = coarse/fine if fine > 0 else math.inf
                lower, upper = Tolerances["time_difference_ratio"]
                self.check(f"time {quantity}: nt{cases[index].parameters['nt']}/{cases[index+1].parameters['nt']}/{cases[index+2].parameters['nt']}",
                           math.isfinite(ratio) and lower <= ratio <= upper,
                           coarse_per_component_rms_difference=coarse,
                           fine_per_component_rms_difference=fine,
                           difference_ratio=ratio, measured_order=math.log2(ratio) if ratio > 0 else None,
                           allowed_ratio=[lower, upper], expected_asymptotic_order=2)

    def run_suite(self) -> bool:
        baseline = {}
        for amplitude in (0.5, 0.8):
            cases = []
            for n in ((32,) if self.arguments.quick else (16, 32, 64)):
                case = self.run_pancake(f"axis_a{int(10*amplitude)}_n{n}", n=n, finalAmplitude=amplitude)
                cases.append(case)
                if n == 32:
                    baseline[amplitude] = case
            if not self.arguments.quick:
                self.check_mesh(cases, amplitude)
        for ranks in (2, 3, 4):
            other = self.run_pancake(f"axis_a8_r{ranks}", finalAmplitude=0.8, ranks=ranks)
            self.compare_ranks(baseline[0.8], other)
        for label, mode in (("y", (0, 1, 0)), ("z", (0, 0, 1)), ("xy", (1, 1, 0))):
            self.run_pancake(f"orientation_{label}", mode=mode)
        if not self.arguments.quick:
            cases = [self.run_pancake(f"time_nt{steps}", steps=steps) for steps in (16, 32, 64)]
            self.check_time([*cases, baseline[0.5]])
        hashesAfter = {str(path): sha256(path) for path in self.provenancePaths}
        self.results["hashes_after"] = hashesAfter
        self.check("executable and analysis/physics sources unchanged during campaign", hashesAfter == self.hashes)
        self.results["passed"] = bool(self.results["checks"]) and all(check["passed"] for check in self.results["checks"])
        self.results["finished_utc"] = datetime.now(timezone.utc).isoformat()
        self.results["qualification"] = ("Pre-shell-crossing finite-amplitude planar trajectories, MPI1-4; "
                                         + ("convergence NOT checked (quick mode)" if self.arguments.quick
                                            else "mesh and timestep convergence checked"))
        self.save()
        print(f"\n{'PASS' if self.results['passed'] else 'FAIL'}: {len(self.results['runs'])} runs, "
              f"{len(self.results['checks'])} checks; {self.resultPath}", flush=True)
        return self.results["passed"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", required=True, type=Path, help="Production Cosmology executable")
    parser.add_argument("--work-dir", type=Path, help="Fresh evidence directory; default unique directory under cwd")
    parser.add_argument("--mpiexec", default="mpirun")
    parser.add_argument("--numproc-flag", default="-n")
    parser.add_argument("--mpi-arg", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=900.0, help="Seconds per MPI simulation")
    parser.add_argument("--quick", action="store_true", help="Omit convergence studies; retain amplitudes, orientations, ranks1--4")
    arguments = parser.parse_args()
    if not arguments.exe.is_file() or not os.access(arguments.exe, os.X_OK):
        parser.error(f"Executable missing or not executable: {arguments.exe}")
    if not math.isfinite(arguments.timeout) or arguments.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    growth_self_test()
    try:
        return 0 if PancakeValidation(arguments).run_suite() else 1
    except (OSError, ValueError) as error:
        print(f"Pancake validation setup/data error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
