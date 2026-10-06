#!/usr/bin/env python3
r"""Reproducible CPU/MPI validation of the IPPL linear cosmology demonstration.

Example (use a new results directory for each invocation)::

    python validate_linear.py --exe build/demos/cosmology/Cosmology \
        --work-dir /tmp/cosmology-validation

The tests measure particle Fourier modes independently of mesh diagnostics.
Flat-LCDM growing modes are obtained by quadrature of the Heath integral,
without calling the implementation under test. Numerical limits are declared
before a simulation is run. Failed checks produce a nonzero exit status and
remain recorded in results.json with inputs, launcher and logs for every run.

The full suite exercises ranks 1, 2, 3 and 4; two OpenMP thread counts; axis
and oblique modes; Gaussian ICs; time convergence; and mesh convergence.
--quick retains all MPI ranks but omits the convergence studies and uses a
smaller Gaussian mesh. Dependencies: numpy and pandas.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from runtime_metadata import validate_runtime_metadata


# CIC scatter and gather approximately filter the continuum force by
# product_j sinc(k_j h/2)^4. These predeclared acceptance limits allow
# O((kh)^2) PM error while rejecting wrong kicks, growth or velocity units.
# The separate mesh study measures convergence of that discretization error.
AxisGrowthTolerance = 0.015
ObliqueGrowthTolerance = 0.035
GaussianGrowthTolerance = 0.05
QuickGaussianGrowthTolerance = 0.12
MassTolerance = 2.0e-12
ReproducibilityTolerance = 1.0e-10
GrowthReferenceTolerance = 1.0e-8
TimeConvergenceMinimum = 2.8
TimeConvergenceMaximum = 5.5
MeshConvergenceMinimum = 2.4
GaussianCoefficientTolerance = 1.0e-8
SnapshotColumns = ("id", "x", "y", "z", "px", "py", "pz")
DiagnosticColumns = (
    "step", "a", "D", "f", "particles", "mass_error", "delta_rms",
    "mode_re", "mode_im", "mode_amplitude", "expected_amplitude",
    "displacement_rms", "momentum_rms", "force_rms",
)
QuadratureNodes, QuadratureWeights = np.polynomial.legendre.leggauss(192)


def growth_reference(a: float, omegaMatter: float) -> tuple[float, float]:
    """Return independent D(a), normalized to D(1)=1, and f=dlnD/dlna.

    D_raw = 5 Omega_m E(a)/2 integral_0^a [x E(x)]^-3 dx.
    Writing the integrand as x^(3/2)/(Omega_m+Omega_L x^3)^(3/2)
    avoids cancellation and the apparent singularity at x=0.
    """
    if not 0.0 < a or not 0.0 < omegaMatter <= 1.0:
        raise ValueError("Growth reference requires a>0 and 0<Omega_m<=1")
    omegaLambda = 1.0 - omegaMatter

    def integral(scale: float) -> float:
        nodes = scale * (QuadratureNodes + 1.0) / 2.0
        integrand = nodes**1.5 / (omegaMatter + omegaLambda * nodes**3)**1.5
        return float(scale / 2.0 * np.dot(QuadratureWeights, integrand))

    integralA = integral(a)
    expansion = math.sqrt(omegaMatter / a**3 + omegaLambda)
    growth = expansion * integralA / integral(1.0)
    omegaAtA = omegaMatter / (omegaMatter + omegaLambda * a**3)
    rate = -1.5 * omegaAtA + a**2.5 / (
        (omegaMatter + omegaLambda * a**3)**1.5 * integralA
    )
    return growth, rate


def fourier_modes(snapshot: pd.DataFrame, boxSize: float,
                  modes: np.ndarray) -> np.ndarray:
    """2 <exp(-i k.x)> from particles, independent of deposition and FFTs."""
    positions = snapshot.loc[:, ["x", "y", "z"]].to_numpy()
    coefficients = []
    # One mode at a time bounds memory for the 64^3 convergence case.
    for mode in modes:
        phases = (2.0 * math.pi / boxSize) * (positions @ mode)
        coefficients.append(2.0 * np.exp(-1j * phases).mean())
    return np.asarray(coefficients)


def bbks_spectrum(waveNumber: np.ndarray, parameters: dict) -> np.ndarray:
    """Independently normalize BBKS P(k) to sigma8 (Zarija cutoff k=10).

    Composite Gauss-Legendre in log k is independent of the C++ Simpson
    quadrature. The missing k<1e-8 integral is negligible (<1e-20 relative).
    """
    def transfer(k):
        q = k / (parameters["Omega_m"] * parameters["hubble"])
        return (np.log1p(2.34 * q) / (2.34 * q)
                / (1.0 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**0.25)

    nodes, weights = np.polynomial.legendre.leggauss(32)
    edges = np.linspace(math.log(1.0e-8), math.log(10.0), 65)
    halfWidths = np.diff(edges) / 2.0
    logK = ((edges[:-1] + edges[1:])[:, None] / 2.0
            + halfWidths[:, None] * nodes[None, :])
    k = np.exp(logK)
    x = 8.0 * k
    small = x < 0.01
    window = np.empty_like(x)
    window[small] = 1.0 - x[small]**2/10.0 + x[small]**4/280.0 - x[small]**6/15120.0
    window[~small] = 3.0 * (np.sin(x[~small]) - x[~small]*np.cos(x[~small])) / x[~small]**3
    integrand = k**(3.0 + parameters["n_s"]) * transfer(k)**2 * window**2
    variance = float(np.sum(integrand * weights[None, :] * halfWidths[:, None])
                     / (2.0 * math.pi**2))
    return (parameters["Sigma_8"]**2 / variance
            * waveNumber**parameters["n_s"] * transfer(waveNumber)**2)


def gaussian_coefficients(n: int, seed: int, power: np.ndarray,
                          boxSize: float) -> np.ndarray:
    """Scalar integer specification of the documented Fourier-pair RNG.

    This only defines the seed/phase contract. The physical normalization
    supplied in power is independently integrated above; recovered particle
    modes test the transforms, global indexing and displacement convention.
    """
    mask64 = (1 << 64) - 1

    def mix(value: int) -> int:
        value = (value + 0x9e3779b97f4a7c15) & mask64
        value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & mask64
        value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & mask64
        return value ^ (value >> 31)

    def uniform(value: int) -> float:
        return ((mix(value) >> 12) + 0.5) / float(1 << 52)

    coefficients = np.zeros((n, n, n), dtype=np.complex128)
    mixedSeed = mix(seed)
    for gz in range(n):
        for gy in range(n):
            for gx in range(n):
                if n//2 in (gx, gy, gz) or (gx == gy == gz == 0):
                    continue
                key = (gx*n + gy)*n + gz
                other = (((n-gx) % n)*n + (n-gy) % n)*n + (n-gz) % n
                canonical = min(key, other)
                u1 = uniform(mixedSeed ^ (2*canonical))
                u2 = uniform(mixedSeed ^ (2*canonical + 1))
                radius = math.sqrt(-math.log(u1) * power[gz, gy, gx] / boxSize**3)
                angle = 2.0 * math.pi * u2
                delta = radius * complex(math.cos(angle), math.sin(angle) * (1 if key <= other else -1))
                wave = [g if g <= n//2 else g-n for g in (gx, gy, gz)]
                phase = math.pi * sum(wave) / n
                coefficients[gz, gy, gx] = delta * complex(math.cos(phase), math.sin(phase))
    return coefficients


def phase_space_difference(left: pd.DataFrame, right: pd.DataFrame,
                           boxSize: float) -> tuple[float, float]:
    """RMS periodic position and canonical momentum differences by ID."""
    if not np.array_equal(left["id"].to_numpy(), right["id"].to_numpy()):
        raise ValueError("Cannot compare snapshots with different particle IDs")
    positionDelta = (left[["x", "y", "z"]].to_numpy()
                     - right[["x", "y", "z"]].to_numpy())
    positionDelta -= boxSize * np.rint(positionDelta / boxSize)
    momentumDelta = (left[["px", "py", "pz"]].to_numpy()
                     - right[["px", "py", "pz"]].to_numpy())
    return (float(np.sqrt(np.mean(positionDelta**2))),
            float(np.sqrt(np.mean(momentumDelta**2))))


@dataclass
class Case:
    name: str
    ranks: int
    threads: int
    parameters: dict
    directory: Path
    diagnostics: pd.DataFrame | None = None
    snapshots: dict[str, pd.DataFrame] = field(default_factory=dict)

    def snapshot(self, epoch: str) -> pd.DataFrame:
        if epoch not in self.snapshots:
            paths = sorted(self.directory.glob(f"particles_{epoch}_rank*.csv"))
            expected = {
                self.directory / f"particles_{epoch}_rank{rank}.csv"
                for rank in range(self.ranks)
            }
            if set(paths) != expected:
                raise ValueError(f"{self.name}: incomplete {epoch} snapshot: {paths}")
            frames = [pd.read_csv(path, dtype={"id": np.uint64}) for path in paths]
            data = pd.concat(frames, ignore_index=True)
            if not set(SnapshotColumns).issubset(data.columns):
                raise ValueError(f"{self.name}: missing columns in {epoch} snapshot")
            self.snapshots[epoch] = data.sort_values("id").reset_index(drop=True)
        return self.snapshots[epoch]


class Validation:
    def __init__(self, arguments: argparse.Namespace):
        self.arguments = arguments
        self.directory = (arguments.work_dir.resolve() if arguments.work_dir is not None
                          else Path(tempfile.mkdtemp(prefix="cosmology-validation-", dir=Path.cwd())))
        self.directory.mkdir(parents=True, exist_ok=True)
        self.resultPath = self.directory / "results.json"
        if self.resultPath.exists():
            raise ValueError("The work directory contains results.json; use a new directory")
        self.results = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "executable": str(arguments.exe.resolve()),
            "command": sys.argv,
            "quick": arguments.quick,
            "execution_scope": "MPI/backend/host-thread configuration; the two-host-thread check is not GPU scaling",
            "tolerances": {
                "axis_growth": AxisGrowthTolerance,
                "oblique_growth": ObliqueGrowthTolerance,
                "gaussian_growth": GaussianGrowthTolerance,
                "quick_gaussian_growth": QuickGaussianGrowthTolerance,
                "relative_mass": MassTolerance,
                "mpi_and_thread_relative": ReproducibilityTolerance,
                "growth_reference_relative": GrowthReferenceTolerance,
                "time_difference_ratio": [TimeConvergenceMinimum, TimeConvergenceMaximum],
                "mesh_error_ratio_minimum": MeshConvergenceMinimum,
                "gaussian_coefficient_relative": GaussianCoefficientTolerance,
            },
            "runs": [],
            "checks": [],
        }
        self.cases: dict[str, Case] = {}
        print(f"Validation results: {self.resultPath}", flush=True)

    def save(self) -> None:
        self.resultPath.write_text(json.dumps(self.results, indent=2, allow_nan=False) + "\n")

    def check(self, name: str, passed: bool, **details) -> None:
        def serializable(value):
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                return str(value)
            if isinstance(value, dict):
                return {key: serializable(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [serializable(item) for item in value]
            return value

        self.results["checks"].append({
            "name": name, "passed": bool(passed), **serializable(details),
        })
        print(f"{'PASS' if passed else 'FAIL'} {name}: {details}", flush=True)
        self.save()

    def run(self, name: str, *, ranks: int = 1, threads: int = 1,
            **overrides) -> Case | None:
        parameters = {
            "np": 32, "nt": 32, "box_size": 200.0, "seed": 73452342811,
            "z_in": 49.0, "z_fi": 24.0, "hubble": 0.675,
            "Omega_m": 0.31, "Omega_bar": 0.0487, "Sigma_8": 0.02,
            "n_s": 0.965, "TFFlag": 4, "ic_mode": "sine",
            "amplitude": 0.001, "mode_x": 1, "mode_y": 0, "mode_z": 0,
            "diagnostics_every": 1, "write_particles": 1,
        }
        parameters.update(overrides)
        directory = self.directory / name
        directory.mkdir(exist_ok=False)
        outputDirectory = directory / "output"
        parameters["output"] = str(outputDirectory)
        inputPath = directory / "input.par"
        inputPath.write_text("".join(f"{key}={value}\n" for key, value in parameters.items()))
        command = (shlex.split(self.arguments.mpiexec) + self.arguments.mpi_arg
                   + [self.arguments.numproc_flag, str(ranks),
                      str(self.arguments.exe.resolve()), str(inputPath)])
        environment = os.environ.copy()
        environment.update({
            "OMP_NUM_THREADS": str(threads), "OMP_PROC_BIND": "false",
            "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        })
        runResult = {
            "name": name, "ranks": ranks, "threads": threads,
            "parameters": parameters, "command": command,
            "environment": {key: environment[key] for key in (
                "OMP_NUM_THREADS", "OMP_PROC_BIND", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        }
        self.results["runs"].append(runResult)
        print(f"RUN  {name}: ranks={ranks}, threads={threads}, N={parameters['np']}, "
              f"steps={parameters['nt']}", flush=True)
        started = time.monotonic()
        try:
            with (directory / "run.log").open("w") as log:
                process = subprocess.Popen(command, cwd=directory, env=environment,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    returnCode = process.wait(timeout=self.arguments.timeout)
                except subprocess.TimeoutExpired:
                    # This process group belongs exclusively to this test run.
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise RuntimeError(f"Simulation exceeded {self.arguments.timeout:g} s")
            runResult["return_code"] = returnCode
            runResult["elapsed_seconds"] = time.monotonic() - started
            if returnCode != 0:
                raise RuntimeError(f"Simulation exited with status {returnCode}; see {directory / 'run.log'}")
            case = Case(name, ranks, threads, parameters, outputDirectory,
                        pd.read_csv(outputDirectory / "diagnostics.csv"))
            self.cases[name] = case
            self.check(name + ": execution", True, elapsed_seconds=runResult["elapsed_seconds"])
            self.check_common(case)
            return case
        except (OSError, ValueError, RuntimeError, pd.errors.ParserError) as error:
            runResult["elapsed_seconds"] = time.monotonic() - started
            runResult["error"] = str(error)
            self.check(name + ": execution/data", False, error=str(error))
            return None

    def check_common(self, case: Case) -> None:
        metadata = dict(line.split("=", 1) for line in
                        (case.directory / "metadata.txt").read_text().splitlines() if "=" in line)
        execution = validate_runtime_metadata(metadata, case.ranks, case.threads)
        self.check(case.name + ": actual MPI/backend/host-thread configuration", True,
                   execution=execution, requested_ranks=case.ranks,
                   requested_host_threads=case.threads)
        data = case.diagnostics
        if not set(DiagnosticColumns).issubset(data.columns):
            raise ValueError(f"{case.name}: missing diagnostic columns")
        finite = bool(np.isfinite(data.loc[:, list(DiagnosticColumns)].to_numpy(dtype=float)).all())
        self.check(case.name + ": finite diagnostics", finite)
        count = case.parameters["np"]**3
        self.check(case.name + ": exact particle count", bool((data["particles"] == count).all()),
                   expected=count)
        self.check(case.name + ": mass conservation", bool(data["mass_error"].abs().max() <= MassTolerance),
                   maximum_error=float(data["mass_error"].abs().max()), tolerance=MassTolerance)
        ai = 1.0 / (1.0 + case.parameters["z_in"])
        af = 1.0 / (1.0 + case.parameters["z_fi"])
        endpointError = max(abs(data["a"].iloc[0] / ai - 1.0),
                            abs(data["a"].iloc[-1] / af - 1.0))
        self.check(case.name + ": integration endpoints", bool(
            data["step"].iloc[0] == 0 and data["step"].iloc[-1] == case.parameters["nt"]
            and endpointError < 2.0e-13), scale_factor_error=float(endpointError))
        references = np.asarray([growth_reference(a, case.parameters["Omega_m"])
                                 for a in data["a"]])
        growthError = float(np.max(np.abs(data["D"].to_numpy() / references[:, 0] - 1.0)))
        rateError = float(np.max(np.abs(data["f"].to_numpy() / references[:, 1] - 1.0)))
        self.check(case.name + ": independent growth/background", bool(
            max(growthError, rateError) < GrowthReferenceTolerance),
            D_relative_error=growthError, f_relative_error=rateError,
            tolerance=GrowthReferenceTolerance)
        for epoch in ("initial", "final"):
            particles = case.snapshot(epoch)
            idsValid = (len(particles) == count and np.array_equal(
                particles["id"].to_numpy(), np.arange(count, dtype=np.uint64)))
            values = particles.loc[:, list(SnapshotColumns[1:])].to_numpy()
            positions = particles[["x", "y", "z"]].to_numpy()
            self.check(case.name + f": {epoch} particle integrity", bool(
                idsValid and np.isfinite(values).all()
                and (positions >= 0.0).all()
                and (positions < case.parameters["box_size"]).all()),
                count=len(particles), unique_ids=int(particles["id"].nunique()))

    def check_uniform(self, case: Case | None) -> None:
        if case is None:
            return
        positionError, momentumError = phase_space_difference(
            case.snapshot("initial"), case.snapshot("final"), case.parameters["box_size"])
        forceError = float(case.diagnostics["force_rms"].abs().max())
        self.check(case.name + ": zero peculiar evolution", bool(
            positionError < 1.0e-12 and momentumError < 1.0e-12 and forceError < 1.0e-10),
            position_rms_error=positionError, momentum_rms_error=momentumError,
            maximum_force_rms=forceError)

    def check_sine(self, case: Case | None, tolerance: float) -> float | None:
        if case is None:
            return None
        mode = np.asarray([[case.parameters[f"mode_{axis}"] for axis in ("x", "y", "z")]])
        initial = fourier_modes(case.snapshot("initial"), case.parameters["box_size"], mode)[0]
        final = fourier_modes(case.snapshot("final"), case.parameters["box_size"], mode)[0]
        ai = 1.0 / (1.0 + case.parameters["z_in"])
        af = 1.0 / (1.0 + case.parameters["z_fi"])
        growth = (growth_reference(af, case.parameters["Omega_m"])[0]
                  / growth_reference(ai, case.parameters["Omega_m"])[0])
        amplitude = case.parameters["amplitude"]
        # Exact displaced-lattice fundamental: 2*J1(A)=A-A^3/8+O(A^5).
        initialExpected = amplitude - amplitude**3 / 8.0
        initialError = float(abs(initial - initialExpected) / abs(initialExpected))
        growthError = float(abs(final - growth * initial) / abs(growth * initial))
        self.check(case.name + ": initial sine mode", initialError < 1.0e-7,
                   relative_error=initialError, amplitude=abs(initial), tolerance=1.0e-7)
        self.check(case.name + ": linear mode growth", growthError < tolerance,
                   relative_error=growthError, measured_growth=float(abs(final / initial)),
                   continuum_growth=growth, tolerance=tolerance)
        # Independently check conversion from displacement to canonical p.
        initialMomentum = case.snapshot("initial")[["px", "py", "pz"]].to_numpy()
        modeVector = mode[0] * (2.0 * math.pi / case.parameters["box_size"])
        # Invert x=q-A sin(k.q) k/k^2 by well-conditioned fixed-point iteration.
        x = case.snapshot("initial")[["x", "y", "z"]].to_numpy()
        q = x.copy()
        for _ in range(4):
            q = x + amplitude * np.sin(q @ modeVector)[:, None] * modeVector / np.dot(modeVector, modeVector)
        expansion = math.sqrt(case.parameters["Omega_m"] / ai**3 + 1.0 - case.parameters["Omega_m"])
        growthRate = growth_reference(ai, case.parameters["Omega_m"])[1]
        expectedMomentum = (-ai**2 * expansion * growthRate * amplitude
                            * np.sin(q @ modeVector)[:, None] * modeVector / np.dot(modeVector, modeVector))
        momentumError = float(np.linalg.norm(initialMomentum - expectedMomentum)
                              / np.linalg.norm(expectedMomentum))
        self.check(case.name + ": canonical initial momentum", momentumError < 1.0e-8,
                   relative_error=momentumError, tolerance=1.0e-8)
        return growthError

    def check_gaussian(self, case: Case | None) -> None:
        if case is None:
            return
        # One member from every +/- pair with 0<|n|^2<=3 avoids double counting.
        modes = np.asarray([
            (nx, ny, nz) for nx in range(-1, 2) for ny in range(-1, 2)
            for nz in range(-1, 2)
            if 0 < nx*nx + ny*ny + nz*nz <= 3
            and next(value for value in (nx, ny, nz) if value != 0) > 0
        ])
        initial = fourier_modes(case.snapshot("initial"), case.parameters["box_size"], modes)
        final = fourier_modes(case.snapshot("final"), case.parameters["box_size"], modes)
        ai = 1.0 / (1.0 + case.parameters["z_in"])
        af = 1.0 / (1.0 + case.parameters["z_fi"])
        growth = (growth_reference(af, case.parameters["Omega_m"])[0]
                  / growth_reference(ai, case.parameters["Omega_m"])[0])
        norm = float(np.linalg.norm(initial))
        self.check(case.name + ": nonzero Gaussian realization", norm > 1.0e-8,
                   low_k_initial_norm=norm)
        error = float(np.linalg.norm(final - growth * initial) / (growth * norm)) if norm else math.inf
        tolerance = QuickGaussianGrowthTolerance if case.parameters["np"] == 16 else GaussianGrowthTolerance
        self.check(case.name + ": Gaussian low-k linear growth", error < tolerance,
                   weighted_complex_relative_error=error, continuum_growth=growth,
                   mode_count=len(modes), tolerance=tolerance)
        self.check_gaussian_initial(case)

    def check_gaussian_initial(self, case: Case) -> None:
        n = case.parameters["np"]
        boxSize = case.parameters["box_size"]
        initial = case.snapshot("initial")
        ids = initial["id"].to_numpy()
        lattice = np.column_stack((ids % n, (ids // n) % n, ids // (n*n)))
        lattice = (lattice + 0.5) * (boxSize / n)
        displacement = initial[["x", "y", "z"]].to_numpy() - lattice
        displacement -= boxSize * np.rint(displacement / boxSize)
        ai = 1.0 / (1.0 + case.parameters["z_in"])
        growth, rate = growth_reference(ai, case.parameters["Omega_m"])
        expansion = math.sqrt(case.parameters["Omega_m"] / ai**3 + 1.0 - case.parameters["Omega_m"])
        predictedMomentum = ai**2 * expansion * rate * displacement
        momentum = initial[["px", "py", "pz"]].to_numpy()
        momentumError = float(np.linalg.norm(momentum-predictedMomentum)
                              / np.linalg.norm(predictedMomentum))
        self.check(case.name + ": Gaussian canonical momentum", momentumError < 1.0e-8,
                   relative_error=momentumError, tolerance=1.0e-8)
        wave = 2.0 * math.pi * np.fft.fftfreq(n, d=boxSize/n)
        kz, ky, kx = np.meshgrid(wave, wave, wave, indexing="ij")
        reconstructed = np.zeros((n, n, n), dtype=np.complex128)
        for component, componentK in enumerate((kx, ky, kz)):
            psi = (displacement[:, component] / growth).reshape((n, n, n))
            reconstructed -= 1j * componentK * np.fft.fftn(psi, norm="forward")
        kMagnitude = np.sqrt(kx*kx + ky*ky + kz*kz)
        valid = kMagnitude > 0
        valid[n//2, :, :] = False
        valid[:, n//2, :] = False
        valid[:, :, n//2] = False
        power = np.zeros_like(kMagnitude)
        power[valid] = bbks_spectrum(kMagnitude[valid], case.parameters)
        expected = gaussian_coefficients(n, case.parameters["seed"], power, boxSize)
        coefficientError = float(np.linalg.norm(reconstructed-expected) / np.linalg.norm(expected))
        self.check(case.name + ": independent Gaussian Fourier coefficients",
                   coefficientError < GaussianCoefficientTolerance,
                   relative_l2_error=coefficientError, tolerance=GaussianCoefficientTolerance)
        # Each independent conjugate pair has exponential |delta|^2/(P/V).
        # Six standard errors provide a predeclared finite-realization gate;
        # the exact coefficient comparison above is the tighter amplitude test.
        independentCount = int(valid.sum()) // 2
        normalizedPower = boxSize**3 * np.abs(reconstructed[valid])**2 / power[valid]
        meanPower = float(normalizedPower.mean())
        statisticalLimit = 6.0 / math.sqrt(independentCount)
        self.check(case.name + ": Gaussian power normalization",
                   abs(meanPower-1.0) < statisticalLimit,
                   mean_normalized_power=meanPower, independent_pairs=independentCount,
                   six_sigma_limit=statisticalLimit)

    def check_reproducibility(self, reference: Case | None, other: Case | None) -> None:
        if reference is None or other is None:
            return
        boxSize = reference.parameters["box_size"]
        for epoch in ("initial", "final"):
            left = reference.snapshot(epoch)
            right = other.snapshot(epoch)
            positionError, momentumError = phase_space_difference(left, right, boxSize)
            initial = reference.snapshot("initial")
            momentumScale = float(np.sqrt(np.mean(initial[["px", "py", "pz"]].to_numpy()**2)))
            ai = 1.0 / (1.0 + reference.parameters["z_in"])
            expansion = math.sqrt(reference.parameters["Omega_m"] / ai**3
                                  + 1.0 - reference.parameters["Omega_m"])
            rate = growth_reference(ai, reference.parameters["Omega_m"])[1]
            displacementScale = momentumScale / (ai**2 * expansion * rate)
            # Include the roundoff incurred representing small displacements
            # relative to coordinates of order boxSize.
            positionLimit = max(displacementScale * ReproducibilityTolerance,
                                128 * np.finfo(float).eps * boxSize)
            momentumLimit = max(momentumScale * ReproducibilityTolerance,
                                np.finfo(float).tiny)
            self.check(other.name + f": {epoch} agrees with {reference.name}", bool(
                positionError < positionLimit and momentumError < momentumLimit),
                position_rms_error=positionError, momentum_rms_error=momentumError,
                position_limit=positionLimit, momentum_limit=momentumLimit)

    def run_suite(self) -> bool:
        for ranks in range(1, 5):
            self.check_uniform(self.run(f"uniform_r{ranks}", ranks=ranks,
                                        np=16, nt=4, ic_mode="uniform"))
        axisCases = []
        gaussianCases = []
        for ranks in range(1, 5):
            case = self.run(f"sine_r{ranks}", ranks=ranks)
            axisCases.append(case)
            self.check_sine(case, AxisGrowthTolerance)
            if ranks > 1:
                self.check_reproducibility(axisCases[0], case)
        for ranks in (1, 4):
            self.check_sine(self.run(f"oblique_r{ranks}", ranks=ranks,
                                     mode_x=1, mode_y=1, mode_z=1), ObliqueGrowthTolerance)
        self.check_sine(self.run("lambda_late_r4", ranks=4, z_in=9.0, z_fi=0.0,
                                 amplitude=0.0001, nt=128), AxisGrowthTolerance)
        for ranks in range(1, 5):
            case = self.run(f"gaussian_r{ranks}", ranks=ranks, ic_mode="gaussian",
                            np=16 if self.arguments.quick else 32)
            gaussianCases.append(case)
            self.check_gaussian(case)
            if ranks > 1:
                self.check_reproducibility(gaussianCases[0], case)
        threadCase = self.run("gaussian_threads2", ranks=1, threads=2, ic_mode="gaussian",
                              np=16 if self.arguments.quick else 32)
        self.check_gaussian(threadCase)
        self.check_reproducibility(gaussianCases[0], threadCase)
        if not self.arguments.quick:
            timeCases = [self.run(f"time_nt{steps}", nt=steps) for steps in (4, 8, 16)]
            if all(case is not None for case in timeCases):
                coarseDifference = phase_space_difference(timeCases[0].snapshot("final"),
                    timeCases[1].snapshot("final"), timeCases[0].parameters["box_size"])
                fineDifference = phase_space_difference(timeCases[1].snapshot("final"),
                    timeCases[2].snapshot("final"), timeCases[0].parameters["box_size"])
                for index, label in enumerate(("position", "momentum")):
                    ratio = coarseDifference[index] / fineDifference[index] if fineDifference[index] else math.inf
                    self.check(f"second-order time convergence: {label}",
                        TimeConvergenceMinimum <= ratio <= TimeConvergenceMaximum,
                        coarse_difference=coarseDifference[index], fine_difference=fineDifference[index],
                        ratio=ratio, observed_order=math.log2(ratio) if ratio > 0 else -math.inf,
                        acceptable_ratio=[TimeConvergenceMinimum, TimeConvergenceMaximum])
            meshErrors = []
            for cells in (16, 32, 64):
                meshCase = self.run(f"mesh_n{cells}", np=cells, nt=64)
                meshErrors.append(self.check_sine(meshCase, AxisGrowthTolerance * (32.0 / cells)**2))
            if all(error is not None for error in meshErrors):
                for index in (0, 1):
                    ratio = meshErrors[index] / meshErrors[index + 1] if meshErrors[index + 1] else math.inf
                    self.check(f"mesh convergence: N={16 * 2**index} to {32 * 2**index}",
                        ratio >= MeshConvergenceMinimum and meshErrors[index + 1] < meshErrors[index],
                        coarse_relative_error=meshErrors[index], fine_relative_error=meshErrors[index + 1],
                        error_ratio=ratio, minimum_ratio=MeshConvergenceMinimum)
        passed = all(check["passed"] for check in self.results["checks"])
        self.results["passed"] = passed
        self.results["finished_utc"] = datetime.now(timezone.utc).isoformat()
        self.save()
        print(f"\n{'PASS' if passed else 'FAIL'}: {len(self.results['runs'])} simulations, "
              f"{len(self.results['checks'])} checks; results: {self.resultPath}", flush=True)
        return passed


def self_test() -> None:
    """Check the independent numerical oracle without a simulation executable."""
    for scale in (0.001, 0.02, 0.2, 0.5, 1.0):
        growth, rate = growth_reference(scale, 1.0)
        assert abs(growth / scale - 1.0) < 2.0e-12, (scale, growth)
        assert abs(rate - 1.0) < 2.0e-10, (scale, rate)
    for scale in (0.02, 0.2, 0.5, 1.0):
        growth, rate = growth_reference(scale, 0.31)
        epsilon = 1.0e-5
        finiteDifference = (math.log(growth_reference(scale * math.exp(epsilon), 0.31)[0])
                            - math.log(growth_reference(scale * math.exp(-epsilon), 0.31)[0])) / (2.0 * epsilon)
        assert abs(finiteDifference - rate) < 2.0e-9, (scale, rate, finiteDifference)
    print("PASS independent growth oracle: Einstein-de Sitter and finite-difference derivative")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", type=Path, help="Path to the Cosmology executable")
    parser.add_argument("--work-dir", type=Path,
                        help="New results directory (default: unique cosmology-validation-* in current directory)")
    parser.add_argument("--mpiexec", default="mpirun", help="MPI launcher (default: mpirun)")
    parser.add_argument("--numproc-flag", default="-n",
                        help="MPI process-count flag (default: -n; specify e.g. --numproc-flag=-np)")
    parser.add_argument("--mpi-arg", action="append", default=[], help="Launcher argument; repeat as needed (e.g. --mpi-arg=--oversubscribe)")
    parser.add_argument("--timeout", type=float, default=240.0, help="Maximum seconds per MPI run")
    parser.add_argument("--quick", action="store_true", help="Retain ranks 1-4; smaller Gaussian mesh, omit convergence studies")
    parser.add_argument("--self-test", action="store_true", help="Test the independent growth oracle and exit")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
        return 0
    if arguments.exe is None:
        parser.error("--exe is required unless --self-test is selected")
    if not arguments.exe.is_file() or not os.access(arguments.exe, os.X_OK):
        parser.error(f"Executable is missing or not executable: {arguments.exe}")
    if arguments.timeout <= 0:
        parser.error("--timeout must be positive")
    self_test()
    try:
        return 0 if Validation(arguments).run_suite() else 1
    except (OSError, ValueError) as error:
        print(f"Validation setup/data error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
