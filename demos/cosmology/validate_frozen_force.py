#!/usr/bin/env python3
"""Frozen CIC/PM qualification; never advances particles or fits a normalization.

The NumPy oracle is independent of both C++ implementations. FastPM's native
R2C Nyquist convention is tested separately, not silently replaced by IPPL's.
Only small local validation problems are supported by this host-side runner.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile

import numpy as np
import pandas as pd


# Fixed before simulation comparisons. Absolute floors scale with the box for
# forces (which have length units); relative budgets cover native float32 k/acc.
LIMITS = {
    "ippl": {"density_atol": 2e-12, "density_rtol": 2e-12,
             "force_atol_over_L": 2e-13, "force_rtol": 2e-11},
    "fastpm": {"density_atol": 5e-12, "density_rtol": 5e-12,
               "force_atol_over_L": 5e-9, "force_rtol": 5e-6},
}
FASTPM_COMMIT = "15b6c4fd7502a81d99dd13f54fcc9cfa44be1331"


def wrap(positions, box):
    """Match both adapters' double fmod wrapping, including rounded endpoints."""
    result = np.fmod(np.asarray(positions, dtype=float), box)
    result[result < 0] += box
    result[result >= box] = 0
    return result


def cic_stencil(positions, n, box):
    u = wrap(positions, box) / (box / n) - 0.5
    lower = np.floor(u).astype(np.int64)
    frac = u - lower
    for corner in itertools.product((0, 1), repeat=3):
        indices = (lower + corner) % n
        weights = np.prod(np.where(np.asarray(corner), frac, 1 - frac), axis=1)
        yield tuple(indices.T), weights


def deposit(positions, n, box):
    density = np.zeros((n, n, n))
    for indices, weights in cic_stencil(positions, n, box):
        np.add.at(density, indices, weights)
    return density - len(positions) / n**3


def gather(field, positions, box):
    result = np.zeros((len(positions), 3))
    for indices, weights in cic_stencil(positions, field.shape[0], box):
        result += weights[:, None] * field[indices]
    return result


def mesh_force(delta, box, omega, convention="ippl"):
    n = delta.shape[0]
    signed = np.fft.fftfreq(n) * n
    if convention == "ippl":
        spectrum = np.fft.fftn(delta, norm="forward")
        ks = np.meshgrid(*(signed * (2 * np.pi / box),) * 3, indexing="ij")
        k2 = sum(k * k for k in ks)
        inverse = np.zeros_like(k2)
        np.divide(1, k2, out=inverse, where=k2 != 0)
        result = []
        for dim, k in enumerate(ks):
            force_k = 1j * (1.5 * omega) * k * inverse * spectrum
            index = [slice(None)] * 3
            index[dim] = n // 2
            force_k[tuple(index)] = 0
            result.append(np.fft.ifftn(force_k, norm="forward").real)
    elif convention == "fastpm":
        # Native FASTPM_KERNEL_NAIVE: k and k*k are stored as floats even with
        # FFT_PRECISION=64. The compressed z Nyquist is NEGATIVE in MeshtoK.
        spectrum = np.fft.rfftn(delta, norm="forward")
        full_k = (signed * (2 * np.pi / box)).astype(np.float32)
        ks = np.meshgrid(full_k, full_k, full_k[:n // 2 + 1], indexing="ij")
        k2 = sum((k * k).astype(np.float64) for k in ks)
        inverse = np.zeros_like(k2)
        np.divide(1, k2, out=inverse, where=k2 != 0)
        corners = np.ones(k2.shape, dtype=bool)
        for k in ks:
            corners &= (k == 0) | (k == full_k[n // 2])
        result = []
        for k in ks:
            force_k = 1j * (1.5 * omega) * k.astype(float) * inverse * spectrum
            force_k[corners] = 0
            result.append(np.fft.irfftn(force_k, s=delta.shape, axes=(0, 1, 2), norm="forward"))
    else:
        raise ValueError(f"Unknown force convention: {convention}")
    return np.stack(result, axis=-1)


def common_band(field):
    """Diagnostic projection only: never applied to either code's force solve."""
    spectrum = np.fft.fftn(field, axes=(0, 1, 2), norm="forward")
    for dim in range(3):
        index = [slice(None)] * spectrum.ndim
        index[dim] = field.shape[0] // 2
        spectrum[tuple(index)] = 0
    spectrum[0, 0, 0] = 0
    return np.fft.ifftn(spectrum, axes=(0, 1, 2), norm="forward").real


def rms(array):
    array = np.asarray(array)
    return float(np.sqrt(np.mean(array * array)))


def fixtures(quick=False):
    box = 168.75
    specs = [("uniform", 16, .31), ("common_band", 16, .31),
             ("axis", 16, .31), ("oblique", 16, .31),
             ("jitter_wrapped", 16, .31), ("cluster", 16, .31),
             ("axis_fine", 32, .31), ("common_band_eds", 16, 1.)]
    if quick:
        specs = [entry for entry in specs if entry[0] in ("common_band", "jitter_wrapped")]
    for name, n, omega in specs:
        indices = np.indices((n, n, n)).reshape(3, -1).T
        h = box / n
        positions = (indices + .5) * h
        if name.startswith("common_band"):
            phase = 2 * np.pi * (indices @ np.array([1, 2, 3])) / n
            positions += h * np.array([.31, .43, .57])
            positions[:, 0] += .1 * h * np.cos(phase)
        elif name.startswith("axis") or name == "oblique":
            mode = np.array([1, 0, 0] if name.startswith("axis") else [1, 2, 1])
            k = mode * 2 * np.pi / box
            positions -= .6 * np.sin(positions @ k)[:, None] * k / (k @ k)
        elif name == "jitter_wrapped":
            rng = np.random.default_rng(417239)
            positions += rng.uniform(-.4, .4, positions.shape) * h
            positions[0] = [0, box, -box]
            positions += rng.integers(-3, 4, positions.shape) * box
        elif name == "cluster":
            positions = box * (.41 + .18 * np.random.default_rng(76112).random(positions.shape))
        yield {"name": name, "n": n, "box": box, "omega": omega, "positions": positions}


def read_output(path, n, ranks):
    def read(kind):
        paths = sorted(path.glob(f"{kind}_rank*.csv"))
        if {p.name for p in paths} != {f"{kind}_rank{i}.csv" for i in range(ranks)}:
            raise ValueError(f"{path}: wrong {kind} rank filename set")
        # A strongly clustered fixture legitimately leaves some ranks empty.
        # Their header-only frames must not coerce the concatenation to object.
        frames = [pd.read_csv(p) for p in paths]
        frame = pd.concat([f for f in frames if len(f)], ignore_index=True)
        if len(frame) != n**3 or not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"{path}: wrong row count or nonfinite {kind}")
        return frame
    particles = read("forces").sort_values("id")
    if not np.array_equal(particles["id"].to_numpy(), np.arange(n**3)):
        raise ValueError(f"{path}: invalid particle IDs")
    grid = read("density").sort_values(["ix", "iy", "iz"])
    if not np.array_equal(grid[["ix", "iy", "iz"]].to_numpy(),
                          np.indices((n, n, n)).reshape(3, -1).T):
        raise ValueError(f"{path}: invalid grid index permutation")
    metadata = dict(line.split("=", 1) for line in (path / "metadata.txt").read_text().splitlines()
                    if "=" in line)
    if int(metadata["ranks"]) != ranks or int(metadata["n_grid"]) != n:
        raise ValueError(f"{path}: metadata does not match run")
    return {"positions": particles[["x", "y", "z"]].to_numpy(),
            "particles": particles[["fx", "fy", "fz"]].to_numpy(),
            "delta": grid["delta"].to_numpy().reshape(n, n, n),
            "mesh": grid[["fx", "fy", "fz"]].to_numpy().reshape(n, n, n, 3),
            "metadata": metadata}


class Checks:
    def __init__(self):
        self.rows = []

    def compare(self, name, actual, expected, atol, rtol=0):
        scale = rms(expected)
        error = rms(np.asarray(actual) - expected)
        limit = atol + rtol * scale
        self.rows.append({"name": name, "error_rms": error, "reference_rms": scale,
                          "relative_rms": error / scale if scale > 0 else None,
                          "limit": limit, "passed": bool(np.isfinite(error) and error <= limit)})

    def scalar(self, name, error, limit):
        self.rows.append({"name": name, "error_rms": float(error), "limit": float(limit),
                          "passed": bool(np.isfinite(error) and error <= limit)})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def launch(command, log, environment):
    # Own a process group so a timeout cannot leave this task's MPI workers.
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               env=environment, start_new_session=True)
    try:
        code = process.wait(timeout=180)
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


def run(args):
    root = Path(args.output).resolve() if args.output else Path(tempfile.mkdtemp(
        prefix="frozen-force-", dir=Path(args.ippl_exe).resolve().parent))
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("Output directory must be new or empty")
    executables = {"ippl": Path(args.ippl_exe).resolve()}
    if args.fastpm_exe:
        executables["fastpm"] = Path(args.fastpm_exe).resolve()
    source_root = Path(__file__).resolve().parent
    tracked = [Path(__file__).resolve(), source_root / "CosmologySimulation.h",
               source_root / "CosmologyConfig.h", source_root / "CosmologyPhysics.h",
               source_root / "tests/CompareCosmologyForce.cpp"] + list(executables.values())
    if args.fastpm_exe:
        tracked.extend([source_root / "reference/FastPMForce.c",
                        source_root / "reference/build_fastpm.sh"])
    if args.fastpm_manifest:
        tracked.append(Path(args.fastpm_manifest).resolve())
    provenance = {str(path): sha(path) for path in tracked}
    checks, records, differences = Checks(), [], []
    ranks_to_run = [1, 2] if args.quick else [1, 2, 3, 4]
    environment = dict(os.environ, OMP_NUM_THREADS="1", OMP_PROC_BIND="false")
    protocol = {"limits": LIMITS, "ranks": ranks_to_run, "quick": args.quick,
                "force": "F0=-grad(phi0); laplacian(phi0)=1.5*Omega_m*delta",
                "native_nyquist": "preserved; raw differences measured, predicted, and separately gated",
                "scope": "frozen equal-mass CIC, not time evolution", "hashes": provenance}
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    def save_report(complete):
        failures = [row for row in checks.rows if not row["passed"]]
        report = {"protocol": protocol, "runs": records, "checks": checks.rows,
                  "native_operator_differences": differences, "failures": failures,
                  "complete": complete, "passed": complete and not failures,
                  "run_count": len(records), "check_count": len(checks.rows)}
        (root / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return failures

    save_report(False)
    for fixture in fixtures(args.quick):
        name, n, box, omega = (fixture[key] for key in ("name", "n", "box", "omega"))
        original = fixture["positions"]
        positions = wrap(original, box)
        frame = pd.DataFrame(original, columns=["x", "y", "z"])
        frame.insert(0, "id", np.arange(n**3))
        frame["mass"] = 1
        input_path = root / (name + ".csv")
        frame.sample(frac=1, random_state=7419).to_csv(input_path, index=False, float_format="%.17g")
        delta = deposit(positions, n, box)
        oracles = {code: mesh_force(delta, box, omega, code) for code in executables}
        particle_oracles = {code: gather(force, positions, box) for code, force in oracles.items()}
        checks.scalar(f"{name}/oracle_mass", abs(float(delta.mean())), 2e-13)
        if name.startswith("common_band"):
            checks.compare(f"{name}/fixture_excluded_modes", delta, common_band(delta), 2e-13)
        output = {}
        for code, exe in executables.items():
            limits = LIMITS[code]
            fa, fr = limits["force_atol_over_L"] * box, limits["force_rtol"]
            da, dr = limits["density_atol"], limits["density_rtol"]
            output[code] = {}
            for ranks in ranks_to_run:
                label = f"{name}/{code}/{ranks}r"
                out = root / f"{name}-{code}-{ranks}r"
                command = [args.mpiexec, *args.mpi_arg, args.numproc_flag, str(ranks), str(exe),
                           str(n), str(box), str(omega), str(input_path), str(out)]
                with (root / f"{name}-{code}-{ranks}r.log").open("w") as log:
                    launch(command, log, environment)
                actual = read_output(out, n, ranks)
                if (int(actual["metadata"]["threads"]) != 1
                        or float(actual["metadata"]["box_size"]) != box
                        or float(actual["metadata"]["omega_m"]) != omega):
                    raise ValueError(f"{out}: mismatched threads or cosmology metadata")
                if code == "fastpm":
                    contract = {"upstream_commit": FASTPM_COMMIT, "kernel": "FASTPM_KERNEL_NAIVE",
                                "softening": "FASTPM_SOFTENING_NONE", "painter": "FASTPM_PAINTER_CIC",
                                "cic_deconvolution": "none", "fft_precision_bits": "64",
                                "particle_acc_precision_bits": "32",
                                "nyquist_zeroing": "fully_self_conjugate_corners_only"}
                    if any(actual["metadata"].get(key) != value for key, value in contract.items()):
                        raise ValueError(f"{out}: native reference does not match the pinned operator contract")
                output[code][ranks] = actual
                records.append({"name": label, "command": command, "output": str(out),
                                "input_sha256": sha(input_path), "metadata": actual["metadata"]})
                checks.compare(label + "/positions", actual["positions"], positions, box * 2e-14)
                checks.compare(label + "/density", actual["delta"], delta, da, dr)
                checks.scalar(label + "/mass", abs(float(actual["delta"].mean())), da)
                checks.compare(label + "/mesh_force", actual["mesh"], oracles[code], fa, fr)
                checks.compare(label + "/particle_force", actual["particles"], particle_oracles[code], fa, fr)
                # Separate solve from painting and gather from solving, using observed inputs.
                checks.compare(label + "/kernel_only", actual["mesh"],
                               mesh_force(actual["delta"], box, omega, code), fa, fr)
                checks.compare(label + "/gather_only", actual["particles"],
                               gather(actual["mesh"], actual["positions"], box), fa, fr)
                checks.scalar(label + "/net_force", rms(actual["particles"].mean(axis=0)),
                              fa + fr * rms(particle_oracles[code]))
                if ranks > 1:
                    for field in ("positions", "delta", "mesh", "particles"):
                        atol = box * 2e-14 if field == "positions" else da if field == "delta" else fa
                        rtol = 0 if field == "positions" else dr if field == "delta" else fr
                        checks.compare(label + "/rank_invariance_" + field,
                                       actual[field], output[code][1][field], atol, rtol)
                print(label, flush=True)
                save_report(False)
        if "fastpm" in output:
            fa = LIMITS["fastpm"]["force_atol_over_L"] * box
            fr = LIMITS["fastpm"]["force_rtol"]
            predicted_difference = particle_oracles["fastpm"] - particle_oracles["ippl"]
            for ranks in ranks_to_run:
                left, right = output["ippl"][ranks], output["fastpm"][ranks]
                label = f"{name}/cross_code/{ranks}r"
                checks.compare(label + "/common_band_mesh", common_band(right["mesh"]),
                               common_band(left["mesh"]), fa, fr)
                raw_difference = right["particles"] - left["particles"]
                # Residual uses full force scale, not the possibly tiny expected difference.
                checks.compare(label + "/predicted_native_difference", raw_difference,
                               predicted_difference, 2 * (fa + fr * rms(particle_oracles["ippl"])))
                if name.startswith("common_band") or name == "uniform":
                    checks.compare(label + "/raw_particle_agreement", right["particles"],
                                   left["particles"], fa, fr)
                differences.append({"name": label, "raw_difference_rms": rms(raw_difference),
                                    "predicted_difference_rms": rms(predicted_difference),
                                    "ippl_force_rms": rms(left["particles"]),
                                    "nyquist_density_rms": rms(delta - common_band(delta))})
    for path, digest in provenance.items():
        if sha(path) != digest:
            raise RuntimeError(f"Source or executable changed during validation: {path}")
    failures = save_report(True)
    print(f"{len(records)} runs, {len(checks.rows)} checks, {len(failures)} failures: {root / 'results.json'}")
    for failure in failures:
        print(json.dumps(failure))
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ippl-exe", required=True)
    parser.add_argument("--fastpm-exe")
    parser.add_argument("--fastpm-manifest")
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--numproc-flag", default="-n")
    parser.add_argument("--mpi-arg", action="append", default=[])
    parser.add_argument("--output")
    parser.add_argument("--quick", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
