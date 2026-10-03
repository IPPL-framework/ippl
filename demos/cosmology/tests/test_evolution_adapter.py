#!/usr/bin/env python3
"""Imported-evolution adapter checks: input integrity, KDK reuse and MPI1--4.

The unequal-mesh tests independently evaluate one EdS KDK step with a NumPy
CIC/spectral oracle. They detect incorrect mean-density normalization, unlike
a uniform-only test. Results, inputs and logs are retained for inspection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_frozen_force as oracle


ParticleGrid = 8
BoxSize = 8.0


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phase_difference(left, right):
    difference = left - right
    difference[:, :3] -= BoxSize * np.rint(difference[:, :3] / BoxSize)
    return difference


class EvolutionAdapterTests:
    def __init__(self, arguments):
        self.arguments = arguments
        self.executable = arguments.exe.resolve()
        self.production = arguments.production_exe.resolve()
        self.directory = arguments.output_dir.resolve() if arguments.output_dir else Path(
            tempfile.mkdtemp(prefix="evolution-adapter-", dir=self.executable.parent))
        self.directory.mkdir(parents=True, exist_ok=True)
        require(not any(self.directory.iterdir()), "Output directory must be new or empty")
        (self.directory / "fixtures").mkdir()
        self.ids = np.arange(ParticleGrid**3)
        self.q = (np.column_stack((self.ids % ParticleGrid, self.ids // ParticleGrid % ParticleGrid,
                                   self.ids // ParticleGrid**2)) + .5) * BoxSize / ParticleGrid
        source = Path(__file__).resolve().parents[1]
        tracked = [self.executable, self.production, Path(__file__).resolve(),
                   source / "tests/CompareCosmologyEvolution.cpp", source / "CosmologySimulation.h",
                   source / "CosmologyPhysics.h", source / "validate_frozen_force.py"]
        self.results = {"passed": False, "expected_runs": 19, "runs": [], "checks": [],
                        "hashes_before": {str(path): sha256(path) for path in tracked},
                        "tolerances": {"phase_space_absolute": 2e-12,
                                       "checkpoint_mass_relative": 2e-12,
                                       "one_step_oracle_absolute": 2e-12,
                                       "production_equivalence_absolute": 2e-12}}
        self.save()
        print(f"Evolution adapter results: {self.directory / 'results.json'}", flush=True)

    def save(self):
        (self.directory / "results.json").write_text(json.dumps(self.results, indent=2, allow_nan=False) + "\n")

    def fixture(self, name, phase):
        path = self.directory / "fixtures" / (name + ".csv")
        frame = pd.DataFrame(phase, columns=["x", "y", "z", "px", "py", "pz"])
        frame.insert(0, "id", self.ids)
        frame["mass"] = 1
        frame.sample(frac=1, random_state=7719).to_csv(path, index=False, float_format="%.17g")
        return path

    def launch(self, name, arguments, ranks, *, reject=()):
        command = shlex.split(self.arguments.mpiexec) + self.arguments.mpi_arg + [
            self.arguments.numproc_flag, str(ranks), *map(str, arguments)]
        record = {"name": name, "command": command, "ranks": ranks, "passed": False,
                  "expected_rejection": bool(reject)}
        self.results["runs"].append(record)
        self.save()
        print(f"RUN {name}", flush=True)
        started = time.monotonic()
        try:
            path = self.directory / (name + ".log")
            environment = dict(os.environ, OMP_NUM_THREADS="1", OMP_PROC_BIND="false",
                               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            with path.open("w") as log:
                process = subprocess.Popen(command, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    returnCode = process.wait(timeout=self.arguments.timeout)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise RuntimeError("MPI adapter regression timed out")
            record["return_code"] = returnCode
            require(returnCode != 0 if reject else returnCode == 0, f"Unexpected exit code {returnCode}")
            if reject:
                require(any(marker in path.read_text() for marker in reject),
                        "MPI failure did not report the expected adapter rejection")
                require(not Path(arguments[-1]).exists(), "Invalid input created evolution output")
            record["passed"] = True
        except (OSError, AssertionError, RuntimeError) as error:
            record["error"] = str(error)
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            self.save()
        return record["passed"]

    def check(self, name, function):
        record = {"name": name, "passed": False}
        try:
            function()
            record["passed"] = True
        except (ValueError, KeyError, OSError, AssertionError) as error:
            record["error"] = str(error)
        self.results["checks"].append(record)
        self.save()
        print(f"{'PASS' if record['passed'] else 'FAIL'} {name}", flush=True)

    def read_snapshot(self, directory, name, ranks):
        files = set(directory.glob(f"particles_{name}_rank*.csv"))
        require(files == {directory / f"particles_{name}_rank{rank}.csv" for rank in range(ranks)},
                "Incomplete or extra snapshot rank files")
        frames = [pd.read_csv(path, float_precision="round_trip") for path in sorted(files)]
        require(all(list(frame.columns) == ["id", "x", "y", "z", "px", "py", "pz"] for frame in frames),
                "Invalid snapshot column contract")
        frame = pd.concat([frame for frame in frames if len(frame)], ignore_index=True).sort_values("id")
        np.testing.assert_array_equal(frame.id, self.ids)
        phase = frame[["x", "y", "z", "px", "py", "pz"]].to_numpy()
        require(np.isfinite(phase).all(), "Nonfinite snapshot")
        require(((phase[:, :3] >= 0) & (phase[:, :3] < BoxSize)).all(), "Unwrapped positions")
        return phase

    def read_evolution(self, directory, ranks, mesh, ai, af, steps, checkpoints, omega=1.0):
        metadata = dict(line.split("=", 1) for line in
                        (directory / "metadata.txt").read_text().splitlines() if "=" in line)
        for key, value in (("ranks", ranks), ("threads", 1), ("n_particles_grid", ParticleGrid),
                           ("n_grid", mesh), ("n_steps", steps), ("n_checkpoints", checkpoints)):
            require(int(metadata[key]) == value, f"Incorrect metadata {key}")
        require(float(metadata["box_size"]) == BoxSize, "Incorrect box size")
        require(float(metadata["omega_m"]) == omega, "Incorrect matter density")
        require(float(metadata["a_initial"]) == ai and float(metadata["a_final"]) == af,
                "Incorrect integration interval")
        table = pd.read_csv(directory / "checkpoints.csv", float_precision="round_trip")
        require(list(table.columns) == ["checkpoint", "step", "a", "mass_error", "max_inverse_imaginary"],
                "Invalid checkpoint table schema")
        require(np.isfinite(table.to_numpy()).all(), "Nonfinite checkpoint diagnostics")
        np.testing.assert_array_equal(table.checkpoint, np.arange(checkpoints + 1))
        np.testing.assert_array_equal(table.step, np.arange(checkpoints + 1) * (steps // checkpoints))
        np.testing.assert_allclose(table.a, ai * np.exp(table.step * (math.log(af / ai) / steps)),
                                   atol=0, rtol=2e-15)
        require((table.mass_error >= 0).all() and table.mass_error.max() <= 2e-12,
                "Checkpoint mass error exceeds regression budget")
        require(set(directory.glob("particles_checkpoint*_rank*.csv")) == {
            directory / f"particles_checkpoint{checkpoint:04d}_rank{rank}.csv"
            for checkpoint in range(checkpoints + 1) for rank in range(ranks)},
            "Unexpected checkpoint snapshot set")
        return table, [self.read_snapshot(directory, f"checkpoint{checkpoint:04d}", ranks)
                       for checkpoint in range(checkpoints + 1)]

    def evolve(self, name, fixture, ranks=1, mesh=8, ai=.1, af=.4, steps=8, checkpoints=4, omega=1.0, reject=()):
        output = self.directory / name
        arguments = [self.executable, ParticleGrid, mesh, BoxSize, omega,
                     ai, af, steps, checkpoints, fixture, output]
        return self.launch(name, arguments, ranks, reject=reject), output

    def run(self):
        uniformMomentum = np.broadcast_to([.125, -.25, .375], self.q.shape)
        ballistic = np.column_stack((self.q + BoxSize * np.array([3, -4, 2]), uniformMomentum))
        fixture = self.fixture("ballistic", ballistic)
        ballisticRuns = []
        for ranks in (1, 2, 3, 4):
            okay, output = self.evolve(f"ballistic-r{ranks}", fixture, ranks=ranks)
            ballisticRuns.append((output, ranks))

            def ballistic_check(output=output, ranks=ranks, okay=okay):
                require(okay, "Missing successful ballistic run")
                table, states = self.read_evolution(output, ranks, 8, .1, .4, 8, 4)
                np.testing.assert_array_equal(states[0][:, :3], self.q)
                np.testing.assert_array_equal(states[0][:, 3:], uniformMomentum)
                for a, state in zip(table.a, states):
                    drift = 2 * (1 / math.sqrt(.1) - 1 / math.sqrt(a))
                    expected = np.column_stack((oracle.wrap(self.q + uniformMomentum * drift, BoxSize),
                                                uniformMomentum))
                    np.testing.assert_allclose(phase_difference(state, expected), 0, atol=2e-12, rtol=0)
            self.check(f"ballistic-r{ranks}: imported state and analytical ballistic checkpoints", ballistic_check)

        def rank_check():
            reference = self.read_evolution(ballisticRuns[0][0], 1, 8, .1, .4, 8, 4)[1]
            for output, ranks in ballisticRuns[1:]:
                other = self.read_evolution(output, ranks, 8, .1, .4, 8, 4)[1]
                for left, right in zip(reference, other):
                    np.testing.assert_allclose(phase_difference(left, right), 0, atol=2e-12, rtol=0)
        self.check("ballistic MPI1-4 phase-space equivalence", rank_check)

        phase = np.column_stack((self.q.copy(), .03 * np.cos(2 * np.pi * self.q / BoxSize)))
        phase[:, 0] -= .1 * np.sin(2 * np.pi * self.q[:, 0] / BoxSize)
        fixture = self.fixture("nonuniform", phase)
        ai, af = .1, .11
        # NM4/rank3 has an unsupported one-cell local axis in core halo exchange;
        # keep NM4/rank1, add NM6/rank1,3, and assert clean rejection below.
        for mesh in (4, 6, 16):
            for ranks in ((1,) if mesh == 4 else (1, 3)):
                okay, output = self.evolve(f"unequal-mesh{mesh}-r{ranks}", fixture, ranks=ranks,
                                           mesh=mesh, ai=ai, af=af, steps=1, checkpoints=1)

                def unequal_check(mesh=mesh, ranks=ranks, output=output, okay=okay):
                    require(okay, "Missing successful unequal-mesh run")
                    _, states = self.read_evolution(output, ranks, mesh, ai, af, 1, 1)
                    np.testing.assert_array_equal(states[0], phase)
                    meanMass = ParticleGrid**3 / mesh**3

                    def force(positions):
                        delta = oracle.deposit(positions, mesh, BoxSize) / meanMass
                        return oracle.gather(oracle.mesh_force(delta, BoxSize, 1.0), positions, BoxSize)

                    midpoint = math.sqrt(ai * af)
                    halfP = phase[:, 3:] + 2 * (math.sqrt(midpoint) - math.sqrt(ai)) * force(phase[:, :3])
                    x = oracle.wrap(phase[:, :3] + 2 * (1 / math.sqrt(ai) - 1 / math.sqrt(af)) * halfP, BoxSize)
                    p = halfP + 2 * (math.sqrt(af) - math.sqrt(midpoint)) * force(x)
                    np.testing.assert_allclose(phase_difference(states[-1], np.column_stack((x, p))),
                                               0, atol=2e-12, rtol=0)
                self.check(f"NP8/NM{mesh}-r{ranks}: one-step independent EdS/CIC/KDK oracle", unequal_check)

        productionOutput = self.directory / "production-sine-output"
        parameters = {"np": 8, "nt": 8, "box_size": BoxSize, "Omega_m": .31, "Omega_bar": 0,
                      "ic_mode": "sine", "amplitude": .01, "z_in": 9, "z_fi": 4,
                      "diagnostics_every": 1, "write_particles": 1, "output": str(productionOutput)}
        config = self.directory / "production-sine.par"
        config.write_text("".join(f"{key}={value}\n" for key, value in parameters.items()))
        productionOkay = self.launch("production-sine", [self.production, config], 1)
        try:
            require(productionOkay, "Missing successful production sine run")
            initial = self.read_snapshot(productionOutput, "initial", 1)
            final = self.read_snapshot(productionOutput, "final", 1)
            fixture = self.fixture("production-initial", initial)
        except (OSError, AssertionError, ValueError) as error:
            self.results["checks"].append({"name": "Production comparison inputs", "passed": False,
                                            "error": str(error)})
        else:
            for ranks in (1, 3):
                okay, output = self.evolve(f"production-import-r{ranks}", fixture, ranks=ranks,
                                           ai=.1, af=.2, omega=.31)

                def production_check(output=output, ranks=ranks, okay=okay):
                    require(okay, "Missing successful imported production run")
                    _, states = self.read_evolution(output, ranks, 8, .1, .2, 8, 4, omega=.31)
                    np.testing.assert_array_equal(states[0], initial)
                    np.testing.assert_allclose(phase_difference(states[-1], final), 0, atol=2e-12, rtol=0)
                self.check(f"production-r{ranks}: initial import and final default-driver equivalence", production_check)

        fixture = self.fixture("rejection-baseline", ballistic)
        with fixture.open(newline="") as stream:
            rows = list(csv.reader(stream))
        for name, mutate, markers in (
            ("bad-header", lambda r: r[0].__setitem__(4, "vx"), ("header must be exactly",)),
            ("duplicate-id", lambda r: r[2].__setitem__(0, r[1][0]), ("Duplicate evolution particle ID",)),
            ("nonfinite-momentum", lambda r: r[1].__setitem__(4, "nan"), ("Invalid momentum", "momenta must be finite")),
            ("bad-mass", lambda r: r[1].__setitem__(7, "2"), ("requires each mass to equal 1",)),
            ("missing-row", lambda r: r.pop(), ("must contain every ID",)),
        ):
            changed = [row.copy() for row in rows]
            mutate(changed)
            path = self.directory / "fixtures" / (name + ".csv")
            with path.open("w", newline="") as stream:
                csv.writer(stream).writerows(changed)
            self.evolve(name, path, ranks=4 if name == "duplicate-id" else 1, reject=markers)
        self.evolve("bad-checkpoint-count", fixture, checkpoints=3,
                    reject=("n_steps must be positive and divisible by n_checkpoints",))
        self.evolve("unsupported-one-cell-halo", fixture, mesh=4, ranks=3,
                    reject=("requires at least two local mesh cells per axis",))
        after = {path: sha256(path) for path in self.results["hashes_before"]}
        self.results["hashes_after"] = after
        self.results["checks"].append({"name": "Source and executables unchanged",
                                        "passed": after == self.results["hashes_before"]})
        self.results["passed"] = (len(self.results["runs"]) == self.results["expected_runs"]
                                  and all(row["passed"] for row in self.results["runs"])
                                  and all(row["passed"] for row in self.results["checks"]))
        self.save()
        print(f"{'PASS' if self.results['passed'] else 'FAIL'}: {len(self.results['runs'])} runs; "
              f"{self.directory / 'results.json'}", flush=True)
        return 0 if self.results["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--production-exe", type=Path, required=True)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--mpi-arg", action="append", default=[])
    parser.add_argument("--numproc-flag", default="-n")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    for executable in (args.exe, args.production_exe):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            parser.error(f"Missing executable: {executable}")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    try:
        return EvolutionAdapterTests(args).run()
    except (ValueError, OSError, AssertionError) as error:
        print(f"Evolution adapter setup/data failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
