#!/usr/bin/env python3
"""MPI input/output regressions for CompareCosmologyForce.

Run with --exe /path/to/CompareCosmologyForce. Results are retained in the
user-selected --output-dir or a unique directory beside the executable.
The twelve runs cover ranks 1 through 4, shuffled IDs, multiple periodic
wraps, rank-invariant nonzero forces, and six malformed-input rejections.
Requires numpy and pandas. This complements validate_frozen_force.py's
independent numerical oracle; it does not replace that physics comparison.
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
from runtime_metadata import validate_runtime_metadata


GridSize = 8
BoxSize = 10.0
OmegaMatter = 0.31


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    # Unlike assert statements, these gates remain active under python -O.
    if not condition:
        raise AssertionError(message)


class AdapterTests:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.executable = args.exe.resolve()
        self.directory = (args.output_dir.resolve() if args.output_dir else
                          Path(tempfile.mkdtemp(prefix="frozen-adapter-", dir=self.executable.parent)))
        self.directory.mkdir(parents=True, exist_ok=True)
        require(not any(self.directory.iterdir()), "Output directory must be new or empty")
        (self.directory / "fixtures").mkdir()
        self.ids = np.arange(GridSize**3)
        self.lattice = (np.column_stack((self.ids % GridSize, (self.ids // GridSize) % GridSize,
                                        self.ids // GridSize**2)) + 0.5) * (BoxSize / GridSize)
        sourceDirectory = Path(__file__).resolve().parents[1]
        sources = (self.executable, Path(__file__).resolve(),
                   sourceDirectory / "tests/CompareCosmologyForce.cpp",
                   sourceDirectory / "ExecutionMetadata.h", sourceDirectory / "runtime_metadata.py",
                   sourceDirectory / "CosmologySimulation.h")
        self.results = {
            "passed": False, "command": sys.argv,
            "profile": {"n_grid": GridSize, "box_size": BoxSize, "omega_m": OmegaMatter},
            "hashes_before": {str(path): sha256(path) for path in sources},
            "expected_runs": 12, "runs": [], "checks": [],
            "tolerances": {"zero_force_absolute": 1e-12, "uniform_position_absolute": 2e-14,
                           "rank_absolute": 2e-12, "rank_relative": 1e-12},
        }
        self.save()
        print(f"Adapter regression results: {self.directory / 'results.json'}", flush=True)

    def save(self):
        (self.directory / "results.json").write_text(json.dumps(self.results, indent=2) + "\n")

    def fixture(self, name, positions):
        path = self.directory / "fixtures" / (name + ".csv")
        order = np.random.default_rng(7719).permutation(GridSize**3)
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("id", "x", "y", "z", "mass"))
            for index in order:
                writer.writerow((index, *positions[index], 1))
        return path

    def read_outputs(self, directory, ranks):
        def read(kind, columns):
            expected = {directory / f"{kind}_rank{rank}.csv" for rank in range(ranks)}
            require(set(directory.glob(f"{kind}_rank*.csv")) == expected,
                    f"Incomplete or unexpected {kind} rank files")
            frames = [pd.read_csv(path) for path in sorted(expected)]
            require(all(list(frame.columns) == columns for frame in frames),
                    f"Unexpected {kind} CSV header")
            data = pd.concat(frames, ignore_index=True)
            require(len(data) == GridSize**3 and np.isfinite(data.to_numpy()).all(),
                    f"Wrong row count or nonfinite {kind} output")
            return data

        particles = read("forces", ["id", "x", "y", "z", "fx", "fy", "fz"]).sort_values("id")
        density = read("density", ["ix", "iy", "iz", "delta", "fx", "fy", "fz"])
        density = density.sort_values(["iz", "iy", "ix"])
        np.testing.assert_array_equal(particles["id"].to_numpy(), self.ids)
        expectedIndices = np.column_stack((self.ids % GridSize, (self.ids // GridSize) % GridSize,
                                           self.ids // GridSize**2))
        np.testing.assert_array_equal(density[["ix", "iy", "iz"]].to_numpy(), expectedIndices)
        positions = particles[["x", "y", "z"]].to_numpy()
        require((positions >= 0).all() and (positions < BoxSize).all(),
                "Positions are outside the periodic box")
        metadata = dict(line.split("=", 1) for line in
                        (directory / "metadata.txt").read_text().splitlines() if "=" in line)
        validate_runtime_metadata(metadata, ranks)
        require(int(metadata["n_grid"]) == GridSize and float(metadata["box_size"]) == BoxSize
                and float(metadata["omega_m"]) == OmegaMatter,
                "Metadata does not match frozen-force arguments")
        require(float(metadata["mass_error"]) < 2e-12, "Deposited mass is not conserved")
        return particles, density

    def run_case(self, name, fixture, ranks, *, rejection=False, uniform=False, errorMarkers=()):
        directory = self.directory / name
        command = (shlex.split(self.args.mpiexec) + self.args.mpi_arg
                   + [self.args.numproc_flag, str(ranks), str(self.executable),
                      str(GridSize), str(BoxSize), str(OmegaMatter), str(fixture), str(directory)])
        overrides = {"OMP_NUM_THREADS": "1", "OMP_PROC_BIND": "false",
                     "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        environment = dict(os.environ, **overrides)
        record = {"name": name, "ranks": ranks, "command": command, "passed": False,
                  "expected_rejection": rejection, "input_sha256": sha256(fixture),
                  "environment_overrides": overrides}
        self.results["runs"].append(record)
        self.save()
        print(f"RUN {name}", flush=True)
        started = time.monotonic()
        try:
            with (self.directory / (name + ".log")).open("w") as log:
                process = subprocess.Popen(command, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    returnCode = process.wait(timeout=self.args.timeout)
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
                    raise RuntimeError(f"Timed out after {self.args.timeout:g} seconds")
            record["return_code"] = returnCode
            require((returnCode != 0) if rejection else (returnCode == 0),
                    f"Unexpected exit code {returnCode}")
            if rejection:
                require(not directory.exists(), "Rejected input created simulation output")
                logText = (self.directory / (name + ".log")).read_text()
                require(any(marker in logText for marker in errorMarkers),
                        "Nonzero exit did not report the expected input-validation error")
                record["expected_error_markers"] = errorMarkers
                output = None
            else:
                output = self.read_outputs(directory, ranks)
                particles, density = output
                if uniform:
                    np.testing.assert_allclose(particles[["x", "y", "z"]].to_numpy(), self.lattice,
                                               atol=2e-14, rtol=0)
                    require(np.abs(particles[["fx", "fy", "fz"]].to_numpy()).max() < 1e-12,
                            "Uniform particles have nonzero force")
                    require(np.abs(density[["delta", "fx", "fy", "fz"]].to_numpy()).max() < 1e-12,
                            "Uniform deposited grid or grid force is nonzero")
                else:
                    require(np.linalg.norm(particles[["fx", "fy", "fz"]].to_numpy()) > 0.01,
                            "Sine force is unexpectedly zero")
            record["passed"] = True
            return output
        except (OSError, ValueError, KeyError, AssertionError, RuntimeError) as error:
            record["error"] = str(error)
            return None
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            self.save()
            print(f"{'PASS' if record['passed'] else 'FAIL'} {name}", flush=True)

    def run(self):
        uniform = self.fixture("uniform-shuffled-multiwrap",
                               self.lattice + BoxSize * np.array([3, -4, 2]))
        for ranks in (1, 2, 3, 4):
            self.run_case(f"uniform-r{ranks}", uniform, ranks, uniform=True)
        positions = self.lattice.copy()
        positions[:, 0] -= 0.02 * np.sin(2 * np.pi * positions[:, 0] / BoxSize) / (2 * np.pi / BoxSize)
        sine = self.fixture("sine-shuffled", positions)
        one = self.run_case("sine-r1", sine, 1)
        four = self.run_case("sine-r4", sine, 4)
        comparison = {"name": "sine rank-invariant phase space and mesh", "passed": False}
        try:
            require(one is not None and four is not None, "Missing successful sine run")
            for oneFrame, fourFrame in zip(one, four):
                np.testing.assert_allclose(oneFrame.to_numpy(), fourFrame.to_numpy(),
                                           atol=2e-12, rtol=1e-12)
            comparison["passed"] = True
        except (AssertionError, ValueError) as error:
            comparison["error"] = str(error)
        self.results["checks"].append(comparison)
        with uniform.open(newline="") as stream:
            rows = list(csv.reader(stream))
        for name, mutation, errorMarkers in (
            ("bad-header", lambda r: r.__setitem__(0, ["bad", "x", "y", "z", "mass"]),
             ("header must be exactly",)),
            ("duplicate-id", lambda r: r[2].__setitem__(0, r[1][0]),
             ("Duplicate frozen particle ID",)),
            ("bad-mass", lambda r: r[1].__setitem__(4, "2"),
             ("requires each mass to equal 1",)),
            ("nonfinite-position", lambda r: r[1].__setitem__(1, "nan"),
             ("Invalid position", "positions must be finite")),
            ("fractional-id", lambda r: r[1].__setitem__(0, "1.5"),
             ("Particle ID must be an integer",)),
            ("missing-row", lambda r: r.pop(),
             ("must contain every ID",)),
        ):
            altered = [row.copy() for row in rows]
            mutation(altered)
            path = self.directory / "fixtures" / (name + ".csv")
            with path.open("w", newline="") as stream:
                csv.writer(stream).writerows(altered)
            self.run_case(name, path, 4 if name == "duplicate-id" else 1, rejection=True,
                          errorMarkers=errorMarkers)
        self.results["hashes_after"] = {path: sha256(Path(path))
                                        for path in self.results["hashes_before"]}
        self.results["checks"].append({
            "name": "Source/executable unchanged during adapter tests",
            "passed": self.results["hashes_after"] == self.results["hashes_before"],
        })
        self.results["passed"] = (len(self.results["runs"]) == self.results["expected_runs"]
                                  and all(case["passed"] for case in self.results["runs"])
                                  and all(check["passed"] for check in self.results["checks"]))
        self.save()
        passed = self.results["passed"]
        print(f"{'PASS' if passed else 'FAIL'}: {len(self.results['runs'])} MPI runs; "
              f"{self.directory / 'results.json'}", flush=True)
        return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--mpi-arg", action="append", default=[])
    parser.add_argument("--numproc-flag", default="-n")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if not args.exe.is_file() or not os.access(args.exe, os.X_OK):
        parser.error("--exe must name an executable CompareCosmologyForce")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    try:
        return AdapterTests(args).run()
    except (OSError, ValueError, AssertionError) as error:
        print(f"Adapter test setup/data error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
