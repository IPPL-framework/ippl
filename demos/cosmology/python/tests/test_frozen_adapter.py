#!/usr/bin/env python3
## @file test_frozen_adapter.py
# @brief MPI input/output regressions for CompareCosmologyForce.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
# Tests assert declared invariants using isolated fixtures; no scientific tolerances are relaxed by documentation.
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
from source_paths import source_path
from runtime_metadata import validate_runtime_metadata


## @var GridSize
# @brief Named GridSize protocol/schema value; the source initializer records its exact contents.
GridSize = 8
## @var BoxSize
# @brief Named BoxSize protocol/schema value; the source initializer records its exact contents.
BoxSize = 10.0
## @var OmegaMatter
# @brief Named OmegaMatter protocol/schema value; the source initializer records its exact contents.
OmegaMatter = 0.31


## @brief Compute the streaming SHA256 digest of the file bytes.
# @see cosmology_tools
#
# @param path Input/output filesystem path; freshness and hash requirements are defined by this routine.
# @return Lowercase hexadecimal SHA256 of the exact retained bytes.
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


## @brief Require the documented module workflow.
# @see cosmology_tools
#
# @param condition Boolean acceptance condition evaluated before recording this check.
# @param message Failure/check diagnostic retained for interpretation of the fixed acceptance condition.
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def require(condition: bool, message: str) -> None:
    # Unlike assert statements, these gates remain active under python -O.
    if not condition:
        raise AssertionError(message)


## @brief Regression suite for Adapter.
# @see cosmology_tools
class AdapterTests:
    ## @brief Initialize the instance with the explicit protocol and artifact ownership passed by the caller.
    # @see cosmology_tools
    #
    # @param args Parsed command-line options; see main/--help and the module workflow contract.
    def __init__(self, args: argparse.Namespace):
        ## @var args
        # @brief Parsed command-line options; see main/--help and the module workflow contract.
        self.args = args
        ## @var executable
        # @brief Selected native/application executable path; its bytes and declared build provenance must match.
        self.executable = args.exe.resolve()
        ## @var directory
        # @brief Artifact directory following this module's ownership/freshness contract.
        self.directory = (args.output_dir.resolve() if args.output_dir else
                          Path(tempfile.mkdtemp(prefix="frozen-adapter-", dir=self.executable.parent)))
        self.directory.mkdir(parents=True, exist_ok=True)
        require(not any(self.directory.iterdir()), "Output directory must be new or empty")
        (self.directory / "fixtures").mkdir()
        ## @var ids
        # @brief Global uint64 particle IDs; the supported diagnostic contract requires every expected ID exactly once.
        self.ids = np.arange(GridSize**3)
        ## @var lattice
        # @brief Retained lattice state owned by this instance; see the initialization and workflow contract.
        self.lattice = (np.column_stack((self.ids % GridSize, (self.ids // GridSize) % GridSize,
                                        self.ids // GridSize**2)) + 0.5) * (BoxSize / GridSize)
        sourceDirectory = Path(__file__).resolve().parents[1]
        sources = (self.executable, Path(__file__).resolve(),
                   source_path("tests/CompareCosmologyForce.cpp"),
                   source_path("ExecutionMetadata.h"), source_path("runtime_metadata.py"),
                   source_path("CosmologySimulation.h"))
        ## @var results
        # @brief Retained results state owned by this instance; see the initialization and workflow contract.
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

    ## @brief Persist the complete current report without discarding recorded failed checks.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def save(self):
        (self.directory / "results.json").write_text(json.dumps(self.results, indent=2) + "\n")

    ## @brief Construct the fixture for the documented module workflow.
    # @see cosmology_tools
    #
    # @param name Stable artifact/run/check identifier as defined by the caller.
    # @param positions Finite particle position array of shape (Nparticles,3), in comoving Mpc/h.
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
    def fixture(self, name, positions):
        path = self.directory / "fixtures" / (name + ".csv")
        order = np.random.default_rng(7719).permutation(GridSize**3)
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("id", "x", "y", "z", "mass"))
            for index in order:
                writer.writerow((index, *positions[index], 1))
        return path

    ## @brief Read and verify outputs.
    # @see cosmology_tools
    #
    # @param directory Artifact directory following this module's ownership/freshness contract.
    # @param ranks Positive MPI rank count; all expected snapshot shards must exist.
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
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

    ## @brief Execute case.
    # @see cosmology_tools
    #
    # @param name Stable artifact/run/check identifier as defined by the caller.
    # @param fixture Shared initial-condition data or its explicit case descriptor; it must remain consistent across compared solvers.
    # @param ranks Positive MPI rank count; all expected snapshot shards must exist.
    # @param rejection Expected malformed-input rejection text/pattern in an isolated regression.
    # @param uniform Whether the fixture is the periodic zero-perturbation state.
    # @param errorMarkers Expected error diagnostics that demonstrate rejection rather than incomplete output.
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
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

    ## @brief Execute the documented module workflow.
    # @see cosmology_tools
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
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


## @brief Parse the documented command-line interface and execute its selected workflow, retaining nonzero failures.
# @see cosmology_tools
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
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


## @cond CLI_DISPATCH
if __name__ == "__main__":
    sys.exit(main())
## @endcond
