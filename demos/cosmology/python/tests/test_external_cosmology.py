#!/usr/bin/env python3
## @file test_external_cosmology.py
# @brief Regress production binary initial conditions, unequal meshes, and exact output epochs.
# @ingroup cosmology_python
# @see cosmology_numerics cosmology_contracts cosmology_validation
# Imported canonical momentum is p=a*v_pec/100 and must not receive a growth rescaling.
# The independent EdS oracle uses K(a,b)=2(sqrt(b)-sqrt(a)) and
# D(a,b)=2(1/sqrt(a)-1/sqrt(b)); density is normalized by N_particle/N_mesh^3.
# Mesh diagnostics satisfy delta_rms=sqrt(sum_cells(delta^2)/N_mesh^3),
# independently of the number of particles deposited onto the mesh.
"""Production external-IC integration checks with retained fixtures and MPI logs.

The fixed 2e-12 absolute phase-space tolerance matches the existing imported
adapter regression. It measures roundoff across FFT/reduction orders on tiny
fixtures, rather than cosmological accuracy. Input roundtrips must be exact.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import signal
import struct
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

## @cond IMPORT_PATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
## @endcond
import validate_frozen_force as oracle
import quijote_io as quijoteIO
import quijote_analysis as analysis
import quijote_benchmark as benchmark
from test_quijote_io import write_fixture as writeGadgetFixture

## @var Header
# @brief Canonical little-endian 128-byte header: magic, local/global counts, a,L,Omega_m,Omega_L,h,mass,flags.
Header = struct.Struct("<8sQQ6dQ48x")
## @var RecordType
# @brief Each 56-byte record preserves an exact uint64 ID and six binary64 phase-space values.
RecordType = np.dtype([("id", "<u8"), ("phase", "<f8", (6,))])
## @var ParticleGrid
# @brief NP=8 gives 512 equal-mass particles independently of the NM force mesh.
ParticleGrid = 8
## @var BoxSize
# @brief Periodic comoving box length in Mpc/h.
BoxSize = 8.0
## @var Hubble
# @brief Dimensionless h recorded and checked independently of the momentum convention.
Hubble = 0.6711
## @var ParticleMass
# @brief Header-only physical particle mass in solar masses/h; PM density uses equal-particle weighting.
ParticleMass = 1.23456789e10
## @var PhaseTolerance
# @brief Existing diagnostic-adapter absolute MPI/oracle tolerance in comoving position and canonical momentum.
PhaseTolerance = 2e-12


## @brief Raise a retained regression failure when an invariant does not hold.
# @param condition Boolean invariant.
# @param message Failure diagnostic.
# @return None when the invariant holds.
def require(condition, message):
    if not condition:
        raise AssertionError(message)


## @brief Compare phase-space states using the nearest periodic image for x.
# @param left ID-sorted N by 6 state.
# @param right ID-sorted N by 6 reference state in identical units.
# @return State residual with dx in [-L/2,L/2] and unmodified dp.
def phaseDifference(left, right):
    difference = left - right
    difference[:, :3] -= BoxSize * np.rint(difference[:, :3] / BoxSize)
    return difference


## @brief Retain end-to-end invariants for external production evolution on MPI1,2,4.
# The tests hold N_particle=8^3 fixed while changing N_mesh, so an erroneous
# assumption N_particle=N_mesh^3 changes the force and fails the EdS oracle.
class ExternalCosmologyTests:
    ## @brief Create a fresh artifact directory and a periodic cubic reference load.
    # @param arguments Validated command-line options.
    def __init__(self, arguments):
        ## @var arguments
        # @brief Launcher and executable arguments selected by the caller.
        self.arguments = arguments
        ## @var directory
        # @brief Fresh retained directory containing inputs, output shards, logs, and results.json.
        self.directory = arguments.output_dir.resolve() if arguments.output_dir else Path(
            tempfile.mkdtemp(prefix="external-cosmology-"))
        self.directory.mkdir(parents=True, exist_ok=True)
        require(not any(self.directory.iterdir()), "Output directory must be new or empty")
        (self.directory / "fixtures").mkdir()
        ## @var ids
        # @brief Exactly contiguous zero-based uint64 IDs.
        self.ids = np.arange(ParticleGrid**3, dtype=np.uint64)
        ## @var positions
        # @brief Cell-centered uniform load q=(i+1/2)L/NP in Mpc/h.
        self.positions = (np.column_stack((self.ids % ParticleGrid,
                            self.ids // ParticleGrid % ParticleGrid,
                            self.ids // ParticleGrid**2)) + .5) * BoxSize / ParticleGrid
        ## @var results
        # @brief Retained fixed tolerances, individual runs, and invariant checks.
        self.results = {"passed": False, "phase_space_absolute_tolerance": PhaseTolerance,
                        "runs": [], "checks": []}
        self.save()
        print(f"External cosmology results: {self.directory / 'results.json'}", flush=True)

    ## @brief Persist successes and failures without hiding an unsuccessful launch.
    # @return None; results.json is overwritten with the complete current report.
    def save(self):
        (self.directory / "results.json").write_text(
            json.dumps(self.results, indent=2, allow_nan=False) + "\n")

    ## @brief Write the canonical sorted complete input independently of production readers.
    # @param name Fixture basename.
    # @param phase N by 6 positions/momenta in canonical units, with positions in [0,L].
    # @param scale Initial dimensionless scale factor.
    # @param omega Matter fraction; Lambda is 1-omega.
    # @return Path of the input fixture.
    def fixture(self, name, phase, scale=.1, omega=1.0):
        path = self.directory / "fixtures" / f"{name}.bin"
        records = np.empty(len(self.ids), dtype=RecordType)
        records["id"] = self.ids
        records["phase"] = phase
        header = Header.pack(b"IPPLPS01", len(self.ids), len(self.ids), scale, BoxSize,
                             omega, 1 - omega, Hubble, ParticleMass, 1)
        path.write_bytes(header + records.tobytes())
        return path

    ## @brief Launch only task-owned MPI processes, terminating their group on timeout.
    # @param name Run/log identifier.
    # @param command Executable and arguments, before MPI launcher prefixes.
    # @param ranks MPI process count; None launches a child CLI that owns its MPI command.
    # @param reject Expected diagnostic substrings for a deliberately invalid input.
    # @param threads Positive OpenMP thread count exported to the launched process.
    # @return True if success or the expected rejection was observed without a timeout.
    def launch(self, name, command, ranks, reject=(), threads=1):
        command = list(map(str, command))
        if ranks is not None:
            command = shlex.split(self.arguments.mpiexec) + self.arguments.mpi_arg + [
                self.arguments.numproc_flag, str(ranks), *command]
        record = {"name": name, "command": command, "ranks": ranks,
                  "expected_rejection": bool(reject), "passed": False}
        self.results["runs"].append(record)
        self.save()
        logPath = self.directory / f"{name}.log"
        environment = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_PROC_BIND="false",
                           OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        started = time.monotonic()
        print(f"RUN {name}", flush=True)
        try:
            with logPath.open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           env=environment, start_new_session=True)
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
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                    raise RuntimeError("MPI external-IC regression timed out")
            record["return_code"] = returnCode
            require(returnCode != 0 if reject else returnCode == 0,
                    f"Unexpected exit code {returnCode}; see {logPath}")
            if reject:
                logText = logPath.read_text().lower()
                require(any(marker.lower() in logText for marker in reject),
                        f"Failure lacks expected input-rejection diagnostic: {reject}")
            record["passed"] = True
        except (OSError, RuntimeError, AssertionError) as error:
            record["error"] = str(error)
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            self.save()
        return record["passed"]

    ## @brief Configure the production driver with a relative IC path and independent particle count.
    # @param name Stable run name.
    # @param fixture Canonical binary input path.
    # @param ranks MPI process count.
    # @param mesh Force mesh NM per dimension, independent of NP=8.
    # @param scale Initial a matching the header.
    # @param finalScale Final a in (scale,1].
    # @param steps Base uniform-log(a) KDK interval count.
    # @param omega Matter fraction.
    # @param outputs Requested decreasing redshifts, if any.
    # @param reject Expected diagnostic substrings for an invalid input.
    # @return (run passed, output directory).
    def evolve(self, name, fixture, ranks=1, mesh=16, scale=.1, finalScale=.11,
               steps=1, omega=1.0, outputs=None, reject=()):
        output = self.directory / name
        parameters = {"np": mesh, "particle_count": len(self.ids), "nt": steps,
                      "box_size": BoxSize, "Omega_m": omega, "Omega_bar": 0,
                      "hubble": Hubble, "ic_mode": "external",
                      "ic_file": os.path.relpath(fixture, self.directory),
                      "z_in": 1 / scale - 1, "z_fi": 1 / finalScale - 1,
                      "diagnostics_every": 1, "write_particles": 1,
                      "snapshot_format": "binary", "output": str(output)}
        if outputs is not None:
            parameters["output_redshifts"] = ",".join(map(str, outputs))
        config = self.directory / f"{name}.par"
        config.write_text("".join(f"{key}={value}\n" for key, value in parameters.items()))
        return self.launch(name, [self.arguments.exe, config], ranks, reject=reject), output

    ## @brief Record a scientific or protocol invariant without discarding other results.
    # @param name Invariant identifier.
    # @param function Callable that raises when its invariant fails.
    # @return None; results.json records acceptance and failures.
    def check(self, name, function):
        record = {"name": name, "passed": False}
        try:
            function()
            record["passed"] = True
        except (ValueError, KeyError, IndexError, OSError, AssertionError) as error:
            record["error"] = str(error)
        self.results["checks"].append(record)
        self.save()
        print(f"{'PASS' if record['passed'] else 'FAIL'} {name}", flush=True)

    ## @brief Verify the independently decoded rank-shard format and exact global ID set.
    # @param directory Production output directory.
    # @param name Snapshot token from snapshots.csv.
    # @param ranks Required number of rank shards.
    # @param scale Required synchronized scale factor.
    # @param omega Required matter fraction.
    # @return N by 6 state sorted by exact uint64 ID.
    def readSnapshot(self, directory, name, ranks, scale, omega=1.0):
        expectedPaths = {directory / f"particles_{name}_rank{rank}.bin" for rank in range(ranks)}
        require(set(directory.glob(f"particles_{name}_rank*.bin")) == expectedPaths,
                f"Incomplete or extra binary snapshot shards for {name}")
        records = []
        for path in sorted(expectedPaths):
            data = path.read_bytes()
            require(len(data) >= Header.size, "Truncated snapshot header")
            magic, localCount, totalCount, a, box, om, ol, h, mass, flags = Header.unpack_from(data)
            require(magic == b"IPPLPS01", "Incorrect binary snapshot magic")
            require(totalCount == len(self.ids), "Incorrect binary global particle count")
            require(flags == 0, "Distributed output must not claim sorted complete input")
            require(data[80:128] == bytes(48), "Nonzero reserved snapshot header bytes")
            np.testing.assert_allclose([a, box, om, ol, h, mass],
                                       [scale, BoxSize, omega, 1 - omega, Hubble, ParticleMass],
                                       rtol=2e-14, atol=0)
            require(len(data) == Header.size + localCount * RecordType.itemsize,
                    "Binary header count disagrees with exact file length")
            records.append(np.frombuffer(data, dtype=RecordType, offset=Header.size))
        particles = np.concatenate(records)
        particles = particles[np.argsort(particles["id"])]
        np.testing.assert_array_equal(particles["id"], self.ids)
        phase = particles["phase"].copy()
        require(np.isfinite(phase).all(), "Nonfinite output particle state")
        require(((phase[:, :3] >= 0) & (phase[:, :3] < BoxSize)).all(),
                "Snapshot contains unwrapped positions")
        return phase

    ## @brief Check manifest epoch/step monotonicity and mass diagnostics.
    # @param directory Production output directory.
    # @param ranks MPI process count recorded in every manifest row.
    # @return Parsed snapshot manifest as a DataFrame.
    def readManifest(self, directory, ranks):
        table = pd.read_csv(directory / "snapshots.csv", float_precision="round_trip")
        require(list(table.columns) == ["name", "step", "a", "z", "ranks", "format"],
                "Unexpected snapshots.csv schema")
        require(len(table) >= 2 and table.name.iloc[0] == "initial"
                and table.name.iloc[-1] == "final", "Manifest lacks initial/final snapshots")
        require(not table.name.duplicated().any(), "Repeated snapshot name")
        require(np.isfinite(table[["step", "a", "z", "ranks"]].to_numpy()).all(),
                "Nonfinite snapshot metadata")
        require((table.ranks == ranks).all() and (table["format"] == "binary").all(),
                "Incorrect shard count or snapshot format in manifest")
        require(table.step.iloc[0] == 0 and (np.diff(table.step) > 0).all(),
                "Snapshots do not correspond to strictly increasing complete KDK steps")
        require((np.diff(table.a) > 0).all(), "Repeated or decreasing output scale factor")
        np.testing.assert_allclose(table.z, 1 / table.a - 1, rtol=0, atol=3e-14)
        diagnostics = pd.read_csv(directory / "diagnostics.csv", float_precision="round_trip")
        require(np.isfinite(diagnostics.mass_error).all()
                and (diagnostics.mass_error.abs() <= PhaseTolerance).all(),
                "Mass error exceeds the existing diagnostic regression budget")
        return table

    ## @brief Evaluate a nonuniform one-step EdS/CIC/PM oracle independent of input/output code.
    # @param phase ID-sorted initial canonical phase space.
    # @param mesh Force grid NM per dimension.
    # @param scale Initial dimensionless a.
    # @param finalScale Final dimensionless a.
    # @return Synchronized canonical phase space after one KDK interval.
    def oneStep(self, phase, mesh, scale=.1, finalScale=.11):
        meanMass = len(self.ids) / mesh**3

        def force(positions):
            delta = oracle.deposit(positions, mesh, BoxSize) / meanMass
            return oracle.gather(oracle.mesh_force(delta, BoxSize, 1.0), positions, BoxSize)

        midpoint = math.sqrt(scale * finalScale)
        halfP = phase[:, 3:] + 2 * (math.sqrt(midpoint) - math.sqrt(scale)) * force(phase[:, :3])
        positions = oracle.wrap(phase[:, :3] + 2 * (1 / math.sqrt(scale)
                                                  - 1 / math.sqrt(finalScale)) * halfP, BoxSize)
        momentum = halfP + 2 * (math.sqrt(finalScale) - math.sqrt(midpoint)) * force(positions)
        return np.column_stack((positions, momentum))

    ## @brief Exercise split Gadget conversion, a frozen run plan, execution, and spectral analysis.
    # The independent Gadget fixture encodes u=v_pec/sqrt(a), whereas its expected
    # canonical state is p=a*v_pec/100. Power self-comparisons require P/P=1 and
    # r=P_cross/sqrt(P_left*P_right)=1 wherever both raw auto powers are nonzero.
    # @return None; retained artifacts and assertions verify every pipeline boundary.
    def pipeline(self):
        pipelineDirectory = self.directory / "pipeline"
        pipelineDirectory.mkdir()
        phase = np.column_stack((self.positions.copy(),
                                 .02 * np.cos(2 * np.pi * self.positions / BoxSize)))
        phase[:, 0] -= .08 * np.sin(2 * np.pi * self.positions[:, 0] / BoxSize)
        scale = .25
        physicalVelocity = 100 * phase[:, 3:] / scale
        rawVelocity = physicalVelocity / math.sqrt(scale)
        rawPosition = 1000 * phase[:, :3]
        permutation = np.random.default_rng(7719).permutation(len(self.ids))
        for split, indices in enumerate(np.array_split(permutation, 2)):
            writeGadgetFixture(pipelineDirectory / f"ics.{split}", ids=self.ids[indices] + 1,
                              total=len(self.ids), files=2, floatSize=8, idSize=8, a=scale,
                              changes=[(128, "d", (1000 * BoxSize,))],
                              positions=rawPosition[indices], velocities=rawVelocity[indices])
        canonical = pipelineDirectory / "initial.bin"
        conversion = quijoteIO.convert(pipelineDirectory / "ics", canonical, chunk_size=37,
                                       expected={"total_count": len(self.ids), "a": scale,
                                                 "box_mpc_h": BoxSize, "omega_m": .3175})
        require(len(conversion["sources"]) == 2, "Split Gadget inputs lost conversion provenance")
        require(conversion["validation"]["unique_complete_ids"], "Conversion lacks exact ID validation")
        converted = np.concatenate(list(quijoteIO.iter_records(canonical, chunk_size=29)))
        np.testing.assert_array_equal(converted["id"], self.ids)
        np.testing.assert_array_equal(converted["position"], rawPosition / 1000)
        np.testing.assert_allclose(converted["momentum"], scale * physicalVelocity / 100,
                                   rtol=2e-15, atol=0)

        # Adapt the suite's launcher options to the runner's frozen executable/-n contract.
        launcher = pipelineDirectory / "mpiexec.py"
        prefix = shlex.split(self.arguments.mpiexec) + self.arguments.mpi_arg
        launcher.write_text(f"#!{sys.executable}\nimport os, sys\n"
                            f"prefix = {prefix!r}\n"
                            "assert sys.argv[1] == '-n'\n"
                            f"os.execvp(prefix[0], prefix + [{self.arguments.numproc_flag!r}] + sys.argv[2:])\n")
        launcher.chmod(0o755)
        preparedDirectory = pipelineDirectory / "prepared"
        arguments = benchmark.parser().parse_args([
            "prepare", "--ic", str(canonical), "--exe", str(self.arguments.exe),
            "--output-dir", str(preparedDirectory), "--mesh", "16", "--steps", "3",
            "--ranks", "2", "--mpiexec", str(launcher), "--redshifts", "1", ".5", "0",
            "--diagnostics-every", "1", "--chunk-size", "31", "--disk-reserve-gib", "0",
            "--allow-nonfiducial"])
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "2", "OMP_PROC_BIND": "false"}):
            prepared = benchmark.prepare(arguments)
        manifestPath = preparedDirectory / "benchmark.json"
        frozen = json.loads(manifestPath.read_text())
        require(prepared == frozen and frozen["state"] == "prepared" and not frozen["executed"],
                "Dry-run preparation did not retain a frozen, unexecuted plan")
        require(not (preparedDirectory / "run").exists(), "Prepare dry run launched the simulation")
        require(frozen["converter"]["manifest"] == conversion, "Prepared plan lost converter provenance")
        require(frozen["config"]["np"] == 16 and frozen["config"]["particle_count"] == 512,
                "Prepared plan coupled force mesh to the imported particle count")
        require(frozen["environment"]["OMP_NUM_THREADS"] == "2", "Prepared plan lost OpenMP environment")
        require(self.launch("pipeline-execute", [sys.executable, benchmark.__file__, "execute",
                                                 "--manifest", manifestPath, "--run"],
                            ranks=None, threads=2), "Frozen pipeline execution failed")
        completed = json.loads(manifestPath.read_text())
        require(completed["state"] == "completed" and completed["executed"],
                "Runner failed to audit successful output completion")
        for key in ("ic", "executable", "config", "config_sha256", "command", "converter"):
            require(completed[key] == frozen[key], f"Execution changed frozen plan field {key}")
        np.testing.assert_allclose([row["a"] for row in completed["outputs"]["epochs"]],
                                   [scale, .5, 2 / 3, 1.], rtol=0, atol=2e-15)
        require(all(len(row["shards"]) == 2 for row in completed["outputs"]["epochs"]),
                "Completed output audit did not validate every MPI shard")
        outputDirectory = Path(completed["config"]["output"])
        metadata = dict(line.split("=", 1) for line in
                        (outputDirectory / "metadata.txt").read_text().splitlines() if "=" in line)
        require(int(metadata["ranks"]) == 2, "Pipeline did not use MPI2")
        if metadata["execution_space"] == "OpenMP":
            require(int(metadata["threads"]) == 2 and int(metadata["host_threads"]) == 2,
                    "OpenMP pipeline did not use two threads per rank")
        manifest = outputDirectory / "snapshots.csv"
        paths, initialScale = analysis.resolve_sources([str(manifest)], "initial")
        initialSource = analysis.ParticleSource(paths, chunk_size=19, expected_a=initialScale)
        initial = np.concatenate(list(initialSource.records()))
        initial = initial[np.argsort(initial["id"])]
        np.testing.assert_array_equal(initial, converted)
        for lineOfSight in (None, "z"):
            name = "real" if lineOfSight is None else "rsd"
            reportPath = pipelineDirectory / f"self-compare-{name}.json"
            compareArguments = ["compare", "--left", str(manifest), "--left-epoch", "final",
                                "--right", str(manifest), "--right-epoch", "final",
                                "--grid", "16", "--k-min", ".5", "--k-max", "3", "--dk", ".5",
                                "--chunk-size", "23", "--workers", "2", "--memory-limit-gib", "1",
                                "--output", str(reportPath)]
            if lineOfSight is not None:
                compareArguments += ["--line-of-sight", lineOfSight]
            report = benchmark.compare_command(benchmark.parser().parse_args(compareArguments))
            require(json.loads(reportPath.read_text()) == report, "Comparison report did not roundtrip")
            json.dumps(report, allow_nan=False)
            rows = [row for row in report["rows"] if row["left_raw"] > 0 and row["right_raw"] > 0]
            require(rows, "Pipeline self-comparison has no nonzero power shells")
            np.testing.assert_allclose([row["ratio"] for row in rows], 1., atol=PhaseTolerance, rtol=0)
            np.testing.assert_allclose([row["correlation_raw"] for row in rows], 1.,
                                       atol=PhaseTolerance, rtol=0)
            require(report["line_of_sight"] == (None if lineOfSight is None else 2),
                    "Comparison report lost real/redshift-space selection")
            require(np.max(np.abs(report["mass_relative_errors"])) <= PhaseTolerance,
                    "Pipeline spectrum deposition failed mass conservation")

    ## @brief Run roundtrip, oracle, epoch, pipeline, rejection, and optional adapter-equivalence tests.
    # @return Zero exactly when every declared run and invariant succeeds.
    def run(self):
        phase = np.column_stack((self.positions.copy(),
                                 .03 * np.cos(2 * np.pi * self.positions / BoxSize)))
        phase[:, 0] -= .1 * np.sin(2 * np.pi * self.positions[:, 0] / BoxSize)
        # The allowed upper endpoint must wrap exactly, without changing p or the ID.
        phase[0, 0] = BoxSize
        fixture = self.fixture("wave", phase)
        wrappedPhase = phase.copy()
        wrappedPhase[:, :3] = oracle.wrap(wrappedPhase[:, :3], BoxSize)
        expected = self.oneStep(wrappedPhase, 16)
        waveRuns = []
        for ranks in (1, 2, 4):
            okay, output = self.evolve(f"wave-np8-nm16-r{ranks}", fixture, ranks=ranks)
            waveRuns.append((okay, output, ranks))

            def checkWave(okay=okay, output=output, ranks=ranks):
                require(okay, "Missing successful wave run")
                table = self.readManifest(output, ranks)
                require(list(table.name) == ["initial", "final"], "Unexpected wave snapshots")
                np.testing.assert_array_equal(self.readSnapshot(output, "initial", ranks, .1), wrappedPhase)
                final = self.readSnapshot(output, "final", ranks, .11)
                np.testing.assert_allclose(phaseDifference(final, expected), 0,
                                           atol=PhaseTolerance, rtol=0)
                diagnostics = pd.read_csv(output / "diagnostics.csv", float_precision="round_trip")
                np.testing.assert_array_equal(diagnostics.step, [0, 1])
                for stateIndex, state in enumerate((wrappedPhase, final)):
                    delta = oracle.deposit(state[:, :3], 16, BoxSize) / (len(self.ids) / 16**3)
                    expectedRms = math.sqrt(np.mean(delta**2))
                    np.testing.assert_allclose(diagnostics.delta_rms.iloc[stateIndex], expectedRms,
                                               atol=PhaseTolerance, rtol=0)
            self.check(f"MPI{ranks}: exact imported state, NP8/NM16 KDK oracle, and mesh RMS", checkWave)

        def checkRanks():
            require(all(run[0] for run in waveRuns), "Missing successful wave rank comparison")
            reference = self.readSnapshot(waveRuns[0][1], "final", 1, .11)
            for _, output, ranks in waveRuns[1:]:
                final = self.readSnapshot(output, "final", ranks, .11)
                np.testing.assert_allclose(phaseDifference(final, reference), 0,
                                           atol=PhaseTolerance, rtol=0)
        self.check("Nonuniform MPI1/2/4 phase-space equivalence", checkRanks)

        if self.arguments.reference_exe:
            csvFixture = self.directory / "fixtures" / "wave.csv"
            frame = pd.DataFrame(wrappedPhase, columns=["x", "y", "z", "px", "py", "pz"])
            frame.insert(0, "id", self.ids)
            frame["mass"] = 1
            frame.to_csv(csvFixture, index=False, float_format="%.17g")
            referenceDirectory = self.directory / "csv-adapter"
            referenceOkay = self.launch("csv-adapter", [self.arguments.reference_exe,
                ParticleGrid, 16, BoxSize, 1.0, .1, .11, 1, 1, csvFixture, referenceDirectory], 1)

            def checkReference():
                require(referenceOkay and waveRuns[0][0], "Missing adapter/production comparison run")
                reference = pd.read_csv(referenceDirectory / "particles_checkpoint0001_rank0.csv",
                                        float_precision="round_trip").sort_values("id")
                np.testing.assert_array_equal(reference.id, self.ids)
                state = reference[["x", "y", "z", "px", "py", "pz"]].to_numpy()
                final = self.readSnapshot(waveRuns[0][1], "final", 1, .11)
                np.testing.assert_allclose(phaseDifference(final, state), 0,
                                           atol=PhaseTolerance, rtol=0)
            self.check("Production binary and existing CSV evolution adapter agree", checkReference)

        momentum = np.broadcast_to([.125, -.25, .375], self.positions.shape)
        ballistic = np.column_stack((self.positions, momentum))
        fixture = self.fixture("ballistic", ballistic, scale=.25)
        for ranks in (1, 4):
            okay, output = self.evolve(f"scheduled-ballistic-r{ranks}", fixture, ranks=ranks,
                mesh=8, scale=.25, finalScale=1., steps=3, outputs=[1., .5, 0.])

            def checkSchedule(okay=okay, output=output, ranks=ranks):
                require(okay, "Missing successful scheduled ballistic run")
                table = self.readManifest(output, ranks)
                np.testing.assert_allclose(table.a, [.25, .5, 2 / 3, 1.], rtol=0, atol=2e-15)
                for row in table.itertuples(index=False):
                    state = self.readSnapshot(output, row.name, ranks, row.a)
                    drift = 2 * (1 / math.sqrt(.25) - 1 / math.sqrt(row.a))
                    expectedState = np.column_stack((oracle.wrap(self.positions + momentum * drift,
                                                                BoxSize), momentum))
                    np.testing.assert_allclose(phaseDifference(state, expectedState), 0,
                                               atol=PhaseTolerance, rtol=0)
            self.check(f"MPI{ranks}: exact requested epochs and synchronized ballistic momentum", checkSchedule)

        # Header, ID and record corruption must fail collectively before evolution.
        original = self.fixture("rejection-baseline", phase).read_bytes()
        corruptions = []
        changed = bytearray(original)
        struct.pack_into("<Q", changed, 8, len(self.ids) - 1)
        corruptions.append(("bad-file-count", changed, ("count", "size", "length", "header")))
        changed = bytearray(original)
        struct.pack_into("<Q", changed, 16, len(self.ids) + 1)
        corruptions.append(("bad-global-count", changed, ("count", "particle", "header")))
        changed = bytearray(original)
        struct.pack_into("<d", changed, 40, .99)
        corruptions.append(("bad-cosmology", changed, ("cosmolog", "omega", "header")))
        changed = bytearray(original)
        struct.pack_into("<d", changed, 24, .2)
        corruptions.append(("bad-epoch", changed, ("epoch", "scale", "header")))
        corruptions.append(("truncated-input", original[:-16], ("truncat", "size", "length", "read")))
        changed = bytearray(original)
        struct.pack_into("<Q", changed, Header.size + RecordType.itemsize, 0)
        corruptions.append(("duplicate-id", changed, ("id", "sorted", "contiguous")))
        changed = bytearray(original)
        struct.pack_into("<d", changed, Header.size + 8 + 3 * 8, math.nan)
        corruptions.append(("nonfinite-momentum", changed, ("finite", "momentum", "phase")))
        for name, data, markers in corruptions:
            path = self.directory / "fixtures" / f"{name}.bin"
            path.write_bytes(data)
            okay, output = self.evolve(name, path, ranks=2, reject=markers)

            def checkRejection(okay=okay, output=output):
                require(okay, "Malformed binary input did not fail cleanly")
                require(not list(output.glob("particles_final_rank*.bin")),
                        "Malformed input produced a final particle state")
            self.check(f"Collective input rejection: {name}", checkRejection)

        ## @cond RUNTIME_CALLBACK
        self.check("Split Gadget to frozen MPI2/OpenMP2 run and real/RSD comparison pipeline", self.pipeline)
        ## @endcond
        self.results["passed"] = (all(row["passed"] for row in self.results["runs"])
                                  and all(row["passed"] for row in self.results["checks"]))
        self.save()
        print(f"{'PASS' if self.results['passed'] else 'FAIL'}: {len(self.results['runs'])} runs; "
              f"{self.directory / 'results.json'}", flush=True)
        return 0 if self.results["passed"] else 1


## @brief Parse the MPI test interface and retain setup errors as a nonzero exit status.
# @return Zero if all tests pass; one for failed checks; two for invalid setup/data.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--reference-exe", type=Path)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--mpi-arg", action="append", default=[])
    parser.add_argument("--numproc-flag", default="-n")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=60.)
    arguments = parser.parse_args()
    for executable in (arguments.exe, arguments.reference_exe):
        if executable is not None and (not executable.is_file() or not os.access(executable, os.X_OK)):
            parser.error(f"Missing executable: {executable}")
    arguments.exe = arguments.exe.resolve()
    if arguments.reference_exe:
        arguments.reference_exe = arguments.reference_exe.resolve()
    if not math.isfinite(arguments.timeout) or arguments.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    try:
        return ExternalCosmologyTests(arguments).run()
    except (ValueError, OSError, AssertionError) as error:
        print(f"External cosmology setup/data failure: {error}", file=sys.stderr)
        return 2


## @cond CLI_DISPATCH
if __name__ == "__main__":
    raise SystemExit(main())
## @endcond
