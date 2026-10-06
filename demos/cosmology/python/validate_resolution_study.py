#!/usr/bin/env python3
## @file validate_resolution_study.py
# @brief Disk-guarded crossed-resolution and common-phase Gaussian PM studies.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
"""Disk-guarded crossed-resolution and common-phase Gaussian PM studies.

Existing production executables, reference operators and earlier qualification
scripts are unchanged. Completion, numerical acceptance, and a qualified band
are separate outcomes. All candidate bands and budgets precede execution.
"""
from __future__ import annotations

from source_paths import source_path

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

import validate_evolution as baseline
from gaussian_fixture import make_gaussian_fixture
from study_spectra import analyze_spectrum, unique_modes, ShellEdges
from study_storage import StudyStorage, DiskSpaceBlocked, GiB, sha256
from runtime_metadata import validate_runtime_metadata


## @var Parameters
# @brief Named Parameters protocol/schema value; the source initializer records its exact contents.
Parameters = {"box_size": 168.75, "omega_m": .31, "checkpoints": 8,
              "spatial_ai": .02, "spatial_af": .2, "gaussian_af": 1.,
              "seed": 20261003, "gaussian_cutoff": 12, "analysis_grid": 128,
              "qualified_max_mode": 4, "shift_finest_cell_fraction": [.37, .23, .41]}
## @var Budgets
# @brief Named Budgets protocol/schema value; the source initializer records its exact contents.
Budgets = {"cross": {32: {"power": .02, "complex": .02, "correlation": .999},
                      64: {"power": .01, "complex": .01, "correlation": .9995}},
           "spatial": {"power": .05, "complex": .05, "correlation": .999},
           "translation": {"power": .01, "complex": .02, "correlation": .9998},
           "start_redshift": {"power": .02, "complex": .03, "correlation": .999},
           "time_shell_complex": .002, "shell_edges": list(ShellEdges),
           "inherited": baseline.Limits}
## @var Codes
# @brief Named Codes protocol/schema value; the source initializer records its exact contents.
Codes = ("ippl", "fastpm")
## @var SourceNames
# @brief Named SourceNames protocol/schema value; the source initializer records its exact contents.
SourceNames = ("source_paths.py", "validate_resolution_study.py", "study_storage.py", "study_spectra.py",
               "gaussian_fixture.py", "validate_evolution.py", "validate_linear.py",
               "CosmologySimulation.h", "CosmologyPhysics.h", "CosmologyConfig.h",
               "ExecutionMetadata.h", "runtime_metadata.py",
               "tests/CompareCosmologyEvolution.cpp", "reference/FastPMEvolution.c",
               "reference/build_fastpm_evolution.sh")


## @brief Convert the protocol object to its stable JSON-comparable representation.
# @see cosmology_tools
#
# @param value Measured or serialized scalar in the declared metric/schema; no normalization is inferred.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


## @brief Immutable fixture/resolution/schedule descriptor used to name and select validation runs.
# @see cosmology_tools
@dataclass(frozen=True)
class Case:
    ## @var stage
    # @brief Declared campaign stage (for example spatial or Gaussian), not a inferred favorable subset.
    stage: str
    ## @var fixture
    # @brief Shared initial-condition data or its explicit case descriptor; it must remain consistent across compared solvers.
    fixture: str
    ## @var particles
    # @brief Expected exact global particle count, not a floating-point mass sum.
    particles: int
    ## @var mesh
    # @brief Force-mesh size per dimension for normalizing particle displacements to cell widths.
    mesh: int
    ## @var steps
    # @brief Positive PM step count; endpoints are uniform in log(a) for the imported drivers.
    steps: int
    ## @var ranks
    # @brief Positive MPI rank count; all expected snapshot shards must exist.
    ranks: int = 1
    ## @var shifted
    # @brief Named shifted protocol/schema value; the source initializer records its exact contents.
    shifted: bool = False
    ## @var redshift
    # @brief Initialization redshift; a=1/(1+redshift).
    redshift: int = 49

    ## @brief Return the stable protocol case name used for artifact identity.
    # @see cosmology_tools
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
    @property
    def name(self):
        shift = "shift" if self.shifted else "base"
        return f"{self.fixture}_p{self.particles}_m{self.mesh}_t{self.steps}_r{self.ranks}_z{self.redshift}_{shift}"

    ## @brief Return the key selecting the exact shared fixture for this case.
    # @see cosmology_tools
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
    @property
    def input_key(self):
        return f"{self.fixture}_p{self.particles}_z{self.redshift}_{'shift' if self.shifted else 'base'}"


## @brief Evaluate the study cases helper in the documented module workflow.
# @see cosmology_tools
#
# @param stage Declared campaign stage (for example spatial or Gaussian), not a inferred favorable subset.
# @param smoke Select engineering smoke parameters; not a full science/accuracy qualification.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def study_cases(stage="all", smoke=False):
    if stage not in ("all", "spatial", "gaussian"):
        raise ValueError("Unknown study stage")
    cases = []
    if stage in ("all", "spatial"):
        if smoke:
            cases += [Case("spatial", "pancake", 8, 8, 16, rank) for rank in (1, 2)]
        else:
            for fixture in ("pancake", "coupled3d"):
                cases += [Case("spatial", fixture, n, m, 1024) for n in (32, 64) for m in (32, 64)]
                cases += [Case("spatial", fixture, 64, m, 1024, shifted=True) for m in (32, 64)]
                cases += [Case("spatial", fixture, 64, 64, steps) for steps in (512, 2048)]
            cases.append(Case("spatial", "coupled3d", 64, 64, 1024, 3))
    if stage in ("all", "gaussian"):
        if smoke:
            cases += [Case("gaussian", "gaussian", 16, 16, 16, rank, redshift=99) for rank in (1, 2)]
        else:
            cases += [Case("gaussian", "gaussian", n, m, 2048) for n in (32, 64) for m in (32, 64)]
            cases += [Case("gaussian", "gaussian", 64, 64, steps) for steps in (1024, 4096)]
            cases += [Case("gaussian", "gaussian", 64, 64, steps, redshift=99) for steps in (2048, 4096)]
            cases.append(Case("gaussian", "gaussian", 64, 64, 2048, 4))
    if len({case.name for case in cases}) != len(cases):
        raise ValueError("Duplicate study cases")
    return cases


## @brief Evaluate the epochs helper in the documented module workflow.
# @see cosmology_tools
#
# @param case Immutable case specification identifying the fixture, resolutions and schedule.
# @param smoke Select engineering smoke parameters; not a full science/accuracy qualification.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def epochs(case, smoke=False):
    if case.stage == "spatial":
        return Parameters["spatial_ai"], .04 if smoke else Parameters["spatial_af"]
    return 1 / (1 + case.redshift), .02 if smoke else Parameters["gaussian_af"]


## @brief Evaluate the translation helper in the documented module workflow.
# @see cosmology_tools
#
# @param case Immutable case specification identifying the fixture, resolutions and schedule.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def translation(case):
    return (np.asarray(Parameters["shift_finest_cell_fraction"]) * Parameters["box_size"] / 64
            if case.shifted else np.zeros(3))


## @brief Encode measured complex Fourier coefficients for a retained JSON record.
# @see cosmology_tools
#
# @param coefficients Complex dimensionless density Fourier coefficients with the module's declared normalization.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def encode_modes(coefficients):
    return {"real": coefficients.real.tolist(), "imag": coefficients.imag.tolist()}


## @brief Recover the retained complex Fourier arrays from their explicit real/imaginary encoding.
# @see cosmology_tools
#
# @param record Structured retained evidence record following this module's declared schema.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def decode_modes(record):
    return np.asarray(record["real"]) + 1j * np.asarray(record["imag"])


## @brief Every predeclared low shell, with undefined signal explicit, never floored.
# @see cosmology_tools
#
# @param left First ID-matched state or Fourier array; the metric specifies denominator and normalization.
# @param right Second ID-matched state or Fourier array; the metric specifies denominator and normalization.
# @param modes Signed integer Fourier-mode array with three Cartesian components per mode.
# @param budget Predeclared comparison tolerances; they are not fitted after observing results.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def shell_comparisons(left, right, modes, budget):
    """Every predeclared low shell, with undefined signal explicit, never floored."""
    radius = np.linalg.norm(modes, axis=1)
    rows = []
    for index, (lower, upper) in enumerate(zip(ShellEdges[:-1], ShellEdges[1:])):
        mask = (radius >= lower) & (radius < upper)
        if not mask.any():
            continue
        metrics = baseline.density_comparison(left[mask], right[mask])
        defined = metrics["normalization_defined"]
        passed = bool(defined and abs(metrics["power_ratio"] - 1) <= budget["power"]
                      and metrics["complex_relative"] <= budget["complex"]
                      and metrics["correlation"] >= budget["correlation"])
        rows.append({"shell": index, "lower": lower, "upper": upper,
                     "maximum_mode": float(radius[mask].max()), "pairs": int(mask.sum()),
                     "metrics": metrics, "budget": budget, "passed": passed})
    return rows


## @brief Largest contiguous prefix; a failed lower shell blocks all higher ones.
# @see cosmology_tools
#
# @param checks Complete set of named fixed-budget checks and their recorded pass/fail evidence.
# @param stage Declared campaign stage (for example spatial or Gaussian), not a inferred favorable subset.
# @param fixture Shared initial-condition data or its explicit case descriptor; it must remain consistent across compared solvers.
# @param required_passed Names/categories that every qualified shell must satisfy; no retrospective subset is accepted.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def qualified_prefix(checks, stage, fixture, required_passed):
    """Largest contiguous prefix; a failed lower shell blocks all higher ones."""
    rows = []
    for shell in range(3):
        applicable = [check for check in checks if check.get("stage") == stage
                      and check.get("fixture") == fixture and check.get("shell") == shell]
        passed = bool(required_passed and applicable and all(check["passed"] for check in applicable))
        rows.append({"shell": shell, "passed": passed, "checks": len(applicable),
                     "failed_checks": [check["name"] for check in applicable if not check["passed"]]})
    count = 0
    for row in rows:
        if not row["passed"]:
            break
        count += 1
    return {"shells": rows, "contiguous_shell_count": count,
            "upper_mode_exclusive": float(ShellEdges[count]) if count else None,
            "hard_maximum_measured_mode": 4,
            "scope": "Engineering robustness over tested cases, not continuum or truth error"}


## @brief Check the pinned native reference executable and build provenance before launching a comparison.
# @see cosmology_tools
#
# @param path Input/output filesystem path; freshness and hash requirements are defined by this routine.
# @param executable Selected native/application executable path; its bytes and declared build provenance must match.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def verify_native_manifest(path, executable):
    text = path.read_text()
    if f"upstream_commit={baseline.FastPMCommit}" not in text:
        raise ValueError("Wrong native reference pin")
    hashes = {}
    for line in text.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and len(parts[0]) == 64 and all(c in "0123456789abcdef" for c in parts[0]):
            target = Path(parts[1].lstrip(" *")).resolve()
            if sha256(target) != parts[0]:
                raise ValueError(f"Native manifest artifact mismatch: {target}")
            hashes[str(target)] = parts[0]
    if len(hashes) < 16 or str(executable) not in hashes:
        raise ValueError("Native manifest lacks required artifact coverage")
    return hashes


## @brief Controller for independently varied particle/mesh resolutions and the complete required qualification controls.
# @see cosmology_tools
class Study:
    ## @brief Initialize the instance with the explicit protocol and artifact ownership passed by the caller.
    # @see cosmology_tools
    #
    # @param args Parsed command-line options; see main/--help and the module workflow contract.
    def __init__(self, args):
        ## @var args
        # @brief Parsed command-line options; see main/--help and the module workflow contract.
        self.args = args
        source = Path(__file__).resolve().parent
        if args.resume:
            ## @var root
            # @brief Campaign or artifact root following this module's ownership contract.
            self.root = args.resume.resolve()
            ## @var report
            # @brief Structured campaign/audit report; recorded failures are not retroactively changed.
            self.report = json.loads((self.root / "results.json").read_text())
            if (self.report.get("schema") != "ippl-resolution-study-v1"
                    or self.report["parameters"] != canonical(Parameters)
                    or self.report["budgets"] != canonical(Budgets)):
                raise ValueError("Resume protocol differs")
            config = self.report["configuration"]
            if args.stage is not None and args.stage != config["stage"]:
                raise ValueError("Resume stage differs")
            if args.smoke and not config["smoke"]:
                raise ValueError("Cannot request smoke while resuming a full study")
            for key in ("ippl_exe", "fastpm_exe", "fastpm_manifest"):
                requested = getattr(args, key)
                if requested is not None and str(requested.resolve()) != config[key]:
                    raise ValueError("Resume executable or manifest differs")
                setattr(args, key, Path(config[key]))
            args.stage, args.smoke = config["stage"], config["smoke"]
            # Launcher choices are part of the recorded execution contract.
            if args.mpiexec != config["mpiexec"] or args.numproc_flag != config["numproc_flag"] or args.mpi_arg != config["mpi_arg"]:
                raise ValueError("Resume launcher differs")
            ## @var hashes
            # @brief Absolute source/artifact paths mapped to expected SHA256 values; changed bytes invalidate provenance.
            self.hashes = self.report["provenance"]
            for path, digest in self.hashes.items():
                if sha256(path) != digest:
                    raise ValueError(f"Resume source/input/artifact differs: {path}")
        else:
            if any(getattr(args, key) is None for key in ("ippl_exe", "fastpm_exe", "fastpm_manifest")):
                raise ValueError("New study requires both executables and the native manifest")
            args.stage = args.stage or "all"
            for key in ("ippl_exe", "fastpm_exe", "fastpm_manifest"):
                setattr(args, key, getattr(args, key).resolve())
            self.root = args.output_dir.resolve() if args.output_dir else Path(tempfile.mkdtemp(
                prefix="resolution-study-", dir=args.ippl_exe.parent))
            self.root.mkdir(parents=True, exist_ok=True)
            if any(self.root.iterdir()):
                raise ValueError("New study directory must be empty")
            self.hashes = {str(source_path(name)): sha256(source_path(name)) for name in SourceNames}
            self.hashes.update({str(path): sha256(path) for path in (args.ippl_exe, args.fastpm_exe, args.fastpm_manifest)})
            self.hashes.update(verify_native_manifest(args.fastpm_manifest, args.fastpm_exe))
            config = {key: str(getattr(args, key)) for key in ("ippl_exe", "fastpm_exe", "fastpm_manifest")}
            config.update(stage=args.stage, smoke=args.smoke, mpiexec=args.mpiexec,
                          numproc_flag=args.numproc_flag, mpi_arg=args.mpi_arg)
            self.report = {"schema": "ippl-resolution-study-v1", "parameters": Parameters,
                "budgets": Budgets, "configuration": config, "provenance": self.hashes,
                "started_utc": datetime.now(timezone.utc).isoformat(), "fixtures": {}, "runs": [],
                "checks": [], "comparisons": [], "time_refinement": [], "qualification": {},
                "smoke_overrides": ({"spatial_af": .04, "gaussian_af": .02, "gaussian_cutoff": 6}
                                    if args.smoke else None),
                "state": "prepared", "complete": False, "passed": False,
                "limitations": ["Finite-resolution studies with recorded backend/rank metadata, not halo-statistics, continuum or exascale qualification",
                    "Gaussian initial modes fixed at |n|<=12; one seed, no variance fitting",
                    "Pure BBKS omits baryonic transfer features; radiation-free flat Lambda, 1LPT",
                    "Qualified comparisons capped at direct Fourier |n|<=4; higher modes characterized only",
                    "Earlier accepted discrepancies and original failed campaigns remain unchanged",
                    "Storage preserves exact parsed numerical values, not original CSV bytes"]}
        ## @var cases
        # @brief Declared case matrix; missing or repeated cases cannot be substituted for completion.
        self.cases = study_cases(args.stage, args.smoke)
        expected = [asdict(case) for case in self.cases]
        if args.resume and self.report["planned_cases"] != expected:
            raise ValueError("Resume case matrix changed")
        self.report["planned_cases"] = expected
        self.report["expected_runs"] = 2 * len(self.cases)
        if not args.resume:
            # Commit a coherent prepared report before initializing storage. A
            # stopped initialization with absent/empty storage can be resumed.
            self.save()
        storage_root = self.root / "storage"
        storage_resume = bool(args.resume)
        if args.resume and not (storage_root / "storage.json").exists():
            if self.report["state"] != "prepared" or (storage_root.exists() and any(storage_root.iterdir())):
                raise ValueError("Incomplete storage initialization retained; inspect before recovery")
            storage_resume = False
        ## @var storage
        # @brief Requested numerical-archive storage helper module; completion is separate from acceptance.
        self.storage = StudyStorage(storage_root, self.hashes,
                                    reserve_bytes=int(args.reserve_gib * GiB), resume=storage_resume)
        if args.resume:
            # The storage lock serializes both journals. Reload a report which
            # another owner may have finished while this process acquired it.
            current = json.loads((self.root / "results.json").read_text())
            for key in ("provenance", "configuration", "parameters", "budgets", "planned_cases"):
                if current[key] != self.report[key]:
                    self.storage.close()
                    raise ValueError("Study identity changed while acquiring the storage lock")
            self.report = current
        source_copies = {}
        for name in SourceNames:
            destination = self.root / "source-snapshot" / name
            if not destination.exists():
                if self.report["state"] != "prepared":
                    raise ValueError("Recorded source snapshot is missing")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path(name), destination)
            expected_hash = self.hashes[str(source_path(name))]
            if sha256(destination) != expected_hash:
                raise ValueError("Source snapshot differs from frozen provenance")
            source_copies[str(destination)] = expected_hash
        self.report["source_snapshot"] = source_copies
        ## @var fixtures
        # @brief Retained fixtures state owned by this instance; see the initialization and workflow contract.
        self.fixtures = {}
        ## @var executables
        # @brief Retained executables state owned by this instance; see the initialization and workflow contract.
        self.executables = {code: getattr(args, code + "_exe") for code in Codes}
        self.save()

    ## @brief Persist the complete current report without discarding recorded failed checks.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def save(self):
        self.report["failed_checks"] = [check for check in self.report["checks"] if not check["passed"]]
        temporary = self.root / "results.json.partial"
        with temporary.open("w") as stream:
            json.dump(self.report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.root / "results.json")

    ## @brief Record the named predeclared check and its diagnostic values.
    # @see cosmology_tools
    #
    # @param name Stable artifact/run/check identifier as defined by the caller.
    # @param passed Boolean result of the predeclared check, not of merely completing execution.
    # @param case Immutable case specification identifying the fixture, resolutions and schedule.
    # @param category Integrity, measurement, cross-code or refinement category used in the saved report.
    # @param values Recorded diagnostic values in the metric/schema defined by the caller.
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def check(self, name, passed, *, case=None, category="integrity", **values):
        row = {"name": name, "passed": bool(passed), "category": category, **values}
        if case is not None:
            row.update(stage=case.stage, fixture=case.fixture)
        self.report["checks"].append(row)

    ## @brief Record whether a measured value stays within its explicit upper budget.
    # @see cosmology_tools
    #
    # @param name Stable artifact/run/check identifier as defined by the caller.
    # @param value Measured or serialized scalar in the declared metric/schema; no normalization is inferred.
    # @param limit Predeclared acceptance limit in the reported metric's units.
    # @param case Immutable case specification identifying the fixture, resolutions and schedule.
    # @param category Integrity, measurement, cross-code or refinement category used in the saved report.
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def upper(self, name, value, limit, *, case, category="integrity"):
        self.check(name, value is not None and np.isfinite(value) and value <= limit,
                   case=case, category=category, value=value, limit=limit)

    ## @brief Construct the fixture for the documented module workflow.
    # @see cosmology_tools
    #
    # @param case Immutable case specification identifying the fixture, resolutions and schedule.
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
    def fixture(self, case):
        key = case.input_key
        if key in self.fixtures:
            return self.fixtures[key]
        prior = self.report["fixtures"].get(key)
        if prior is not None:
            path = Path(prior["path"])
            if sha256(path) != prior["sha256"]:
                raise ValueError("Shared input changed")
            frame = pd.read_csv(path, dtype={"id": np.uint64}, float_precision="round_trip").sort_values("id").reset_index(drop=True)
            baseline.validate_snapshot(frame)
        else:
            free = shutil.disk_usage(self.root).free
            required = int(self.args.reserve_gib * GiB) + case.particles**3 * 240 + 16 * 1024**2
            if free < required:
                raise DiskSpaceBlocked(f"Need {required} free bytes before preparing input; have {free}")
            if case.stage == "gaussian":
                frame, description = make_gaussian_fixture(case.particles, case.redshift, Parameters["seed"],
                    cutoff=6 if self.args.smoke else Parameters["gaussian_cutoff"],
                    cosmology={"Omega_m": Parameters["omega_m"], "box_size": Parameters["box_size"]})
            else:
                frame, description = baseline.make_fixture(case.fixture, case.particles)
            shift = translation(case)
            if case.shifted:
                frame[["x", "y", "z"]] = (frame[["x", "y", "z"]] + shift) % Parameters["box_size"]
            path = self.root / (key + ".csv")
            # Interrupted input writes are never mistaken for a completed input.
            if path.exists():
                raise ValueError(f"Unrecorded input exists; retain and inspect {path}")
            frame.sample(frac=1, random_state=77821).to_csv(path, index=False, float_format="%.17g")
            # Verify the actual shared CSV before either code sees it.
            recovered = pd.read_csv(path, dtype={"id": np.uint64}, float_precision="round_trip").sort_values("id").reset_index(drop=True)
            if not np.array_equal(recovered[baseline.Columns].to_numpy(), frame[baseline.Columns].to_numpy()):
                raise ValueError("Shared input failed decimal round trip")
            prior = {"path": str(path), "sha256": sha256(path), "description": description,
                     "translation_mpc_over_h": shift.tolist()}
            self.report["fixtures"][key] = prior
            self.save()
        self.fixtures[key] = (frame, prior)
        return frame, prior

    ## @brief Read the snapshot for the documented module workflow.
    # @see cosmology_tools
    #
    # @param name Stable artifact/run/check identifier as defined by the caller.
    # @param checkpoint Synchronized saved epoch index, with zero denoting imported initial state.
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
    @lru_cache(maxsize=6)
    def snapshot(self, name, checkpoint):
        return self.storage.read_snapshot(name, checkpoint)

    ## @brief Execute record.
    # @see cosmology_tools
    #
    # @param case Immutable case specification identifying the fixture, resolutions and schedule.
    # @param code Solver identifier (IPPL, native FastPM or GADGET) selected by the protocol.
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
    def run_record(self, case, code):
        rows = [run for run in self.report["runs"] if run["name"] == case.name + "_" + code]
        if len(rows) != 1:
            raise ValueError("Missing or duplicate analyzed run")
        return rows[0]

    ## @brief Evaluate the coefficients helper in the documented module workflow.
    # @see cosmology_tools
    #
    # @param case Immutable case specification identifying the fixture, resolutions and schedule.
    # @param code Solver identifier (IPPL, native FastPM or GADGET) selected by the protocol.
    # @param checkpoint Synchronized saved epoch index, with zero denoting imported initial state.
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
    def coefficients(self, case, code, checkpoint):
        return decode_modes(self.run_record(case, code)["density"][checkpoint]["direct"])

    ## @brief Evaluate the modes helper in the documented module workflow.
    # @see cosmology_tools
    #
    # @param case Immutable case specification identifying the fixture, resolutions and schedule.
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
    def modes(self, case):
        return unique_modes(4) if case.stage == "gaussian" else baseline.resolved_modes(case.fixture)

    ## @brief Check factors.
    # @see cosmology_tools
    #
    # @param output Output path; use a fresh destination where the workflow rejects existing evidence.
    # @param name Stable artifact/run/check identifier as defined by the caller.
    # @param case Immutable case specification identifying the fixture, resolutions and schedule.
    # @param ai Positive initial dimensionless scale factor.
    # @param af Positive final dimensionless scale factor, ordered after ai for forward evolution.
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def check_factors(self, output, name, case, ai, af):
        data = pd.read_csv(output / "factors.csv", float_precision="round_trip")
        baseline.validate_factor_schedule(data, case.steps, ai, af)
        nodes, weights = np.polynomial.legendre.leggauss(64)
        for field, lo, hi, power in (("drift", "a0", "a1", 2), ("canonical_kick0", "a0", "ah", 1),
                                     ("canonical_kick1", "ah", "a1", 1)):
            lower, upper = np.log(data[lo].to_numpy()), np.log(data[hi].to_numpy())
            a = np.exp((lower + upper)[:, None] / 2 + (upper - lower)[:, None] * nodes / 2)
            expected = (upper - lower) * np.sum(weights / (a**power * np.sqrt(
                Parameters["omega_m"] / a**3 + 1 - Parameters["omega_m"])), axis=1) / 2
            self.upper(name + "/factor_" + field, float(np.max(abs(data[field] / expected - 1))),
                       baseline.Limits["native_time_factor_relative"], case=case)

    ## @brief Analyze run.
    # @see cosmology_tools
    #
    # @param case Immutable case specification identifying the fixture, resolutions and schedule.
    # @param code Solver identifier (IPPL, native FastPM or GADGET) selected by the protocol.
    # @param archived Numerical particle archive record with verified file and array digests.
    # @param original Original shared input or retained reference record; it is not modified to fit the comparison.
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def analyze_run(self, case, code, archived, original):
        name = case.name + "_" + code
        output = Path(archived["descriptor"]["output"])
        metadata = dict(line.split("=", 1) for line in (output / "metadata.txt").read_text().splitlines() if "=" in line)
        execution = validate_runtime_metadata(metadata, case.ranks, code=code)
        ai, af = epochs(case, self.args.smoke)
        contract = {"n_particles_grid": case.particles, "n_grid": case.mesh, "n_steps": case.steps,
                    "n_checkpoints": Parameters["checkpoints"], "ranks": case.ranks,
                    "box_size": Parameters["box_size"], "omega_m": Parameters["omega_m"], "a_initial": ai, "a_final": af}
        if any(float(metadata[key]) != value for key, value in contract.items()):
            raise ValueError("Execution metadata violates requested contract")
        table = pd.read_csv(output / "checkpoints.csv", float_precision="round_trip")
        expected_a = ai * np.exp(np.arange(9) * np.log(af / ai) / 8)
        if (not np.array_equal(table.checkpoint, np.arange(9))
                or not np.array_equal(table.step, np.arange(9) * (case.steps // 8))
                or not np.isfinite(table.to_numpy()).all()):
            raise ValueError("Malformed checkpoint schedule")
        self.upper(name + "/schedule", float(np.max(abs(table.a / expected_a - 1))),
                   baseline.Limits["schedule_relative"], case=case)
        if code == "fastpm":
            required = {"upstream_commit": baseline.FastPMCommit, "integrator": "fastpm_solver_evolve",
                        "force_type": "FASTPM_FORCE_PM", "momentum_precision_bits": "32"}
            if any(metadata.get(key) != value for key, value in required.items()):
                raise ValueError("Reference is not pinned native plain PM")
            self.check_factors(output, name, case, ai, af)
            expected_e = np.sqrt(Parameters["omega_m"] / expected_a**3 + 1 - Parameters["omega_m"])
            self.upper(name + "/background", float(np.max(abs(table.E / expected_e - 1))), 1e-12, case=case)
        record = {"name": name, **asdict(case), "code": code, "metadata": metadata, "execution": execution,
                  "storage_output": str(output), "input_sha256": archived["descriptor"]["input_sha256"],
                  "checkpoints": table.to_dict(orient="records"), "density": [], "observations": [],
                  "mass_note": "IPPL is deposited mesh mass; native is unit-particle count, not equivalent diagnostics"}
        if code == "ippl":
            record["exceeds_historical_strict_mass_limit"] = float(metadata["maximum_mass_error"]) > 2e-12
        initial_mean = original[["px", "py", "pz"]].to_numpy().mean(axis=0)
        record["imported_mean_momentum"] = initial_mean.tolist()
        modes, box = self.modes(case), Parameters["box_size"]
        for checkpoint, a in enumerate(expected_a):
            frame = self.snapshot(name, checkpoint)
            position, momentum = frame[["x", "y", "z"]].to_numpy(), frame[["px", "py", "pz"]].to_numpy()
            self.check(f"{name}/{checkpoint}/periodic", (position >= 0).all() and (position < box).all(), case=case)
            if checkpoint == 0:
                maximum = float(np.max(abs(baseline.periodic_difference(position, original[["x", "y", "z"]].to_numpy(), box))))
                self.upper(name + "/initial_x", maximum, 128 * np.finfo(float).eps * box, case=case)
                self.check(name + "/initial_p", np.array_equal(momentum, original[["px", "py", "pz"]].to_numpy()), case=case)
            p_rms = baseline.vector_rms(momentum)
            net = float(np.linalg.norm(momentum.mean(axis=0) - initial_mean)) / p_rms if p_rms else None
            self.upper(f"{name}/{checkpoint}/mean_momentum", net,
                       baseline.Limits[f"net_momentum_{code}_relative"], case=case)
            observation = {"checkpoint": checkpoint, "a": float(a), "momentum_rms": p_rms}
            if case.stage == "gaussian":
                spectrum = analyze_spectrum(position, box, grid=Parameters["analysis_grid"],
                    max_mode=6 if self.args.smoke else 12, direct_max_mode=4)
                if not np.array_equal(spectrum["direct_low_modes"], modes):
                    raise ValueError("Fourier mode order mismatch")
                coeff = spectrum["direct_low_coefficients"]
                diagnostics = spectrum["diagnostics"]
                self.check(f"{name}/{checkpoint}/spectrum_mass", diagnostics["mass_gate_passed"], case=case)
                self.check(f"{name}/{checkpoint}/spectrum_low_extraction", diagnostics["direct_gate_passed"],
                           case=case, category="measurement", diagnostics=diagnostics)
                density = {"checkpoint": checkpoint, "a": float(a), "direct": encode_modes(coeff),
                           "fft": encode_modes(spectrum["coefficients"]), "fft_shells": spectrum["shells"],
                           "diagnostics": diagnostics, "above_mode4": "characterization only"}
            else:
                coeff = baseline.density_modes(position, box, modes)
                coeff *= np.exp(1j * (2 * np.pi / box) * (modes @ translation(case)))
                density = {"checkpoint": checkpoint, "a": float(a), "direct": encode_modes(coeff)}
                if case.fixture == "pancake":
                    corrected = frame.copy()
                    corrected[["x", "y", "z"]] = (position - translation(case)) % box
                    observation.update(baseline.planar_ordering(corrected, case.particles, box))
                    n, wave = case.particles, 2 * np.pi / box
                    q = (np.arange(n) + .5) * box / n
                    displacement = baseline.periodic_difference(corrected.x.to_numpy()[:n], q, box)
                    observation["profile"] = {"q": q.tolist(), "displacement": displacement.tolist(),
                        "px": momentum[:n, 0].tolist(), "interval_jacobian": (
                            1 + (np.roll(displacement, -1) - displacement) / (box / n)).tolist()}
                    growth, rate = baseline.growth_reference(float(a), Parameters["omega_m"])
                    final_growth = baseline.growth_reference(Parameters["spatial_af"], Parameters["omega_m"])[0]
                    amplitude = 1.5 * growth / final_growth
                    if amplitude < 1:
                        exact = -amplitude * np.sin(wave * q + .17) / wave
                        exact_p = a*a * np.sqrt(Parameters["omega_m"] / a**3 + 1 - Parameters["omega_m"]) * rate * exact
                        exact_j = 1 + (np.roll(exact, -1) - exact) / (box / n)
                        observation["analytic_precross_characterization"] = {
                            "amplitude": float(amplitude),
                            "displacement_relative_rms": float(np.linalg.norm(displacement - exact) / np.linalg.norm(exact)),
                            "momentum_relative_rms": float(np.linalg.norm(momentum[:n, 0] - exact_p) / np.linalg.norm(exact_p)),
                            "sampled_jacobian_maximum_error": float(np.max(abs(np.asarray(observation["profile"]["interval_jacobian"]) - exact_j)))}
                    if checkpoint == 8 and not self.args.smoke:
                        self.check(name + "/sheet_crossed", observation["minimum_sampled_jacobian"] < -.01,
                                   case=case, minimum=observation["minimum_sampled_jacobian"])
            record["density"].append(density)
            record["observations"].append(observation)
        self.report["runs"].append(record)
        print("ANALYZED", name, flush=True)
        self.save()

    ## @brief Compare spectra.
    # @see cosmology_tools
    #
    # @param left First ID-matched state or Fourier array; the metric specifies denominator and normalization.
    # @param lcode Solver label associated with the left comparison state.
    # @param right Second ID-matched state or Fourier array; the metric specifies denominator and normalization.
    # @param rcode Solver label associated with the right comparison state.
    # @param category Integrity, measurement, cross-code or refinement category used in the saved report.
    # @param budget Predeclared comparison tolerances; they are not fitted after observing results.
    # @param checkpoints Saved synchronized interval count; the PM step count must be divisible by it.
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def compare_spectra(self, left, lcode, right, rcode, category, budget, checkpoints=range(9)):
        if not np.array_equal(self.modes(left), self.modes(right)):
            raise ValueError("Only identical Fourier mode sets may be compared")
        for checkpoint in checkpoints:
            lr, rr = self.run_record(left, lcode), self.run_record(right, rcode)
            if abs(lr["density"][checkpoint]["a"] / rr["density"][checkpoint]["a"] - 1) > 2e-13:
                raise ValueError("Comparison epochs differ; interpolation is not permitted")
            rows = shell_comparisons(self.coefficients(left, lcode, checkpoint),
                self.coefficients(right, rcode, checkpoint), self.modes(left), budget)
            label = f"{category}/{left.name}/{lcode}_vs_{right.name}/{rcode}/{checkpoint}"
            for row in rows:
                self.check(label + f"/shell{row['shell']}", row["passed"], case=left, category=category, **{
                    key: value for key, value in row.items() if key != "passed"})
            self.report["comparisons"].append({"name": label, "category": category,
                "left": left.name + "_" + lcode, "right": right.name + "_" + rcode,
                "checkpoint": checkpoint, "a": lr["density"][checkpoint]["a"], "shells": rows})

    ## @brief Compare phase.
    # @see cosmology_tools
    #
    # @param left First ID-matched state or Fourier array; the metric specifies denominator and normalization.
    # @param lcode Solver label associated with the left comparison state.
    # @param right Second ID-matched state or Fourier array; the metric specifies denominator and normalization.
    # @param rcode Solver label associated with the right comparison state.
    # @param category Integrity, measurement, cross-code or refinement category used in the saved report.
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def compare_phase(self, left, lcode, right, rcode, category):
        for checkpoint in range(9):
            lframe = self.snapshot(left.name + "_" + lcode, checkpoint)
            rframe = self.snapshot(right.name + "_" + rcode, checkpoint)
            if left.shifted != right.shifted:
                lframe, rframe = lframe.copy(), rframe.copy()
                for case, frame in ((left, lframe), (right, rframe)):
                    frame[["x", "y", "z"]] = (frame[["x", "y", "z"]] - translation(case)) % Parameters["box_size"]
            metrics = baseline.phase_space_metrics(lframe, rframe, Parameters["box_size"], left.mesh)
            record = {"category": category, "left": left.name + "_" + lcode,
                      "right": right.name + "_" + rcode, "checkpoint": checkpoint, "phase_space": metrics}
            self.report["comparisons"].append(record)
            label = f"{category}/{left.name}/{lcode}/{checkpoint}"
            if category == "rank":
                floor = 128 * np.finfo(float).eps * left.mesh
                self.upper(label + "/x", metrics["position_cells"], baseline.Limits[f"rank_{lcode}_position_cells"] + floor,
                           case=left, category=category)
                self.upper(label + "/p", metrics["momentum_relative"], baseline.Limits[f"rank_{lcode}_momentum_relative"],
                           case=left, category=category)
            elif category == "cross_phase" and left.fixture == "pancake":
                for key in ("position_cells", "momentum_relative"):
                    self.upper(label + "/" + key, metrics[key], baseline.Limits["planar_cross_" + key],
                               case=left, category="cross_planar")

    ## @brief Evaluate the temporal helper in the documented module workflow.
    # @see cosmology_tools
    #
    # @param group Case group defining an independent refinement/control comparison.
    # @param code Solver identifier (IPPL, native FastPM or GADGET) selected by the protocol.
    # @param order Whether a successive-difference order is evaluated above the declared numerical floor.
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def temporal(self, group, code, order=True):
        fine = group[-1]
        modes = self.modes(fine)
        for checkpoint in range(1, 9):
            spectra = [self.coefficients(case, code, checkpoint) for case in group]
            finest = shell_comparisons(spectra[-2], spectra[-1], modes,
                {"power": math.inf, "complex": Budgets["time_shell_complex"], "correlation": -1.})
            for row in finest:
                # A finite complex norm is the predeclared density temporal gate;
                # no implicit extra power/correlation budget is introduced here.
                metric = row["metrics"]["complex_relative"]
                self.check(f"time/{fine.name}/{code}/{checkpoint}/shell{row['shell']}",
                    metric is not None and metric <= Budgets["time_shell_complex"], case=fine, category="time_shell",
                    shell=row["shell"], value=metric, limit=Budgets["time_shell_complex"])
            if checkpoint not in (4, 8):
                continue
            frames = [self.snapshot(case.name + "_" + code, checkpoint) for case in group]
            differences = [baseline.phase_space_metrics(a, b, Parameters["box_size"], fine.mesh)
                           for a, b in zip(frames[:-1], frames[1:])]
            density = [baseline.density_comparison(a, b) for a, b in zip(spectra[:-1], spectra[1:])]
            row = {"fixture": fine.fixture, "stage": fine.stage, "code": code, "redshift": fine.redshift,
                   "checkpoint": checkpoint, "steps": [case.steps for case in group],
                   "phase_space_differences": differences, "density_differences": density, "orders": {}}
            if order:
                lower, upper = (baseline.Limits["time_precross_ratio_range"]
                                if checkpoint == 4 and fine.stage == "spatial" else (1.5, None))
                for quantity in ("position_cells", "momentum_relative", "complex_relative"):
                    values = [entry[quantity] for entry in (density if quantity == "complex_relative" else differences)]
                    result = baseline.refinement(*values, baseline.Limits["time_precision_" + quantity], lower, upper)
                    row["orders"][quantity] = result
                    self.check(f"time/{fine.name}/{code}/{checkpoint}/{quantity}", result["passed"],
                        case=fine, category="time_global", **{key: value for key, value in result.items() if key != "passed"})
            if checkpoint == 8:
                for quantity in ("position_cells", "momentum_relative"):
                    self.upper(f"time/{fine.name}/{code}/finest_{quantity}", differences[-1][quantity],
                        baseline.Limits["time_finest_" + quantity], case=fine, category="time_global")
            self.report["time_refinement"].append(row)

    ## @brief Evaluate the comparisons helper in the documented module workflow.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def comparisons(self):
        # Derived evidence can be deterministically regenerated on resume.
        available = {run["name"] for run in self.report["runs"]}
        completed_stages = [stage for stage in ("spatial", "gaussian")
            if any(case.stage == stage for case in self.cases)
            and all(case.name + "_" + code in available for case in self.cases
                    if case.stage == stage for code in Codes)]
        self.report["completed_stages"] = completed_stages
        self.report["checks"] = [row for row in self.report["checks"] if row["category"] in ("integrity", "measurement")]
        self.report["comparisons"], self.report["time_refinement"] = [], []
        for case in self.cases:
            if case.stage not in completed_stages:
                continue
            if not self.args.smoke:
                self.compare_spectra(case, "ippl", case, "fastpm", "cross", Budgets["cross"][case.mesh])
            self.compare_phase(case, "ippl", case, "fastpm", "cross_phase")
            if case.ranks > 1:
                for code in Codes:
                    self.compare_phase(case, code, replace(case, ranks=1), code, "rank")
        if self.args.smoke:
            return
        for stage in ("spatial", "gaussian"):
            if stage not in completed_stages:
                continue
            for fixture in (("pancake", "coupled3d") if stage == "spatial" else ("gaussian",)):
                steps = 1024 if stage == "spatial" else 2048
                for code in Codes:
                    for mesh in (32, 64):
                        coarse = Case(stage, fixture, 32, mesh, steps)
                        self.compare_spectra(coarse, code, replace(coarse, particles=64), code, "particle_resolution", Budgets["spatial"])
                    for particles in (32, 64):
                        coarse = Case(stage, fixture, particles, 32, steps)
                        self.compare_spectra(coarse, code, replace(coarse, mesh=64), code, "mesh_resolution", Budgets["spatial"])
                    finest = Case(stage, fixture, 64, 64, steps)
                    group = [replace(finest, steps=nt) for nt in (steps // 2, steps, steps * 2)]
                    self.temporal(group, code)
                    if stage == "spatial":
                        for mesh in (32, 64):
                            base = replace(finest, mesh=mesh)
                            shifted = replace(base, shifted=True)
                            self.compare_spectra(shifted, code, base, code, "translation", Budgets["translation"])
                            self.compare_phase(shifted, code, base, code, "translation_phase")
                    else:
                        earlier = [replace(finest, steps=nt, redshift=99) for nt in (2048, 4096)]
                        self.temporal(earlier, code, order=False)
                        self.compare_spectra(group[-1], code, earlier[-1], code, "start_redshift", Budgets["start_redshift"], checkpoints=(8,))

    ## @brief Evaluate qualification for the documented module workflow.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def qualify(self):
        result = {}
        for stage, fixtures in (("spatial", ("pancake", "coupled3d")), ("gaussian", ("gaussian",))):
            if stage not in self.report.get("completed_stages", []):
                continue
            for fixture in fixtures:
                global_checks = [check for check in self.report["checks"] if check.get("stage") == stage
                    and check.get("fixture") == fixture and "shell" not in check]
                required_passed = bool(global_checks and all(check["passed"] for check in global_checks))
                result[fixture] = qualified_prefix(self.report["checks"], stage, fixture, required_passed)
                result[fixture]["required_global_checks_passed"] = required_passed
                result[fixture]["global_failures"] = [check["name"] for check in global_checks if not check["passed"]]
        self.report["qualification"] = result

    ## @brief Analyze completed stages.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def analyze_completed_stages(self):
        analyzed = {run["name"] for run in self.report["runs"]}
        new_stages = [stage for stage in ("spatial", "gaussian")
            if stage not in self.report.get("completed_stages", [])
            and any(case.stage == stage for case in self.cases)
            and all(case.name + "_" + code in analyzed for case in self.cases
                    if case.stage == stage for code in Codes)]
        if new_stages:
            self.comparisons()
            if not self.args.smoke:
                self.qualify()
            self.save()
            for stage in new_stages:
                print("STAGE ANALYZED", stage, flush=True)

    ## @brief Execute the documented module workflow.
    # @see cosmology_tools
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
    def run(self):
        print("Evidence:", self.root, flush=True)
        self.report["state"] = "running"
        self.save()
        completed = 0
        for case in self.cases:
            original, fixture = self.fixture(case)
            ai, af = epochs(case, self.args.smoke)
            for code in Codes:
                name = case.name + "_" + code
                command = [self.args.mpiexec, *self.args.mpi_arg, self.args.numproc_flag, str(case.ranks),
                    str(self.executables[code]), str(case.particles), str(case.mesh), str(Parameters["box_size"]),
                    str(Parameters["omega_m"]), str(ai), str(af), str(case.steps), "8", fixture["path"], str(self.storage.output_dir(name))]
                archived = self.storage.execute(name, command, particle_grid=case.particles, ranks=case.ranks,
                    checkpoints=8, timeout=self.args.timeout)
                if not any(run["name"] == name for run in self.report["runs"]):
                    # A restarted analysis never keeps partial per-run checks.
                    self.report["checks"] = [row for row in self.report["checks"] if not row["name"].startswith(name + "/")]
                    self.analyze_run(case, code, archived, original)
                    completed += 1
                # Persist completed-stage evidence even when this exact run is
                # the requested batch boundary or the next input is disk-blocked.
                self.analyze_completed_stages()
                if self.args.stop_after and completed >= self.args.stop_after:
                    self.report["state"] = "batch_complete"
                    self.save()
                    return 0
        self.comparisons()
        for path, digest in self.hashes.items():
            if sha256(path) != digest:
                raise ValueError(f"Provenance changed during study: {path}")
        for fixture in self.report["fixtures"].values():
            if sha256(fixture["path"]) != fixture["sha256"]:
                raise ValueError("Shared input changed during study")
        self.report["complete"] = len(self.report["runs"]) == self.report["expected_runs"]
        self.report["all_checks_passed"] = all(check["passed"] for check in self.report["checks"])
        self.report["integrity_passed"] = all(check["passed"] for check in self.report["checks"] if check["category"] == "integrity")
        self.report["passed"] = bool(self.report["complete"] and self.report["all_checks_passed"])
        self.report["state"] = "complete"
        self.report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        self.report["provenance_after"] = {path: sha256(path) for path in self.hashes}
        if not self.args.smoke:
            self.qualify()
        else:
            self.report["qualification"] = {"scope": "Pipeline smoke only, no resolution qualification"}
        self.save()
        print(f"{len(self.report['runs'])} runs, {len(self.report['checks'])} checks, {len(self.report['failed_checks'])} failed", flush=True)
        print(json.dumps(self.report["qualification"], indent=2), flush=True)
        return 0 if self.report["passed"] else 1


## @brief Parse the documented command-line interface and execute its selected workflow, retaining nonzero failures.
# @see cosmology_tools
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ippl-exe", type=Path)
    parser.add_argument("--fastpm-exe", type=Path)
    parser.add_argument("--fastpm-manifest", type=Path)
    parser.add_argument("--stage", choices=("spatial", "gaussian", "all"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--numproc-flag", default="-n")
    parser.add_argument("--mpi-arg", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--reserve-gib", type=float, default=1.)
    parser.add_argument("--stop-after", type=int, help="Finish this many additional runs, preserve incomplete study for resume")
    args = parser.parse_args()
    if (not math.isfinite(args.timeout) or args.timeout <= 0 or not math.isfinite(args.reserve_gib)
            or args.reserve_gib < 1 or (args.stop_after is not None and args.stop_after < 1)
            or (args.resume and args.output_dir)):
        parser.error("Require timeout>0, reserve>=1GiB, positive stop-after, and only one output/resume path")
    study = None
    try:
        study = Study(args)
        return study.run()
    except DiskSpaceBlocked as error:
        if study is not None:
            study.report.update(state="blocked_disk", complete=False, passed=False, disk_message=str(error))
            study.save()
        print("DISK BLOCK:", error, flush=True)
        return 3
    except Exception as error:
        if study is not None:
            study.report.update(state="error", complete=False, passed=False, execution_error=str(error))
            study.save()
        raise
    finally:
        if study is not None:
            study.storage.close()


## @cond CLI_DISPATCH
if __name__ == "__main__":
    raise SystemExit(main())
## @endcond
