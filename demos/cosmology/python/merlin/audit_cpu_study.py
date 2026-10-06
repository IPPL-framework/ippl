#!/usr/bin/env python3
## @file audit_cpu_study.py
# @brief Audit a completed 18-run CPU Gaussian study without rerunning simulations.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
"""Audit a completed 18-run CPU Gaussian study without rerunning simulations.

Exit zero means scope and saved integrity passed, NOT that scientific gates
passed. Original CSV bytes are not recovered from numerical NPZ archives.
The explicit source directory permits deployment outside a frozen checkout.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import importlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_paths import source_path
from types import SimpleNamespace

## @cond RUNTIME_SETTINGS
sys.dont_write_bytecode = True
## @endcond

# Independently enumerated for the frozen nine-case/two-code Gaussian matrix.
## @var ExpectedCheckCounts
# @brief Named ExpectedCheckCounts protocol/schema value; the source initializer records its exact contents.
ExpectedCheckCounts = {"integrity": 576, "measurement": 162, "cross": 243,
                       "particle_resolution": 108, "mesh_resolution": 108,
                       "rank": 36, "start_redshift": 6, "time_shell": 96, "time_global": 20}


## @brief Evaluate the expected run checks helper in the documented module workflow.
# @see cosmology_tools
#
# @param name Stable artifact/run/check identifier as defined by the caller.
# @param code Solver identifier (IPPL, native FastPM or GADGET) selected by the protocol.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def expected_run_checks(name, code):
    names = {name + "/" + suffix: "integrity" for suffix in ("schedule", "initial_x", "initial_p")}
    if code == "fastpm":
        names.update({name + "/" + suffix: "integrity" for suffix in
                      ("background", "factor_drift", "factor_canonical_kick0", "factor_canonical_kick1")})
    for checkpoint in range(9):
        names.update({f"{name}/{checkpoint}/{suffix}": "integrity"
                      for suffix in ("periodic", "mean_momentum", "spectrum_mass")})
        names[f"{name}/{checkpoint}/spectrum_low_extraction"] = "measurement"
    return names


## @brief Enumerate the frozen Gaussian names/tags without any particle analysis.
# @see cosmology_tools
#
# @param protocol Frozen case/observable/budget specification defining expected coverage and qualification.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def expected_check_contract(protocol):
    """Enumerate the frozen Gaussian names/tags without any particle analysis."""
    expected = {}

    def add(name, category, shell=None):
        row = dict(category=category, stage="gaussian", fixture="gaussian")
        if shell is not None:
            row["shell"] = shell
        if name in expected:
            raise ValueError("Duplicate expected check identity")
        expected[name] = row

    def shells(left, lcode, right, rcode, category, checkpoints=range(9)):
        for checkpoint in checkpoints:
            for shell in range(3):
                add(f"{category}/{left.name}/{lcode}_vs_{right.name}/{rcode}/{checkpoint}/shell{shell}",
                    category, shell)

    for case in protocol.study_cases("gaussian", False):
        for code in protocol.Codes:
            for name, category in expected_run_checks(case.name + "_" + code, code).items():
                add(name, category)
            if case.ranks > 1:
                for checkpoint in range(9):
                    for quantity in ("x", "p"):
                        add(f"rank/{case.name}/{code}/{checkpoint}/{quantity}", "rank")
        shells(case, "ippl", case, "fastpm", "cross")
    for code in protocol.Codes:
        for fixed in (32, 64):
            left = protocol.Case("gaussian", "gaussian", 32, fixed, 2048)
            shells(left, code, replace(left, particles=64), code, "particle_resolution")
            left = protocol.Case("gaussian", "gaussian", fixed, 32, 2048)
            shells(left, code, replace(left, mesh=64), code, "mesh_resolution")
        fine = protocol.Case("gaussian", "gaussian", 64, 64, 4096)
        for redshift in (49, 99):
            case = replace(fine, redshift=redshift)
            for checkpoint in range(1, 9):
                for shell in range(3):
                    add(f"time/{case.name}/{code}/{checkpoint}/shell{shell}", "time_shell", shell)
            if redshift == 49:
                for checkpoint in (4, 8):
                    for quantity in ("position_cells", "momentum_relative", "complex_relative"):
                        add(f"time/{case.name}/{code}/{checkpoint}/{quantity}", "time_global")
            for quantity in ("position_cells", "momentum_relative"):
                add(f"time/{case.name}/{code}/finest_{quantity}", "time_global")
        shells(fine, code, replace(fine, redshift=99), code, "start_redshift", checkpoints=(8,))
    return expected


## @brief Verify check links and qualification.
# @see cosmology_tools
#
# @param report Structured campaign/audit report; recorded failures are not retroactively changed.
# @param protocol Frozen case/observable/budget specification defining expected coverage and qualification.
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def verify_check_links_and_qualification(report, protocol):
    expected = expected_check_contract(protocol)
    checks = {row["name"]: row for row in report["checks"]}
    if set(checks) != set(expected):
        raise ValueError("Scientific check identities differ from the frozen protocol")
    for name, contract in expected.items():
        row = checks[name]
        if (any(row.get(key) != value for key, value in contract.items())
                or ("shell" in row) != ("shell" in contract)):
            raise ValueError("Scientific check category/stage/fixture/shell tags differ")
    for comparison in report["comparisons"]:
        if "shells" not in comparison:
            continue
        left, lcode = comparison["left"].rsplit("_", 1)
        right, rcode = comparison["right"].rsplit("_", 1)
        label = f"{comparison['category']}/{left}/{lcode}_vs_{right}/{rcode}/{comparison['checkpoint']}"
        for saved in comparison["shells"]:
            row = checks[label + f"/shell{saved['shell']}"]
            if any(row.get(key) != value for key, value in saved.items()):
                raise ValueError("Scientific shell check differs from its saved comparison row")
    temporary = SimpleNamespace(report={"completed_stages": ["gaussian"], "checks": report["checks"]})
    protocol.Study.qualify(temporary)
    if report["qualification"] != temporary.report["qualification"]:
        raise ValueError("Qualification/global prefix differs from the saved checks")


## @brief Load the requested frozen analysis helpers and verify that their module paths match the requested source.
# @see cosmology_tools
#
# @param source_dir Frozen cosmology source root or its Python directory; helper provenance must match.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def load_helpers(source_dir):
    source_dir = Path(source_dir).resolve(strict=True)
    if source_dir.name != "python":
        source_dir = source_dir / "python"
    sys.path.insert(0, str(source_dir))
    modules = [importlib.import_module(name) for name in
               ("validate_resolution_study", "study_storage", "plot_resolution_study")]
    if any(Path(module.__file__).resolve() != source_dir / (module.__name__ + ".py")
           for module in modules):
        raise ValueError("Imported audit dependency is outside the requested source directory")
    return modules


## @brief Validate scope.
# @see cosmology_tools
#
# @param report Structured campaign/audit report; recorded failures are not retroactively changed.
# @param protocol Frozen case/observable/budget specification defining expected coverage and qualification.
# @param plot Requested frozen plotting helper module used to check consistency, not rerun scientific evolution.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def validate_scope(report, protocol, plot):
    cases = protocol.study_cases("gaussian", False)
    if (report.get("schema") != "ippl-resolution-study-v1" or report.get("state") != "complete"
            or report.get("complete") is not True or report.get("integrity_passed") is not True):
        raise ValueError("Require a completed study with passing integrity gates")
    if (report["configuration"]["stage"] != "gaussian"
            or report["configuration"]["smoke"] is not False
            or report["completed_stages"] != ["gaussian"]
            or report["planned_cases"] != [asdict(case) for case in cases]
            or report["parameters"] != protocol.canonical(protocol.Parameters)
            or report["budgets"] != protocol.canonical(protocol.Budgets)):
        raise ValueError("Study differs from the canonical full Gaussian protocol")
    expected = {case.name + "_" + code: (case, code) for case in cases for code in protocol.Codes}
    if (report["expected_runs"] != 18 or len(report["runs"]) != 18
            or {run["name"] for run in report["runs"]} != set(expected)):
        raise ValueError("Require exactly the canonical 18 executed runs")
    for run in report["runs"]:
        case, code = expected[run["name"]]
        if run["code"] != code or any(run[key] != value for key, value in asdict(case).items()):
            raise ValueError("Run descriptor differs from the planned case")
        ai, af = protocol.epochs(case, False)
        metadata = run["metadata"]
        contract = dict(n_particles_grid=case.particles, n_grid=case.mesh, n_steps=case.steps,
                        n_checkpoints=8, box_size=protocol.Parameters["box_size"],
                        omega_m=protocol.Parameters["omega_m"], a_initial=ai, a_final=af)
        if (any(float(metadata[key]) != value for key, value in contract.items())
                or metadata["ranks"] != str(case.ranks) or metadata["threads"] != "1"):
            raise ValueError("Mesh/time/cosmology/rank/thread metadata differs from protocol")
        if code == "ippl":
            if metadata.get("execution_space") != "OpenMP" or metadata.get("memory_space") != "Host":
                raise ValueError("CPU IPPL must report actual OpenMP execution and Host memory")
        elif any(metadata.get(key) != value for key, value in {
                "upstream_commit": protocol.baseline.FastPMCommit,
                "integrator": "fastpm_solver_evolve", "force_type": "FASTPM_FORCE_PM",
                "momentum_precision_bits": "32"}.items()):
            raise ValueError("Reference is not the pinned native plain-PM integrator")
        rows = run["checkpoints"]
        if ([row["checkpoint"] for row in rows] != list(range(9))
                or [row["step"] for row in rows] != [i * (case.steps // 8) for i in range(9)]):
            raise ValueError("Missing, duplicate or wrong checkpoint schedule")
        for i, row in enumerate(rows):
            expected_a = ai * math.exp(i * math.log(af / ai) / 8)
            if not math.isclose(row["a"], expected_a,
                                rel_tol=protocol.baseline.Limits["schedule_relative"], abs_tol=0):
                raise ValueError("Wrong checkpoint scale factor")
    checks = report["checks"]
    if (not checks or len({row["name"] for row in checks}) != len(checks)
            or any(type(row.get("passed")) is not bool for row in checks)):
        raise ValueError("Checks must be nonempty, unique and explicitly boolean")
    if Counter(row.get("category") for row in checks) != ExpectedCheckCounts:
        raise ValueError("Check category coverage differs from the frozen 1355-check protocol")
    integrity = [row for row in checks if row.get("category") == "integrity"]
    if not integrity or not all(row["passed"] for row in integrity):
        raise ValueError("Recorded integrity checks are missing or failed")
    for name, (_, code) in expected.items():
        actual = {row["name"]: row["category"] for row in checks if row["name"].startswith(name + "/")}
        if actual != expected_run_checks(name, code):
            raise ValueError("A run lacks its exact integrity/measurement check coverage")
    failures = [row for row in checks if not row["passed"]]
    if (report["failed_checks"] != failures
            or report["all_checks_passed"] is not (not failures)
            or report["passed"] is not (not failures)):
        raise ValueError("Scientific pass/fail summaries are inconsistent")
    verify_check_links_and_qualification(report, protocol)
    # This verifies the existing exact comparison/time matrix and checks saved
    # shell metrics against saved Fourier vectors; it does not remeasure particles.
    extracted = plot.extract_data(report)
    if extracted["completed_stages"] != ["gaussian"] or extracted["omitted_stages"]:
        raise ValueError("Gaussian comparison/time evidence is incomplete")
    return dict(executed_runs=18, ippl_backend="OpenMP", checkpoints_per_run=9,
                checks=len(checks), integrity_checks=len(integrity), failed_checks=len(failures),
                scientific_acceptance=report["passed"], qualification=report["qualification"],
                comparison_records=len(report["comparisons"]),
                time_refinement_records=len(report["time_refinement"]))


## @brief Verify recorded archives, retained artifacts and source snapshots without rerunning the simulation.
# @see cosmology_tools
#
# @param report_path Retained report path, including the provenance needed by the audit.
# @param report Structured campaign/audit report; recorded failures are not retroactively changed.
# @param protocol Frozen case/observable/budget specification defining expected coverage and qualification.
# @param storage Requested numerical-archive storage helper module; completion is separate from acceptance.
# @param source_dir Frozen cosmology source root or its Python directory; helper provenance must match.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def verify_saved_files(report_path, report, protocol, storage, source_dir):
    report_path, source_dir = Path(report_path), Path(source_dir).resolve()
    hashes = {}

    def include(path, digest):
        path = str(Path(path).resolve(strict=True))
        if path in hashes and hashes[path] != digest:
            raise ValueError("Conflicting artifact digests")
        hashes[path] = digest

    if not report["provenance"] or report["provenance_after"] != report["provenance"]:
        raise ValueError("Missing or changed before/after provenance")
    for path, digest in report["provenance"].items():
        include(path, digest)
    snapshots = report["source_snapshot"]
    expected_copies = {str(report_path.parent / "source-snapshot" / name):
                      report["provenance"][str(source_path(name, source_dir))] for name in protocol.SourceNames}
    if snapshots != expected_copies:
        raise ValueError("Source snapshot coverage differs from frozen protocol sources")
    for path, digest in snapshots.items():
        include(path, digest)
    for fixture in report["fixtures"].values():
        include(fixture["path"], fixture["sha256"])
    config = report["configuration"]
    native = protocol.verify_native_manifest(Path(config["fastpm_manifest"]), Path(config["fastpm_exe"]))
    if any(report["provenance"].get(path) != digest for path, digest in native.items()):
        raise ValueError("Native reference manifest is not covered by provenance")
    journal_path = report_path.parent / "storage/storage.json"
    journal_hash = storage.sha256(journal_path)
    journal = json.loads(journal_path.read_text())
    runs = {run["name"]: run for run in report["runs"]}
    if journal["provenance"] != report["provenance"] or set(journal["runs"]) != set(runs):
        raise ValueError("Storage provenance or executed-run coverage differs")
    archives = 0
    for name, archived in journal["runs"].items():
        run, descriptor = runs[name], archived["descriptor"]
        if (archived["state"] != "complete" or archived.get("storage_complete") is not True
                or archived["return_code"] != 0):
            raise ValueError("Execution/archive is not complete")
        output = Path(descriptor["output"])
        fixture_key = f"gaussian_p{run['particles']}_z{run['redshift']}_base"
        fixture = report["fixtures"][fixture_key]
        expected_args = [config[run["code"] + "_exe"], str(run["particles"]), str(run["mesh"]),
                         str(protocol.Parameters["box_size"]), str(protocol.Parameters["omega_m"]),
                         str(1 / (1 + run["redshift"])), str(protocol.Parameters["gaussian_af"]),
                         str(run["steps"]), "8", fixture["path"], str(output)]
        expected_command = [config["mpiexec"], *config["mpi_arg"], config["numproc_flag"],
                            str(run["ranks"]), *expected_args]
        if (descriptor["command"] != expected_command or str(output) != run["storage_output"]
                or descriptor["particle_grid"] != run["particles"] or descriptor["ranks"] != run["ranks"]
                or descriptor["checkpoints"] != 8 or descriptor["environment"] != storage.FixedEnvironment
                or descriptor["input_sha256"] != fixture["sha256"]
                or run["input_sha256"] != fixture["sha256"]):
            raise ValueError("Stored command/input/environment differs from analyzed run")
        include(expected_args[0], descriptor["executable_sha256"])
        required = {str(output / "metadata.txt"), str(output / "checkpoints.csv"), archived["log_path"]}
        if run["code"] == "fastpm":
            required.add(str(output / "factors.csv"))
        if not required.issubset(archived["retained_artifacts"]):
            raise ValueError("Missing retained metadata/checkpoint/log/factor digest")
        for path, digest in archived["retained_artifacts"].items():
            include(path, digest)
        pairs = [line.split("=", 1) for line in (output / "metadata.txt").read_text().splitlines() if "=" in line]
        if len(dict(pairs)) != len(pairs) or dict(pairs) != run["metadata"]:
            raise ValueError("Saved metadata differs from analyzed metadata")
        table = protocol.pd.read_csv(output / "checkpoints.csv", float_precision="round_trip")
        if table.to_dict(orient="records") != run["checkpoints"]:
            raise ValueError("Saved checkpoint table differs from analyzed checkpoints")
        if sorted(row["checkpoint"] for row in archived["snapshots"]) != list(range(9)):
            raise ValueError("Missing or duplicate numerical checkpoint archive")
        for row in archived["snapshots"]:
            if row["particles"] != run["particles"]**3:
                raise ValueError("Archive particle count differs from protocol")
            include(row["path"], row["sha256"])
            storage.read_archive(row["path"], row["particles"], expected_sha=row["sha256"],
                                 expected_digest=row["array_sha256"])
            archives += 1
    for path, digest in hashes.items():
        if storage.sha256(path) != digest:
            raise ValueError(f"Artifact changed: {path}")
    if storage.sha256(journal_path) != journal_hash:
        raise ValueError("Storage journal changed during audit")
    return dict(verified_file_hashes=len(hashes), verified_numerical_archives=archives,
                storage_journal_sha256=journal_hash)


## @brief Parse the documented command-line interface and execute its selected workflow, retaining nonzero failures.
# @see cosmology_tools
#
# @param argv Command-line argument vector; program-specific parsing is documented by main/usage.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--source-dir", type=Path, required=True, help="Frozen checkout demos/cosmology directory")
    parser.add_argument("--output", type=Path, required=True, help="New audit JSON; never overwritten")
    args = parser.parse_args(argv)
    result = dict(schema="ippl-cpu-completion-audit-v1", audit_passed=False,
                  scope="Integrity/scope audit, not independent physics recomputation or physical acceptance")
    # Claim only a new output. A rejected audit still leaves an explicit result.
    with args.output.open("x") as output:
        try:
            protocol, storage, plot = load_helpers(args.source_dir)
            report_path = args.report.resolve(strict=True)
            before = storage.sha256(report_path)
            report = json.loads(report_path.read_text())
            result.update(validate_scope(report, protocol, plot))
            result.update(verify_saved_files(report_path, report, protocol, storage, args.source_dir))
            if storage.sha256(report_path) != before:
                raise ValueError("Report changed during audit")
            result.update(audit_passed=True, report_path=str(report_path), report_sha256=before,
                          audit_helper_sha256=storage.sha256(__file__),
                          comparison_helper_sha256=storage.sha256(plot.__file__))
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["audit_passed"] else 1


## @cond CLI_DISPATCH
if __name__ == "__main__":
    raise SystemExit(main())
## @endcond
