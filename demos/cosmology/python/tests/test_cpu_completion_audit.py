## @file test_cpu_completion_audit.py
# @brief CPU audit contract tests with synthetic reports and tiny numerical archives.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
# Tests assert declared invariants using isolated fixtures; no scientific tolerances are relaxed by documentation.
"""CPU audit contract tests with synthetic reports and tiny numerical archives."""
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

## @var Source
# @brief Named Source protocol/schema value; the source initializer records its exact contents.
Source = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Source / "merlin"))
sys.path.insert(0, str(Source))
import audit_cpu_study as audit
import plot_resolution_study as plot
import study_storage as storage
import validate_resolution_study as protocol
from test_plot_resolution_study import synthetic_report


## @brief Evaluate the complete report helper in the documented module workflow.
# @see cosmology_tools
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def complete_report():
    report = synthetic_report()
    report.update(state="complete", integrity_passed=True, all_checks_passed=False, passed=False,
                  expected_runs=18, completed_stages=["gaussian"])
    report["configuration"] = dict(stage="gaussian", smoke=False)
    report["planned_cases"] = [asdict(case) for case in protocol.study_cases("gaussian", False)]
    report["runs"] = [row for row in report["runs"] if row["stage"] == "gaussian"]
    report["comparisons"] = [row for row in report["comparisons"] if row["left"].startswith("gaussian")]
    report["time_refinement"] = [row for row in report["time_refinement"] if row["stage"] == "gaussian"]
    report["qualification"] = {"gaussian": report["qualification"]["gaussian"]}
    for run in report["runs"]:
        ai = 1 / (1 + run["redshift"])
        metadata = dict(n_particles_grid=str(run["particles"]), n_grid=str(run["mesh"]),
                        n_steps=str(run["steps"]), n_checkpoints="8", box_size="168.75", omega_m="0.31",
                        a_initial=str(ai), a_final="1.0", ranks=str(run["ranks"]), threads="1")
        if run["code"] == "ippl":
            metadata.update(execution_space="OpenMP", memory_space="Host")
        else:
            metadata.update(upstream_commit=protocol.baseline.FastPMCommit, integrator="fastpm_solver_evolve",
                            force_type="FASTPM_FORCE_PM", momentum_precision_bits="32")
        run["metadata"] = metadata
        run["checkpoints"] = [dict(checkpoint=i, step=i*(run["steps"]//8),
                                   a=ai*math.exp(i*math.log(1/ai)/8)) for i in range(9)]
    report["checks"] = [dict(name=name, passed=True, **contract)
                        for name, contract in audit.expected_check_contract(protocol).items()]
    checks = {row["name"]: row for row in report["checks"]}
    for comparison in report["comparisons"]:
        if "shells" not in comparison:
            continue
        left, lcode = comparison["left"].rsplit("_", 1)
        right, rcode = comparison["right"].rsplit("_", 1)
        label = f"{comparison['category']}/{left}/{lcode}_vs_{right}/{rcode}/{comparison['checkpoint']}"
        for saved in comparison["shells"]:
            checks[label + f"/shell{saved['shell']}"].update(saved)
    failure = next(row for row in report["checks"] if row["category"] == "time_global")
    failure.update(passed=False, stage="gaussian")
    report["failed_checks"] = [failure]
    protocol.Study.qualify(SimpleNamespace(report=report))
    return report


## @brief One-particle I/O unit fixture; deliberately NOT a canonical physics study.
# @see cosmology_tools
#
# @param root Campaign or artifact root following this module's ownership contract.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def saved_fixture(root):
    """One-particle I/O unit fixture; deliberately NOT a canonical physics study."""
    source = root / "source"
    source.mkdir()
    original = source / "source.py"
    original.write_text("# synthetic frozen source\n")
    copy = root / "source-snapshot/source.py"
    copy.parent.mkdir()
    copy.write_bytes(original.read_bytes())
    executable = root / "executable"
    executable.write_bytes(b"synthetic executable identity")
    manifest = root / "native-manifest"
    manifest.write_text("synthetic unit-test manifest")
    fixture = root / "gaussian_p1_z49_base.csv"
    fixture.write_text("id,x,y,z,px,py,pz,mass\n0,0,0,0,0,0,0,1\n")
    output = root / "storage/run"
    output.mkdir(parents=True)
    log = root / "storage/run.log"
    log.write_text("synthetic completed execution")
    metadata = {"ranks": "1", "threads": "1", "execution_space": "OpenMP", "memory_space": "Host"}
    (output / "metadata.txt").write_text("".join(f"{key}={value}\n" for key, value in metadata.items()))
    points = [dict(checkpoint=i, step=i, a=.02*math.exp(i*math.log(50)/8)) for i in range(9)]
    pd.DataFrame(points).to_csv(output / "checkpoints.csv", index=False)
    points = pd.read_csv(output / "checkpoints.csv", float_precision="round_trip").to_dict(orient="records")
    config = dict(ippl_exe=str(executable), fastpm_exe=str(executable), fastpm_manifest=str(manifest),
                  mpiexec="mpiexec", mpi_arg=["--bind-to", "none"], numproc_flag="-n")
    command = ["mpiexec", "--bind-to", "none", "-n", "1", str(executable), "1", "1", "1.0", "0.31",
               "0.02", "1.0", "8", "8", str(fixture), str(output)]
    descriptor = dict(command=command, output=str(output), particle_grid=1, ranks=1, checkpoints=8,
                      environment=storage.FixedEnvironment.copy(), input_sha256=storage.sha256(fixture),
                      executable_sha256=storage.sha256(executable))
    ids, phase = np.arange(1, dtype=np.uint64), np.zeros((1, 6), dtype=np.float64)
    snapshots = []
    for checkpoint in range(9):
        path = output / f"particles_checkpoint{checkpoint:04d}.npz"
        np.savez_compressed(path, ids=ids, phase_space=phase)
        snapshots.append(dict(path=str(path), sha256=storage.sha256(path), particles=1, checkpoint=checkpoint,
                              array_sha256=storage.array_digest(ids, phase)))
    provenance = {str(path): storage.sha256(path) for path in (original, executable, manifest)}
    run = dict(name="run", code="ippl", particles=1, mesh=1, steps=8, ranks=1, redshift=49,
               storage_output=str(output), metadata=metadata, checkpoints=points,
               input_sha256=descriptor["input_sha256"])
    report = dict(provenance=provenance, provenance_after=provenance.copy(),
                  source_snapshot={str(copy): storage.sha256(copy)}, configuration=config,
                  fixtures={"gaussian_p1_z49_base": dict(path=str(fixture), sha256=storage.sha256(fixture))},
                  runs=[run])
    retained = {str(path): storage.sha256(path) for path in
                (output / "metadata.txt", output / "checkpoints.csv", log)}
    journal = dict(provenance=provenance, runs={"run": dict(state="complete", return_code=0,
        storage_complete=True, log_path=str(log), descriptor=descriptor, snapshots=snapshots,
        retained_artifacts=retained)})
    journal_path = root / "storage/storage.json"
    journal_path.write_text(json.dumps(journal))
    small_protocol = SimpleNamespace(SourceNames=("source.py",), pd=pd,
        Parameters=dict(box_size=1., omega_m=.31, gaussian_af=1.), verify_native_manifest=lambda *args: {})
    return report, journal, journal_path, source, small_protocol


## @brief Regression suite for CPUCompletionAudit.
# @see cosmology_tools
class CPUCompletionAuditTests(unittest.TestCase):
    ## @brief Evaluate the setUpClass helper in the documented module workflow.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    @classmethod
    def setUpClass(cls):
        ## @var report
        # @brief Structured campaign/audit report; recorded failures are not retroactively changed.
        cls.report = complete_report()

    ## @brief Verify complete scope preserves scientific failure.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_complete_scope_preserves_scientific_failure(self):
        result = audit.validate_scope(self.report, protocol, plot)
        self.assertEqual(result["executed_runs"], 18)
        self.assertEqual(result["failed_checks"], 1)
        self.assertFalse(result["scientific_acceptance"])

    ## @brief Verify matrix backend and budget mismatches rejected.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_matrix_backend_and_budget_mismatches_rejected(self):
        for mutation in ("missing", "duplicate", "smoke", "budget", "backend", "threads", "reference", "checkpoint"):
            report = deepcopy(self.report)
            if mutation == "missing": report["runs"].pop()
            if mutation == "duplicate": report["runs"][-1] = deepcopy(report["runs"][0])
            if mutation == "smoke": report["configuration"]["smoke"] = True
            if mutation == "budget": report["budgets"]["time_shell_complex"] *= 2
            if mutation == "backend": report["runs"][0]["metadata"]["execution_space"] = "Cuda"
            if mutation == "threads": report["runs"][0]["metadata"]["threads"] = "2"
            if mutation == "reference": report["runs"][1]["metadata"]["force_type"] = "COLA"
            if mutation == "checkpoint": report["runs"][0]["checkpoints"].pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit.validate_scope(report, protocol, plot)

    ## @brief Verify integrity and success flags not trusted.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_integrity_and_success_flags_not_trusted(self):
        for mutation in ("missing_run_check", "wrong_run_check", "missing_science_check", "failed_integrity",
                         "nonboolean", "duplicate", "summary"):
            report = deepcopy(self.report)
            if mutation == "missing_run_check": report["checks"].pop(0)
            if mutation == "wrong_run_check": report["checks"][0]["name"] += "_unexpected"
            if mutation == "missing_science_check": report["checks"].pop(-1)
            if mutation == "failed_integrity": report["checks"][0]["passed"] = False
            if mutation == "nonboolean": report["checks"][0]["passed"] = 1
            if mutation == "duplicate": report["checks"].append(deepcopy(report["checks"][0]))
            if mutation == "summary": report["passed"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit.validate_scope(report, protocol, plot)

    ## @brief Verify comparison and time coverage required.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_comparison_and_time_coverage_required(self):
        for field in ("comparisons", "time_refinement"):
            report = deepcopy(self.report)
            report[field].pop()
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.validate_scope(report, protocol, plot)

    ## @brief Verify exact scientific identities tags and comparison links.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_exact_scientific_identities_tags_and_comparison_links(self):
        for mutation in ("name", "stage", "fixture", "shell", "comparison_pass"):
            report = deepcopy(self.report)
            row = next(row for row in report["checks"] if row["category"] == "cross")
            if mutation == "name": row["name"] = "unrelated_cross_check"
            if mutation == "stage": row["stage"] = "spatial"
            if mutation == "fixture": row["fixture"] = "pancake"
            if mutation == "shell": row["shell"] = 2
            if mutation == "comparison_pass": report["comparisons"][0]["shells"][0]["passed"] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit.validate_scope(report, protocol, plot)

    ## @brief Verify qualification cannot overstate prefix or remove global failure.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_qualification_cannot_overstate_prefix_or_remove_global_failure(self):
        for mutation in ("prefix", "global_pass", "global_failures", "shell_count"):
            report = deepcopy(self.report)
            qualified = report["qualification"]["gaussian"]
            if mutation == "prefix": qualified["contiguous_shell_count"] = 99
            if mutation == "global_pass": qualified["required_global_checks_passed"] = True
            if mutation == "global_failures": qualified["global_failures"] = []
            if mutation == "shell_count": qualified["shells"][0]["checks"] -= 1
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "Qualification"):
                audit.validate_scope(report, protocol, plot)

    ## @brief Verify nine numerical archives and retained files.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_nine_numerical_archives_and_retained_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            report, journal, journal_path, source, small = saved_fixture(root)
            result = audit.verify_saved_files(root / "results.json", report, small, storage, source)
            self.assertEqual(result["verified_numerical_archives"], 9)
            self.assertEqual(result["storage_journal_sha256"], storage.sha256(journal_path))

    ## @brief Verify archive numerical digest detects rehashed value change.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_archive_numerical_digest_detects_rehashed_value_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            report, journal, journal_path, source, small = saved_fixture(root)
            row = journal["runs"]["run"]["snapshots"][0]
            np.savez_compressed(row["path"], ids=np.arange(1, dtype=np.uint64),
                                phase_space=np.ones((1, 6), dtype=np.float64))
            row["sha256"] = storage.sha256(row["path"])
            journal_path.write_text(json.dumps(journal))
            with self.assertRaisesRegex(ValueError, "numerical digest"):
                audit.verify_saved_files(root / "results.json", report, small, storage, source)

    ## @brief Verify missing artifacts schedule or provenance rejected.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_missing_artifacts_schedule_or_provenance_rejected(self):
        for mutation in ("checkpoint", "retained", "command", "before_after", "source_copy", "input", "table"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                report, journal, journal_path, source, small = saved_fixture(root)
                run = journal["runs"]["run"]
                if mutation == "checkpoint": run["snapshots"].pop()
                if mutation == "retained": run["retained_artifacts"].pop(run["log_path"])
                if mutation == "command": run["descriptor"]["command"][-4] = "16"
                if mutation == "before_after": report["provenance_after"] = {}
                if mutation == "source_copy": next(iter(Path(root / "source-snapshot").iterdir())).write_text("changed")
                if mutation == "input": Path(report["fixtures"]["gaussian_p1_z49_base"]["path"]).write_text("changed")
                if mutation == "table": report["runs"][0]["checkpoints"][0]["a"] = .5
                journal_path.write_text(json.dumps(journal))
                with self.assertRaises(ValueError):
                    audit.verify_saved_files(root / "results.json", report, small, storage, source)

    ## @brief Verify exclusive output and explicit failed audit.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_exclusive_output_and_explicit_failed_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "audit.json"
            arguments = [str(root / "missing-report.json"), "--source-dir", str(Source), "--output", str(output)]
            with patch("builtins.print"):
                self.assertEqual(audit.main(arguments), 1)
            result = json.loads(output.read_text())
            self.assertFalse(result["audit_passed"])
            self.assertIn("FileNotFoundError", result["error"])
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                audit.main(arguments)
            self.assertEqual(output.read_bytes(), before)


## @cond CLI_DISPATCH
if __name__ == "__main__":
    unittest.main()
## @endcond
