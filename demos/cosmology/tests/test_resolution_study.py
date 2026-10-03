"""Independent matrix, comparison, qualification and control-flow checks.

No MPI launch or simulation executable is used. Small synthetic particle
samples test translation and complex Fourier norms, not physical evolution.
"""
from argparse import Namespace
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_resolution_study as study


def bare_study(stage="all", smoke=False):
    result = study.Study.__new__(study.Study)
    result.args = Namespace(stage=stage, smoke=smoke)
    result.report = {"checks": [], "comparisons": [], "time_refinement": [], "qualification": {}, "runs": []}
    return result


def shell_check(shell, passed=True, *, stage="spatial", fixture="coupled3d", category="cross"):
    return {"name": f"{stage}/{fixture}/{category}/{shell}", "stage": stage, "fixture": fixture,
            "category": category, "shell": shell, "passed": passed}


def particle_frame():
    position = np.array([[.2, .4, .6], [1.3, 2.5, 3.7], [4.2, 5.4, 6.6], [7.1, 8.3, 9.5]])
    result = pd.DataFrame(position, columns=["x", "y", "z"])
    result.insert(0, "id", np.arange(len(position), dtype=np.uint64))
    result[["px", "py", "pz"]] = np.array([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9], [1., 1.1, 1.2]])
    return result


class ResolutionMatrixTests(unittest.TestCase):
    def test_full_matrix_is_exactly_52_runs_with_no_duplicates(self):
        cases = study.study_cases()
        self.assertEqual(len(cases), 26)
        self.assertEqual(len({case.name for case in cases}), 26)
        self.assertEqual(len(study.Codes) * len(cases), 52)
        self.assertEqual(sum(case.stage == "spatial" for case in cases), 17)
        self.assertEqual(sum(case.stage == "gaussian" for case in cases), 9)
        self.assertEqual(set(study.study_cases("spatial")) | set(study.study_cases("gaussian")), set(cases))
        self.assertFalse(set(study.study_cases("spatial")) & set(study.study_cases("gaussian")))
        for fixture, stage, steps in (("pancake", "spatial", 1024), ("coupled3d", "spatial", 1024),
                                      ("gaussian", "gaussian", 2048)):
            for particles in (32, 64):
                for mesh in (32, 64):
                    self.assertIn(study.Case(stage, fixture, particles, mesh, steps), cases)
            finest = study.Case(stage, fixture, 64, 64, steps)
            for count in (steps // 2, steps, steps * 2):
                self.assertIn(replace(finest, steps=count), cases)
        self.assertEqual({case for case in cases if case.ranks > 1}, {
            study.Case("spatial", "coupled3d", 64, 64, 1024, 3),
            study.Case("gaussian", "gaussian", 64, 64, 2048, 4)})
        shifted = {case for case in cases if case.shifted}
        self.assertEqual(shifted, {study.Case("spatial", fixture, 64, mesh, 1024, shifted=True)
                                  for fixture in ("pancake", "coupled3d") for mesh in (32, 64)})
        self.assertEqual({case.steps for case in cases if case.redshift == 99}, {2048, 4096})
        self.assertTrue(all(case.steps % study.Parameters["checkpoints"] == 0 for case in cases))

    def test_smoke_is_eight_runs_and_not_scientific_matrix(self):
        cases = study.study_cases(smoke=True)
        expected = {study.Case("spatial", "pancake", 8, 8, 16, ranks) for ranks in (1, 2)}
        expected |= {study.Case("gaussian", "gaussian", 16, 16, 16, ranks, redshift=99) for ranks in (1, 2)}
        self.assertEqual(set(cases), expected)
        self.assertEqual(len(cases) * len(study.Codes), 8)
        self.assertEqual(len(study.study_cases("spatial", smoke=True)), 2)
        self.assertEqual(len(study.study_cases("gaussian", smoke=True)), 2)
        self.assertFalse(any(case.shifted for case in cases))
        with self.assertRaises(ValueError):
            study.study_cases("unknown")

    def test_input_identity_excludes_execution_mesh_timestep_and_rank(self):
        case = study.Case("gaussian", "gaussian", 32, 32, 2048)
        for other in (replace(case, mesh=64), replace(case, steps=4096), replace(case, ranks=4)):
            self.assertNotEqual(case.name, other.name)
            self.assertEqual(case.input_key, other.input_key)
        for other in (replace(case, particles=64), replace(case, redshift=99), replace(case, shifted=True)):
            self.assertNotEqual(case.input_key, other.input_key)

    def test_epochs_and_fixed_physical_translation(self):
        case = study.Case("spatial", "coupled3d", 64, 64, 1024, shifted=True)
        self.assertEqual(study.epochs(case), (.02, .2))
        self.assertEqual(study.epochs(case, smoke=True), (.02, .04))
        self.assertEqual(study.epochs(study.Case("gaussian", "gaussian", 64, 64, 2048)), (.02, 1.))
        earlier = study.Case("gaussian", "gaussian", 64, 64, 2048, redshift=99)
        self.assertEqual(study.epochs(earlier), (.01, 1.))
        self.assertEqual(study.epochs(earlier, smoke=True), (.01, .02))
        expected = np.asarray([.37, .23, .41]) * 168.75 / 64
        np.testing.assert_array_equal(study.translation(case), expected)
        np.testing.assert_array_equal(study.translation(replace(case, mesh=32, particles=32)), expected)
        np.testing.assert_array_equal(study.translation(replace(case, shifted=False)), np.zeros(3))


class ShellComparisonTests(unittest.TestCase):
    def setUp(self):
        self.modes = np.array([[1, 0, 0], [1, 1, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])
        self.budget = {"power": .1, "complex": .1, "correlation": .99}

    def test_exact_complex_norm_not_difference_of_rms_or_absolute_correlation(self):
        left = np.array([1., 0., 1., 1., 1.], dtype=complex)
        right = np.array([0., 1., 1j, 1., 1.], dtype=complex)
        rows = study.shell_comparisons(left, right, self.modes, self.budget)
        self.assertEqual([row["pairs"] for row in rows], [2, 1, 2])
        self.assertEqual([row["shell"] for row in rows], [0, 1, 2])
        for row in rows[:2]:
            self.assertEqual(row["metrics"]["power_ratio"], 1.)
            self.assertAlmostEqual(row["metrics"]["complex_relative"], np.sqrt(2.))
            self.assertEqual(row["metrics"]["correlation"], 0.)
            self.assertFalse(row["passed"])
        self.assertTrue(rows[2]["passed"])

    def test_independent_normalization_and_real_cross_power(self):
        left = np.array([1 + .5j, .4 - .7j, .3 + .8j, 1 - .3j, -.1 + .9j])
        right = np.array([.9 + .6j, .3 - .6j, .4 + .7j, .9 - .4j, -.2 + .8j])
        rows = study.shell_comparisons(left, right, self.modes, self.budget)
        radius = np.sqrt(np.sum(self.modes**2, axis=1))
        for row in rows:
            mask = (radius >= row["lower"]) & (radius < row["upper"])
            a, b = left[mask], right[mask]
            pa, pb = np.sum(abs(a)**2), np.sum(abs(b)**2)
            denominator = np.sqrt(pa * pb)
            self.assertAlmostEqual(row["metrics"]["power_ratio"], pa / pb)
            self.assertAlmostEqual(row["metrics"]["complex_relative"],
                                   np.sqrt(np.sum(abs(a-b)**2)) / np.sqrt(denominator))
            self.assertAlmostEqual(row["metrics"]["correlation"], np.sum((a * b.conjugate()).real) / denominator)
            self.assertEqual(row["budget"], self.budget)

    def test_zero_and_subfloor_signals_are_undefined_not_passed(self):
        for value in (0., 1e-16):
            with self.subTest(value=value):
                rows = study.shell_comparisons(np.full(5, value), np.full(5, value), self.modes, self.budget)
                self.assertTrue(rows)
                for row in rows:
                    self.assertFalse(row["passed"])
                    self.assertFalse(row["metrics"]["normalization_defined"])
                    for key in ("power_ratio", "complex_relative", "correlation"):
                        self.assertIsNone(row["metrics"][key])
        rows = study.shell_comparisons(np.full(5, 1e-12), np.full(5, 1e-12), self.modes, self.budget)
        self.assertTrue(all(row["passed"] for row in rows))

    def test_nonfinite_signal_rejected_and_encode_decode_roundtrip_exact(self):
        coefficients = np.array([1. + .4j, -.2 - .7j, 2j, .1, -3.])
        np.testing.assert_array_equal(study.decode_modes(study.encode_modes(coefficients)), coefficients)
        coefficients[0] = np.nan
        with self.assertRaises(ValueError):
            study.shell_comparisons(coefficients, np.ones(5), self.modes, self.budget)


class QualificationTests(unittest.TestCase):
    def test_contiguous_prefix_never_jumps_a_failed_or_missing_shell(self):
        for failed in range(3):
            checks = [shell_check(shell, shell != failed) for shell in range(3)]
            result = study.qualified_prefix(checks, "spatial", "coupled3d", True)
            self.assertEqual(result["contiguous_shell_count"], failed)
            self.assertEqual(result["upper_mode_exclusive"], float(study.ShellEdges[failed]) if failed else None)
        checks = [shell_check(0), shell_check(2)]
        self.assertEqual(study.qualified_prefix(checks, "spatial", "coupled3d", True)["contiguous_shell_count"], 1)
        self.assertEqual(study.qualified_prefix([], "spatial", "coupled3d", True)["contiguous_shell_count"], 0)

    def test_global_veto_and_any_applicable_failure(self):
        checks = [shell_check(shell) for shell in range(3)]
        self.assertEqual(study.qualified_prefix(checks, "spatial", "coupled3d", True)["contiguous_shell_count"], 3)
        result = study.qualified_prefix(checks, "spatial", "coupled3d", False)
        self.assertEqual(result["contiguous_shell_count"], 0)
        self.assertTrue(all(not row["passed"] for row in result["shells"]))
        checks.append(shell_check(0, False, category="translation"))
        self.assertEqual(study.qualified_prefix(checks, "spatial", "coupled3d", True)["contiguous_shell_count"], 0)

    def test_unrelated_fixture_stage_and_high_characterization_shells_do_not_extend_band(self):
        checks = [shell_check(shell) for shell in range(3)]
        checks += [shell_check(0, False, fixture="pancake"),
                   shell_check(0, False, stage="gaussian", fixture="gaussian"), shell_check(3)]
        result = study.qualified_prefix(checks, "spatial", "coupled3d", True)
        self.assertEqual(result["contiguous_shell_count"], 3)
        self.assertEqual(result["hard_maximum_measured_mode"], 4)
        self.assertEqual(result["upper_mode_exclusive"], 4.5)
        self.assertEqual([row["checks"] for row in result["shells"]], [1, 1, 1])

    def test_qualify_integration_global_time_failure_vetoes_otherwise_passing_band(self):
        instance = bare_study(stage="spatial")
        instance.report["completed_stages"] = ["spatial"]
        instance.report["checks"] = [shell_check(shell) for shell in range(3)]
        instance.report["checks"].append({"name": "time/finest_momentum", "stage": "spatial",
            "fixture": "coupled3d", "category": "time_global", "passed": False})
        instance.qualify()
        result = instance.report["qualification"]["coupled3d"]
        self.assertEqual(result["contiguous_shell_count"], 0)
        self.assertFalse(result["required_global_checks_passed"])
        self.assertEqual(result["global_failures"], ["time/finest_momentum"])
        instance.report["checks"][-1]["passed"] = True
        instance.qualify()
        self.assertEqual(instance.report["qualification"]["coupled3d"]["contiguous_shell_count"], 3)
        self.assertEqual(instance.report["qualification"]["pancake"]["contiguous_shell_count"], 0)


class TranslationAndControlTests(unittest.TestCase):
    def test_rigid_translation_fourier_sign_and_phase_space_undo(self):
        instance = bare_study(stage="spatial")
        case = study.Case("spatial", "coupled3d", 32, 32, 1024)
        shifted = replace(case, shifted=True)
        frame = particle_frame()
        moved = frame.copy()
        box = study.Parameters["box_size"]
        delta = study.translation(shifted)
        moved[["x", "y", "z"]] = (moved[["x", "y", "z"]] + delta) % box
        modes = study.baseline.resolved_modes("coupled3d")
        left = study.baseline.density_modes(frame[["x", "y", "z"]].to_numpy(), box, modes)
        right = study.baseline.density_modes(moved[["x", "y", "z"]].to_numpy(), box, modes)
        np.testing.assert_allclose(right, left * np.exp(-2j * np.pi / box * (modes @ delta)), atol=4e-15, rtol=0)
        corrected = right * np.exp(2j * np.pi / box * (modes @ delta))
        np.testing.assert_allclose(corrected, left, atol=4e-15, rtol=0)
        original_bytes = moved.to_numpy().tobytes()
        with patch.object(instance, "snapshot", side_effect=lambda name, _: moved if "_shift_" in name else frame):
            instance.compare_phase(shifted, "ippl", case, "ippl", "translation_phase")
        self.assertEqual(len(instance.report["comparisons"]), 9)
        for row in instance.report["comparisons"]:
            self.assertLess(row["phase_space"]["position_cells"], 2e-15)
            self.assertEqual(row["phase_space"]["momentum_relative"], 0.)
        self.assertEqual(moved.to_numpy().tobytes(), original_bytes)
        self.assertFalse(instance.report["checks"])  # Raw 3D translation phase is characterized, not newly gated.

    def test_spectral_comparisons_reject_different_epochs_without_interpolation(self):
        instance = bare_study()
        left = study.Case("gaussian", "gaussian", 64, 64, 4096)
        right = replace(left, redshift=99)
        records = {left.name: {"density": [{"a": .2}] * 9}, right.name: {"density": [{"a": .21}] * 9}}
        with patch.object(instance, "run_record", side_effect=lambda case, _: records[case.name]):
            with self.assertRaisesRegex(ValueError, "epochs differ"):
                instance.compare_spectra(left, "ippl", right, "ippl", "start_redshift", study.Budgets["start_redshift"], (8,))
        self.assertFalse(instance.report["checks"])

    def test_comparison_matrix_references_only_planned_cases_and_shared_inputs(self):
        instance = bare_study()
        instance.cases = study.study_cases()
        instance.report["runs"] = [{"name": case.name + "_" + code}
                                   for case in instance.cases for code in study.Codes]
        calls = []
        def spectra(left, lcode, right, rcode, category, budget, checkpoints=range(9)):
            calls.append((left, right, category, tuple(checkpoints)))
        def phase(left, lcode, right, rcode, category):
            calls.append((left, right, category, tuple(range(9))))
        def temporal(group, code, order=True):
            self.assertEqual(len(group), 3 if order else 2)
            self.assertTrue(all(case in instance.cases for case in group))
            self.assertTrue(all(a.steps * 2 == b.steps for a, b in zip(group[:-1], group[1:])))
            calls.append((group[0], group[-1], "temporal", (4, 8)))
        with (patch.object(instance, "compare_spectra", side_effect=spectra),
              patch.object(instance, "compare_phase", side_effect=phase),
              patch.object(instance, "temporal", side_effect=temporal)):
            instance.comparisons()
        self.assertTrue(calls)
        for left, right, category, checkpoints in calls:
            self.assertIn(left, instance.cases)
            self.assertIn(right, instance.cases)
            if category in ("rank", "mesh_resolution", "temporal"):
                self.assertEqual(left.input_key, right.input_key)
            if category == "start_redshift":
                self.assertEqual(checkpoints, (8,))
                self.assertEqual({left.redshift, right.redshift}, {49, 99})
            if category.startswith("translation"):
                self.assertNotEqual(left.shifted, right.shifted)
                self.assertEqual(left.particles, right.particles)
                self.assertEqual(left.mesh, right.mesh)

    def test_derived_checks_regenerated_without_dropping_measurement_or_integrity(self):
        instance = bare_study(smoke=True)
        instance.cases = []
        instance.report["checks"] = [{"name": category, "category": category, "passed": False}
                                     for category in ("integrity", "measurement", "cross", "time_global", "rank")]
        instance.report["comparisons"] = [{"old": True}]
        instance.report["time_refinement"] = [{"old": True}]
        instance.comparisons()
        self.assertEqual([check["category"] for check in instance.report["checks"]], ["integrity", "measurement"])
        self.assertFalse(instance.report["comparisons"])
        self.assertFalse(instance.report["time_refinement"])

    def test_completed_spatial_stage_qualifies_while_gaussian_is_partial(self):
        instance = bare_study()
        instance.cases = study.study_cases()
        spatial = [case for case in instance.cases if case.stage == "spatial"]
        first_gaussian = next(case for case in instance.cases if case.stage == "gaussian")
        instance.report["runs"] = [{"name": case.name + "_" + code}
                                   for case in spatial + [first_gaussian] for code in study.Codes]
        instance.report.update(complete=False, passed=False)
        instance.report["checks"] = [{"name": fixture + "/integrity", "passed": True,
                                      "stage": "spatial", "fixture": fixture, "category": "integrity"}
                                     for fixture in ("pancake", "coupled3d")]
        # An incomplete Gaussian stage must neither be qualified nor veto an
        # independently complete spatial stage.
        instance.report["checks"].append({"name": "gaussian/partial", "passed": False,
            "stage": "gaussian", "fixture": "gaussian", "category": "integrity"})
        compared = []
        def spectra(left, lcode, right, rcode, category, budget, checkpoints=range(9)):
            compared.extend((left, right))
            instance.report["checks"].extend(shell_check(shell, stage=left.stage, fixture=left.fixture,
                                                         category=category) for shell in range(3))
        def phase(left, lcode, right, rcode, category):
            compared.extend((left, right))
        def temporal(group, code, order=True):
            compared.extend(group)
        with (patch.object(instance, "compare_spectra", side_effect=spectra),
              patch.object(instance, "compare_phase", side_effect=phase),
              patch.object(instance, "temporal", side_effect=temporal)):
            instance.comparisons()
            instance.qualify()
            self.assertEqual(instance.report["completed_stages"], ["spatial"])
            self.assertTrue(compared)
            self.assertTrue(all(case.stage == "spatial" for case in compared))
            self.assertEqual(set(instance.report["qualification"]), {"pancake", "coupled3d"})
            self.assertTrue(all(row["contiguous_shell_count"] == 3
                                for row in instance.report["qualification"].values()))
            self.assertFalse(instance.report["complete"])
            self.assertFalse(instance.report["passed"])
            # Removing even one analyzed run withdraws the stage qualification.
            instance.report["runs"].pop(0)
            instance.comparisons()
            instance.qualify()
            self.assertEqual(instance.report["completed_stages"], [])
            self.assertEqual(instance.report["qualification"], {})

    def test_batch_stop_retains_incomplete_status_and_resume_does_not_reanalyze_completed_run(self):
        instance = bare_study(smoke=True)
        instance.root = Path("/unused-mocked-study")
        instance.hashes = {}
        instance.cases = study.study_cases(smoke=True)
        instance.report.update(fixtures={}, state="prepared", complete=False, passed=False,
                               expected_runs=8, failed_checks=[])
        instance.args = Namespace(stage="all", smoke=True, stop_after=1, mpiexec="not-launched",
                                  mpi_arg=[], numproc_flag="-n", timeout=1)
        instance.executables = {code: Path("/not-launched-" + code) for code in study.Codes}
        instance.storage = Mock()
        instance.storage.output_dir.side_effect = lambda name: instance.root / "storage" / name
        analyzed = []
        def analyze(case, code, archived, original):
            name = case.name + "_" + code
            analyzed.append(name)
            instance.report["runs"].append({"name": name})
            instance.report["checks"].append({"name": name + "/integrity", "passed": True, "category": "integrity"})
        def save():
            instance.report["failed_checks"] = [check for check in instance.report["checks"] if not check["passed"]]
        with (patch.object(instance, "fixture", return_value=(particle_frame(), {"path": "/unchanged-input.csv"})),
              patch.object(instance, "analyze_run", side_effect=analyze),
              patch.object(instance, "save", side_effect=save),
              patch.object(instance, "comparisons") as comparisons,
              redirect_stdout(io.StringIO())):
            self.assertEqual(instance.run(), 0)
            self.assertEqual(len(analyzed), 1)
            self.assertEqual(instance.report["state"], "batch_complete")
            self.assertFalse(instance.report["complete"])
            self.assertFalse(instance.report["passed"])
            comparisons.assert_not_called()
            instance.args.stop_after = None
            self.assertEqual(instance.run(), 0)
        self.assertEqual(len(analyzed), 8)
        self.assertEqual(len(set(analyzed)), 8)
        self.assertEqual(instance.storage.execute.call_count, 9)  # First completed archive reverified on resume.
        self.assertTrue(instance.report["complete"])
        self.assertTrue(instance.report["passed"])
        self.assertEqual(instance.report["qualification"], {"scope": "Pipeline smoke only, no resolution qualification"})

    def test_disk_block_main_returns_nonzero_and_never_reports_complete_or_passed(self):
        instance = Mock()
        instance.report = {"state": "running", "complete": False, "passed": False}
        instance.run.side_effect = study.DiskSpaceBlocked("synthetic low disk")
        with (patch.object(study, "Study", return_value=instance),
              patch.object(sys, "argv", ["validate_resolution_study.py"]), redirect_stdout(io.StringIO())):
            self.assertEqual(study.main(), 3)
        self.assertEqual(instance.report["state"], "blocked_disk")
        self.assertFalse(instance.report["complete"])
        self.assertFalse(instance.report["passed"])
        self.assertEqual(instance.report["disk_message"], "synthetic low disk")
        instance.save.assert_called_once()
        instance.storage.close.assert_called_once()

    def test_exact_spatial_batch_boundary_persists_qualification_before_return(self):
        instance = bare_study()
        instance.root = Path("/unused-mocked-stage-boundary")
        instance.hashes = {}
        instance.cases = study.study_cases()
        spatial = [case for case in instance.cases if case.stage == "spatial"]
        boundary = 2 * len(spatial)
        instance.report.update(fixtures={}, state="prepared", complete=False, passed=False,
                               expected_runs=52, failed_checks=[])
        instance.args = Namespace(stage="all", smoke=False, stop_after=boundary, mpiexec="not-launched",
                                  mpi_arg=[], numproc_flag="-n", timeout=1)
        instance.executables = {code: Path("/not-launched-" + code) for code in study.Codes}
        instance.storage = Mock()
        instance.storage.output_dir.side_effect = lambda name: instance.root / "storage" / name
        saved = []
        def fixture(case):
            self.assertEqual(case.stage, "spatial", "Batch must return before preparing Gaussian input")
            return particle_frame(), {"path": "/unchanged-input.csv"}
        def analyze(case, code, archived, original):
            name = case.name + "_" + code
            instance.report["runs"].append({"name": name})
            instance.report["checks"].append({"name": name + "/integrity", "passed": True,
                "category": "integrity", "stage": case.stage, "fixture": case.fixture})
        def spectra(left, lcode, right, rcode, category, budget, checkpoints=range(9)):
            self.assertEqual(left.stage, "spatial")
            self.assertEqual(right.stage, "spatial")
            instance.report["checks"].extend(shell_check(shell, stage=left.stage,
                fixture=left.fixture, category=category) for shell in range(3))
        def save():
            instance.report["failed_checks"] = [check for check in instance.report["checks"] if not check["passed"]]
            saved.append(deepcopy(instance.report))
        with (patch.object(instance, "fixture", side_effect=fixture),
              patch.object(instance, "analyze_run", side_effect=analyze),
              patch.object(instance, "compare_spectra", side_effect=spectra),
              patch.object(instance, "compare_phase"), patch.object(instance, "temporal"),
              patch.object(instance, "save", side_effect=save), redirect_stdout(io.StringIO())):
            self.assertEqual(instance.run(), 0)
        self.assertEqual(instance.storage.execute.call_count, boundary)
        self.assertEqual(len(instance.report["runs"]), 34)
        self.assertEqual(instance.report["state"], "batch_complete")
        self.assertFalse(instance.report["complete"])
        self.assertFalse(instance.report["passed"])
        self.assertEqual(saved[-1]["completed_stages"], ["spatial"])
        self.assertEqual(set(saved[-1]["qualification"]), {"pancake", "coupled3d"})
        self.assertTrue(all(row["contiguous_shell_count"] == 3 for row in saved[-1]["qualification"].values()))


class ConfigurationTests(unittest.TestCase):
    def test_resume_keeps_original_case_matrix_and_rejects_changed_execution_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for filename in ("ippl", "fastpm", "manifest"):
                (root / filename).write_text("mock provenance\n")
            args = Namespace(ippl_exe=root / "ippl", fastpm_exe=root / "fastpm", fastpm_manifest=root / "manifest",
                stage="spatial", smoke=False, output_dir=root / "study", resume=None, mpiexec="mpiexec",
                numproc_flag="-n", mpi_arg=[], reserve_gib=1., timeout=1., stop_after=None)
            with (patch.object(study, "StudyStorage"), patch.object(study, "verify_native_manifest", return_value={})):
                created = study.Study(args)
                self.assertEqual(created.report["expected_runs"], 34)
                self.assertFalse(created.report["complete"])
                self.assertFalse(created.report["passed"])
                resume_args = vars(args) | {"resume": root / "study", "output_dir": None,
                    "stage": None, "ippl_exe": None, "fastpm_exe": None, "fastpm_manifest": None}
                resumed = study.Study(Namespace(**resume_args))
                self.assertEqual(resumed.cases, created.cases)
                self.assertEqual(resumed.args.ippl_exe, root / "ippl")
                for key, value in (("stage", "gaussian"), ("mpiexec", "other-mpi"),
                                   ("mpi_arg", ["--changed"]), ("numproc_flag", "-np"),
                                   ("ippl_exe", root / "different"), ("smoke", True)):
                    with self.subTest(key=key):
                        with self.assertRaises(ValueError):
                            study.Study(Namespace(**(resume_args | {key: value})))
                result_path = root / "study" / "results.json"
                unchanged = json.loads(result_path.read_text())
                modified = deepcopy(unchanged)
                modified["planned_cases"] = modified["planned_cases"][::-1]
                result_path.write_text(json.dumps(modified))
                with self.assertRaisesRegex(ValueError, "case matrix changed"):
                    study.Study(Namespace(**resume_args))
                modified = deepcopy(unchanged)
                modified["budgets"]["spatial"]["complex"] = .2
                result_path.write_text(json.dumps(modified))
                with self.assertRaisesRegex(ValueError, "protocol differs"):
                    study.Study(Namespace(**resume_args))
                result_path.write_text(json.dumps(unchanged))
                (root / "ippl").write_text("changed executable\n")
                with self.assertRaisesRegex(ValueError, "differs"):
                    study.Study(Namespace(**resume_args))

    def test_prepared_initialization_with_absent_or_empty_storage_recovers(self):
        for empty_directory in (False, True):
            with self.subTest(empty_directory=empty_directory), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                for filename in ("ippl", "fastpm", "manifest"):
                    (root / filename).write_text("mock provenance\n")
                args = Namespace(ippl_exe=root / "ippl", fastpm_exe=root / "fastpm", fastpm_manifest=root / "manifest",
                    stage="spatial", smoke=False, output_dir=root / "study", resume=None, mpiexec="mpiexec",
                    numproc_flag="-n", mpi_arg=[], reserve_gib=1., timeout=1., stop_after=None)
                def interrupted_initialization(storage_root, *unused, **options):
                    if empty_directory:
                        storage_root.mkdir()
                    raise RuntimeError("Synthetic stop before storage journal")
                with (patch.object(study, "StudyStorage", side_effect=interrupted_initialization),
                      patch.object(study, "verify_native_manifest", return_value={})):
                    with self.assertRaisesRegex(RuntimeError, "Synthetic stop"):
                        study.Study(args)
                report_path = root / "study" / "results.json"
                prepared = json.loads(report_path.read_text())
                self.assertEqual(prepared["state"], "prepared")
                self.assertEqual(len(prepared["planned_cases"]), 17)
                self.assertEqual(prepared["expected_runs"], 34)
                self.assertFalse(prepared["complete"])
                self.assertFalse(prepared["passed"])
                resume_args = vars(args) | {"resume": root / "study", "output_dir": None,
                    "stage": None, "ippl_exe": None, "fastpm_exe": None, "fastpm_manifest": None}
                resumed = study.Study(Namespace(**resume_args))
                try:
                    self.assertEqual(resumed.cases, study.study_cases("spatial"))
                    self.assertTrue((root / "study" / "storage" / "storage.json").is_file())
                    self.assertEqual(len(resumed.report["source_snapshot"]), len(study.SourceNames))
                    for path, digest in resumed.report["source_snapshot"].items():
                        self.assertEqual(study.sha256(path), digest)
                    self.assertEqual(resumed.report["state"], "prepared")
                finally:
                    resumed.storage.close()

    def test_prepared_recovery_rejects_nonempty_unjournaled_storage_and_nonprepared_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for filename in ("ippl", "fastpm", "manifest"):
                (root / filename).write_text("mock provenance\n")
            args = Namespace(ippl_exe=root / "ippl", fastpm_exe=root / "fastpm", fastpm_manifest=root / "manifest",
                stage="spatial", smoke=False, output_dir=root / "study", resume=None, mpiexec="mpiexec",
                numproc_flag="-n", mpi_arg=[], reserve_gib=1., timeout=1., stop_after=None)
            with (patch.object(study, "StudyStorage", side_effect=RuntimeError("Synthetic stop")),
                  patch.object(study, "verify_native_manifest", return_value={})):
                with self.assertRaises(RuntimeError):
                    study.Study(args)
            resume_args = vars(args) | {"resume": root / "study", "output_dir": None,
                "stage": None, "ippl_exe": None, "fastpm_exe": None, "fastpm_manifest": None}
            report_path = root / "study" / "results.json"
            prepared = json.loads(report_path.read_text())
            nonprepared = deepcopy(prepared)
            nonprepared["state"] = "running"
            report_path.write_text(json.dumps(nonprepared))
            with self.assertRaisesRegex(ValueError, "Incomplete storage initialization"):
                study.Study(Namespace(**resume_args))
            report_path.write_text(json.dumps(prepared))
            storage_root = root / "study" / "storage"
            storage_root.mkdir()
            retained = storage_root / "unrecognized-evidence.txt"
            retained.write_text("retain these bytes\n")
            with self.assertRaisesRegex(ValueError, "Incomplete storage initialization"):
                study.Study(Namespace(**resume_args))
            self.assertEqual(retained.read_text(), "retain these bytes\n")


if __name__ == "__main__":
    unittest.main()
