## @file test_plot_resolution_study.py
# @brief Bounded synthetic report tests; no executable, particle archive or MPI I/O.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
# Tests assert declared invariants using isolated fixtures; no scientific tolerances are relaxed by documentation.
"""Bounded synthetic report tests; no executable, particle archive or MPI I/O."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plot_resolution_study as plot
import validate_resolution_study as protocol


## @brief Complete numerical report shape, tiny synthetic vectors, no particles.
# @see cosmology_tools
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def synthetic_report():
    """Complete numerical report shape, tiny synthetic vectors, no particles."""
    cases = protocol.study_cases()
    report = {"schema": "ippl-resolution-study-v1", "configuration": {"smoke": False},
              "parameters": deepcopy(protocol.Parameters), "budgets": deepcopy(protocol.Budgets),
              "planned_cases": [case.__dict__ for case in cases], "completed_stages": ["spatial", "gaussian"],
              "runs": [], "checks": [], "failed_checks": [], "comparisons": [], "time_refinement": [],
              "qualification": {name: {"contiguous_shell_count": 3, "hard_maximum_measured_mode": 4}
                                for name in ("pancake", "coupled3d", "gaussian")},
              "limitations": ["Synthetic plot test, not physical results"], "complete": True, "passed": True}
    by_name = {}
    for case in cases:
        ai, af = protocol.epochs(case)
        modes = plot.modes_for(case.fixture)
        radius = np.linalg.norm(modes, axis=1)
        for code in plot.Codes:
            run = {**case.__dict__, "name": case.name+"_"+code, "code": code, "density": []}
            factor = (1+.0001*(case.particles == 32)+.0002*(case.mesh == 32)
                      +.0003*case.shifted+.0004*(case.redshift == 99)+.00001*(code == "fastpm"))
            for checkpoint in range(9):
                a = ai*np.exp(checkpoint*np.log(af/ai)/8)
                coefficients = a*factor*(1+.3j)/(1+radius**2)
                row = {"checkpoint": checkpoint, "a": float(a),
                       "direct": {"real": coefficients.real.tolist(), "imag": coefficients.imag.tolist()}}
                # Only the final FFT data are consumed by the renderer. Keep
                # this fixture bounded rather than manufacture unneeded data.
                if case.fixture == "gaussian" and case.steps == 2048 and checkpoint == 8:
                    full_modes = plot.modes_for("gaussian", 12)
                    full = a*factor*(1+.3j)/(1+np.sum(full_modes**2, axis=1))
                    row["fft"] = {"real": full.real.tolist(), "imag": full.imag.tolist()}
                run["density"].append(row)
            report["runs"].append(run)
            by_name[run["name"]] = run

    def compare(left, lcode, right, rcode, category, checkpoints=range(9)):
        for checkpoint in checkpoints:
            lrun, rrun = by_name[left.name+"_"+lcode], by_name[right.name+"_"+rcode]
            modes = plot.modes_for(left.fixture)
            radius = np.linalg.norm(modes, axis=1)
            a, b = (plot.decode(run["density"][checkpoint]["direct"], len(modes)) for run in (lrun, rrun))
            shells = []
            for index, (lower, upper) in enumerate(zip(plot.Edges[:3], plot.Edges[1:4])):
                mask = (radius >= lower) & (radius < upper)
                shells.append({"shell": index, "metrics": plot.metrics(a[mask], b[mask]), "passed": True})
            report["comparisons"].append({"category": category, "left": lrun["name"], "right": rrun["name"],
                                          "checkpoint": checkpoint, "shells": shells})

    def phase_compare(left, lcode, right, rcode, category):
        for checkpoint in range(9):
            report["comparisons"].append({"category": category, "left": left.name+"_"+lcode,
                "right": right.name+"_"+rcode, "checkpoint": checkpoint,
                "phase_space": {"position_cells": 0., "momentum_relative": 0.}})

    for case in cases:
        compare(case, "ippl", case, "fastpm", "cross")
        phase_compare(case, "ippl", case, "fastpm", "cross_phase")
        if case.ranks > 1:
            for code in plot.Codes:
                phase_compare(case, code, protocol.replace(case, ranks=1), code, "rank")
    for fixture in ("pancake", "coupled3d", "gaussian"):
        stage = "gaussian" if fixture == "gaussian" else "spatial"
        steps = 2048 if stage == "gaussian" else 1024
        for code in plot.Codes:
            for fixed in (32, 64):
                compare(protocol.Case(stage, fixture, 32, fixed, steps), code,
                        protocol.Case(stage, fixture, 64, fixed, steps), code, "particle_resolution")
                compare(protocol.Case(stage, fixture, fixed, 32, steps), code,
                        protocol.Case(stage, fixture, fixed, 64, steps), code, "mesh_resolution")
                if stage == "spatial":
                    compare(protocol.Case(stage, fixture, 64, fixed, steps, shifted=True), code,
                            protocol.Case(stage, fixture, 64, fixed, steps), code, "translation")
                    phase_compare(protocol.Case(stage, fixture, 64, fixed, steps, shifted=True), code,
                                  protocol.Case(stage, fixture, 64, fixed, steps), code, "translation_phase")
            for checkpoint in (4, 8):
                report["time_refinement"].append({"fixture": fixture, "stage": stage, "code": code,
                    "redshift": 49, "checkpoint": checkpoint, "steps": [steps//2, steps, steps*2],
                    "phase_space_differences": [{"momentum_relative": .0008}, {"momentum_relative": .0002}],
                    "orders": {"momentum_relative": {"status": "measured", "measured_order": 2.}}})
            if stage == "gaussian":
                compare(protocol.Case(stage, fixture, 64, 64, 4096), code,
                        protocol.Case(stage, fixture, 64, 64, 4096, redshift=99), code,
                        "start_redshift", checkpoints=(8,))
                for checkpoint in (4, 8):
                    report["time_refinement"].append({"fixture": fixture, "stage": stage, "code": code,
                        "redshift": 99, "checkpoint": checkpoint, "steps": [2048, 4096],
                        "phase_space_differences": [{"momentum_relative": .00025}], "orders": {}})
    return json.loads(json.dumps(report, allow_nan=False))


## @brief Regression suite for ResolutionPlot.
# @see cosmology_tools
class ResolutionPlotTests(unittest.TestCase):
    ## @brief Evaluate the setUpClass helper in the documented module workflow.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    @classmethod
    def setUpClass(cls):
        ## @var report
        # @brief Structured campaign/audit report; recorded failures are not retroactively changed.
        cls.report = synthetic_report()

    ## @brief Verify complete stage coverage and output data.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_complete_stage_coverage_and_output_data(self):
        data = plot.extract_data(self.report)
        self.assertEqual(data["completed_stages"], ["spatial", "gaussian"])
        self.assertEqual(len(data["spatial"]["pancake"]["resolution"]), 4)
        self.assertEqual(len(data["gaussian"]["spectra"]), 4)
        self.assertEqual([p["measurement"] for p in data["gaussian"]["spectra"][0]["points"]],
                         ["direct"]*3 + ["FFT characterization"]*4)
        json.dumps(data, allow_nan=False)

    ## @brief Verify partial campaign renders only complete stage.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_partial_campaign_renders_only_complete_stage(self):
        report = deepcopy(self.report)
        report["complete"], report["passed"] = False, False
        report["completed_stages"] = ["spatial"]
        report["runs"] = [row for row in report["runs"] if row["stage"] == "spatial"]
        report["comparisons"] = [row for row in report["comparisons"] if not row["left"].startswith("gaussian")]
        report["time_refinement"] = [row for row in report["time_refinement"] if row["stage"] == "spatial"]
        del report["qualification"]["gaussian"]
        data = plot.extract_data(report)
        self.assertEqual(data["completed_stages"], ["spatial"])
        self.assertIn("gaussian", data["omitted_stages"])
        self.assertEqual(data["gaussian"], {})
        self.assertFalse(data["campaign_passed"])

    ## @brief Verify runs alone do not imply completed analysis.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_runs_alone_do_not_imply_completed_analysis(self):
        report = deepcopy(self.report)
        report["completed_stages"] = []
        with self.assertRaisesRegex(ValueError, "No complete analyzed stage"):
            plot.extract_data(report)
        report["completed_stages"] = ["spatial", "gaussian"]
        report["time_refinement"] = []
        with self.assertRaisesRegex(ValueError, "No complete analyzed stage"):
            plot.extract_data(report)

    ## @brief Verify truncated comparison or time evidence withholds stage.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_truncated_comparison_or_time_evidence_withholds_stage(self):
        for kind in ("comparison", "time", "duplicate"):
            report = deepcopy(self.report)
            if kind == "comparison":
                index = next(i for i, row in enumerate(report["comparisons"])
                             if row["category"] == "particle_resolution" and row["left"].startswith("gaussian"))
                report["comparisons"].pop(index)
            elif kind == "time":
                index = next(i for i, row in enumerate(report["time_refinement"])
                             if row["stage"] == "gaussian" and row["redshift"] == 99 and row["checkpoint"] == 4)
                report["time_refinement"].pop(index)
            else:
                row = next(row for row in report["comparisons"] if row["left"].startswith("gaussian"))
                report["comparisons"].append(deepcopy(row))
            with self.subTest(kind=kind):
                data = plot.extract_data(report)
                self.assertEqual(data["completed_stages"], ["spatial"])
                self.assertIn("evidence", data["omitted_stages"]["gaussian"])
    ## @brief Verify missing planned run omits its stage.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_missing_planned_run_omits_its_stage(self):
        report = deepcopy(self.report)
        report["runs"] = report["runs"][1:]
        data = plot.extract_data(report)
        self.assertEqual(data["completed_stages"], ["gaussian"])
        self.assertIn("spatial", data["omitted_stages"])

    ## @brief Verify failures and zero prefix survive plotting.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_failures_and_zero_prefix_survive_plotting(self):
        report = deepcopy(self.report)
        failure = {"name": "pancake/time", "stage": "spatial", "fixture": "pancake", "passed": False}
        report["checks"] = [failure]
        report["failed_checks"] = [failure]
        report["passed"] = False
        report["qualification"]["pancake"]["contiguous_shell_count"] = 0
        data = plot.extract_data(report)
        self.assertEqual(data["failed_checks"], [failure])
        self.assertIn("1 failed checks retained", plot.status_text(data, "spatial"))
        self.assertIn("pancake: 0/3", plot.status_text(data, "spatial"))
        self.assertEqual(data["qualification"]["pancake"]["hard_maximum_measured_mode"], 4)

    ## @brief Verify corrupted recorded metric is rejected.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_corrupted_recorded_metric_is_rejected(self):
        report = deepcopy(self.report)
        report["comparisons"][0]["shells"][0]["metrics"]["power_ratio"] *= 1.01
        with self.assertRaisesRegex(ValueError, "metric disagrees"):
            plot.extract_data(report)

    ## @brief Verify wrong density epoch or vector is rejected.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_wrong_density_epoch_or_vector_is_rejected(self):
        report = deepcopy(self.report)
        report["runs"][0]["density"][1]["a"] *= 1.1
        with self.assertRaisesRegex(ValueError, "epochs do not align"):
            plot.extract_data(report)
        report = deepcopy(self.report)
        report["runs"][0]["density"][0]["direct"]["real"] = [1.]
        with self.assertRaisesRegex(ValueError, "vector length"):
            plot.extract_data(report)

    ## @brief Verify direct power normalization never uses low band fft.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_direct_power_normalization_never_uses_low_band_fft(self):
        report = deepcopy(self.report)
        row = next(run for run in report["runs"] if run["fixture"] == "gaussian"
                   and run["particles"] == run["mesh"] == 32 and run["code"] == "ippl")
        row["density"][8]["fft"]["real"] = (np.asarray(row["density"][8]["fft"]["real"])*5).tolist()
        data = plot.extract_data(report)
        first = data["gaussian"]["spectra"][0]["points"][0]
        modes = plot.modes_for("gaussian")
        coefficients = plot.decode(row["density"][8]["direct"], len(modes))
        mask = np.linalg.norm(modes, axis=1) < 1.5
        expected = 168.75**3 * np.mean(np.abs(coefficients[mask])**2)
        self.assertAlmostEqual(first["power_ippl"], expected, delta=expected*1e-14)
        self.assertEqual(first["measurement"], "direct")

    ## @brief Verify envelope uses max not mean and retains undefined failures.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_envelope_uses_max_not_mean_and_retains_undefined_failures(self):
        rows = []
        for value in (.01, .07):
            rows.append({"fixture": "pancake", "category": "translation", "mesh": 32,
                "particles": 64, "shell": 0, "passed": value < .02,
                "metrics": {"normalization_defined": True, "power_ratio": 1+value,
                            "complex_relative": 2*value, "correlation": 1-value}})
        rows.append({**rows[0], "passed": False, "metrics": {"normalization_defined": False}})
        point = plot._envelope(rows, "pancake", "translation")[0]["points"][0]
        self.assertAlmostEqual(point["power_fraction_max"], .07)
        self.assertEqual(point["complex_relative_max"], .14)
        self.assertEqual(point["undefined_comparisons"], 1)
        self.assertEqual(point["failed_comparisons"], 2)
        self.assertEqual(plot.metrics(np.zeros(2), np.zeros(2))["complex_relative"], None)
        self.assertEqual(plot.metrics(np.ones(2), np.ones(2))["complex_relative"], 0.)

    ## @brief Verify zero spectrum undefined correlation render as gaps without floor.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_zero_spectrum_undefined_correlation_render_as_gaps_without_floor(self):
        data = plot.extract_data(self.report)
        for series in data["gaussian"]["spectra"]:
            for point in series["points"]:
                point["power_ippl"] = point["power_fastpm"] = 0.
                for key in ("power_ratio", "complex_relative", "correlation"):
                    point["comparison"][key] = None
                point["comparison"]["normalization_defined"] = False
        figure = plot.gaussian_figure(data)
        axis = figure.axes[0]
        self.assertEqual(axis.get_yscale(), "linear")
        self.assertTrue(all(np.isnan(line.get_ydata()).all() for line in axis.lines if len(line.get_ydata()) > 2))
        self.assertTrue(any("No positive shell power" in text.get_text() for text in axis.texts))
        self.assertIn("undefined", figure.axes[2].get_title())
        self.assertIn("complex residual", figure.axes[2].get_title())
        self.assertEqual(data["gaussian"]["spectra"][0]["points"][0]["power_ippl"], 0.)
        plot.plt.close(figure)

    ## @brief Verify mixed nonpositive powers are gaps on remaining log axis.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_mixed_nonpositive_powers_are_gaps_on_remaining_log_axis(self):
        data = plot.extract_data(self.report)
        first = data["gaussian"]["spectra"][0]["points"][0]
        first["power_ippl"], first["power_fastpm"] = 0., -1.
        figure = plot.gaussian_figure(data)
        axis = figure.axes[0]
        self.assertEqual(axis.get_yscale(), "log")
        self.assertTrue(np.isnan(axis.lines[0].get_ydata()[0]))
        self.assertTrue(np.isnan(axis.lines[1].get_ydata()[0]))
        self.assertTrue(any("no floor" in text.get_text() for text in axis.texts))
        self.assertEqual(first["power_fastpm"], -1.)
        plot.plt.close(figure)

    ## @brief Verify status consistency smoke and duplicate run guards.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_status_consistency_smoke_and_duplicate_run_guards(self):
        for mode in ("smoke", "passed", "duplicate"):
            report = deepcopy(self.report)
            if mode == "smoke":
                report["configuration"]["smoke"] = True
            elif mode == "passed":
                report["passed"] = False
            else:
                report["runs"].append(deepcopy(report["runs"][0]))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                plot.extract_data(report)

    ## @brief Verify render manifest hashes no overwrite and no report mutation.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_render_manifest_hashes_no_overwrite_and_no_report_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory/"results.json"
            source.write_text(json.dumps(self.report))
            before = plot.sha256(source)
            target = directory/"figures"
            manifest = plot.render(source, target)
            self.assertEqual(before, plot.sha256(source))
            self.assertEqual(manifest["input_report"]["sha256"], before)
            self.assertEqual(len(manifest["outputs"]), 6)
            self.assertLess(sum(row["bytes"] for row in manifest["outputs"]), 8*1024**2)
            for row in manifest["outputs"]:
                self.assertEqual(plot.sha256(row["path"]), row["sha256"])
            self.assertTrue((target/"manifest.json").is_file())
            self.assertEqual(hashlib.sha256(gzip.decompress((target/"input-report.json.gz").read_bytes())).hexdigest(), before)
            self.assertTrue(manifest["source_report_unchanged_during_plot"])
            with self.assertRaises(FileExistsError):
                plot.render(source, target)

    ## @brief Verify live report update uses preserved input snapshot.
    # @see cosmology_tools
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
    def test_live_report_update_uses_preserved_input_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory/"results.json"
            source.write_text(json.dumps(self.report))
            original = source.read_bytes()
            def update_report(data):
                source.write_bytes(original+b"\n")
                return plot.plt.figure(figsize=(1, 1))
            with (patch.object(plot, "spatial_figure", side_effect=update_report),
                  patch.object(plot, "gaussian_figure", side_effect=lambda data: plot.plt.figure(figsize=(1, 1)))):
                manifest = plot.render(source, directory/"figures")
            self.assertFalse(manifest["source_report_unchanged_during_plot"])
            self.assertEqual(manifest["input_report"]["sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(gzip.decompress(Path(manifest["input_report"]["snapshot"]).read_bytes()), original)


## @cond CLI_DISPATCH
if __name__ == "__main__":
    unittest.main()
## @endcond
