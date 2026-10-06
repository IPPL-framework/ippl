#!/usr/bin/env python3
## @file test_plot_pm_validation.py
# @brief Focused numerical extraction tests for saved-evidence PM figures.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
# Tests assert declared invariants using isolated fixtures; no scientific tolerances are relaxed by documentation.
"""Focused numerical extraction tests for saved-evidence PM figures.

Uses small synthetic campaign dictionaries, not simulation output or MPI.
No files are rendered; the figure-range test inspects and closes its figure.
"""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plot_pm_validation as plots


## @brief Evaluate the frozen campaign helper in the documented module workflow.
# @see cosmology_tools
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def frozen_campaign():
    campaign = {"native_operator_differences": [], "checks": []}
    for fixture in ("uniform", *plots.FixtureLabels):
        for rank in (1, 2, 3, 4):
            name = f"{fixture}/cross_code/{rank}r"
            isUniform = fixture == "uniform"
            campaign["native_operator_differences"].append({
                "name": name, "ippl_force_rms": 0.0 if isUniform else 2.0,
                "raw_difference_rms": 0.0 if isUniform else 4.0,
                "predicted_difference_rms": 0.0 if isUniform else 3.0,
            })
            # The norm of a difference need not equal the difference of norms.
            # Norms 4 and 3 can have a difference norm 5 (orthogonal fields).
            campaign["checks"].append({
                "name": name + "/predicted_native_difference",
                "error_rms": 0.0 if isUniform else 5.0,
            })
            for code in ("ippl", "fastpm"):
                campaign["checks"].append({
                    "name": f"{fixture}/{code}/{rank}r/particle_force",
                    "relative_rms": None if isUniform else
                        (rank * 1e-15 if code == "ippl" else rank * 1e-8),
                })
    return campaign


## @brief Evaluate the pancake campaign helper in the documented module workflow.
# @see cosmology_tools
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def pancake_campaign():
    campaign = {"trajectory_metrics": {}, "checks": []}
    for amplitude in (.5, .8):
        for n in (16, 32, 64):
            campaign["trajectory_metrics"][f"axis_a{int(amplitude * 10)}_n{n}"] = {
                "final": {"displacement_relative_error": amplitude / n,
                          "momentum_relative_error": 2 * amplitude / n,
                          "reference_displacement_rms": np.sqrt(3) * 10,
                          "reference_momentum_rms": np.sqrt(3) * 2},
            }
    for quantity, values in (("position", (8.0, 2.0, .5)),
                             ("momentum", (4.0, 1.0, .25))):
        for suffix, coarse, fine in (("nt16/32/64", values[0], values[1]),
                                    ("nt32/64/128", values[1], values[2])):
            campaign["checks"].append({
                "name": f"time {quantity}: {suffix}",
                "coarse_per_component_rms_difference": coarse,
                "fine_per_component_rms_difference": fine,
            })
    return campaign


## @brief Regression suite for PlotExtraction.
# @see cosmology_tools
class PlotExtractionTests(unittest.TestCase):
    ## @brief Verify native residual is field difference not difference of norms.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_native_residual_is_field_difference_not_difference_of_norms(self):
        data, _ = plots.force_data(frozen_campaign())
        np.testing.assert_allclose(data["raw"], 2.0)
        np.testing.assert_allclose(data["prediction"], 1.5)
        np.testing.assert_allclose(data["residual"], 2.5)
        self.assertTrue((data["residual"] != abs(data["raw"] - data["prediction"])).all())

    ## @brief Verify uniform zero relative values are omitted not floored.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_uniform_zero_relative_values_are_omitted_not_floored(self):
        data, zeros = plots.force_data(frozen_campaign())
        self.assertEqual(len(data), 4 * len(plots.FixtureLabels))
        self.assertNotIn("uniform", set(data["fixture"]))
        self.assertEqual(zeros, [f"uniform/cross_code/{rank}r" for rank in (1, 2, 3, 4)])
        self.assertTrue(np.isfinite(data[["raw", "prediction", "residual", "ippl", "fastpm"]]).all().all())

    ## @brief Verify zero force with nonzero difference is rejected.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_zero_force_with_nonzero_difference_is_rejected(self):
        for key in ("raw_difference_rms", "predicted_difference_rms"):
            with self.subTest(key=key):
                campaign = frozen_campaign()
                campaign["native_operator_differences"][0][key] = 1e-15
                with self.assertRaisesRegex(ValueError, "Zero IPPL force"):
                    plots.force_data(campaign)

    ## @brief Verify missing or duplicate nonzero rank is rejected.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_missing_or_duplicate_nonzero_rank_is_rejected(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate):
                campaign = frozen_campaign()
                rows = campaign["native_operator_differences"]
                index = next(i for i, row in enumerate(rows)
                             if row["name"] == "oblique/cross_code/3r")
                if duplicate:
                    rows.append(deepcopy(rows[index]))
                else:
                    rows.pop(index)
                with self.assertRaisesRegex(ValueError, "Incomplete rank coverage"):
                    plots.force_data(campaign)

    ## @brief Verify missing uniform rank is rejected.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_missing_uniform_rank_is_rejected(self):
        campaign = frozen_campaign()
        campaign["native_operator_differences"].pop(0)
        with self.assertRaisesRegex(ValueError, "four exactly-zero"):
            plots.force_data(campaign)

    ## @brief Verify duplicate saved check is rejected.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_duplicate_saved_check_is_rejected(self):
        campaign = frozen_campaign()
        check = next(row for row in campaign["checks"]
                     if row["name"] == "oblique/cross_code/1r/predicted_native_difference")
        campaign["checks"].append(deepcopy(check))
        with self.assertRaisesRegex(ValueError, "Missing or duplicate check"):
            plots.force_data(campaign)

    ## @brief Verify logarithmic values are never silently floored.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_logarithmic_values_are_never_silently_floored(self):
        for invalid in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "do not invent a floor"):
                    plots.positive(np.asarray([1e-30, invalid]), "test")
        plots.positive(np.asarray([1e-30, 1.0]), "test")

    ## @brief Verify timestep difference uses per component reference.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_timestep_difference_uses_per_component_reference(self):
        spatial, temporal = plots.convergence_data(pancake_campaign())
        self.assertEqual(len(spatial), 12)
        self.assertEqual(len(temporal), 6)
        for quantity, expected in (("position", [.8, .2, .05]),
                                   ("momentum", [2.0, .5, .125])):
            group = temporal[temporal.quantity == quantity]
            np.testing.assert_array_equal(group.steps, [16, 32, 64])
            np.testing.assert_allclose(group.relative_difference, expected, rtol=2e-15)
        # Spatial errors are already relative vector RMS; do not rescale them.
        row = spatial[(spatial.amplitude == .8) & (spatial.n == 64)
                      & (spatial.quantity == "momentum")].iloc[0]
        self.assertEqual(row.relative_error, 1.6 / 64)

    ## @brief Verify inconsistent overlapping timestep difference is rejected.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_inconsistent_overlapping_timestep_difference_is_rejected(self):
        campaign = pancake_campaign()
        campaign["checks"][1]["coarse_per_component_rms_difference"] += .001
        with self.assertRaisesRegex(ValueError, "overlapping timestep differences disagree"):
            plots.convergence_data(campaign)

    ## @brief Verify force axis limits keep every positive marker visible.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_force_axis_limits_keep_every_positive_marker_visible(self):
        campaign = frozen_campaign()
        for row in campaign["checks"]:
            if row["name"].startswith("axis/ippl/"):
                row["relative_rms"] = 6.725282811058402e-16
        data, _ = plots.force_data(campaign)
        data["raw"] = .01
        data["prediction"] = .01
        data["residual"] = 1e-8
        captured = {}

        def capture(figure, directory, name):
            captured["limits"] = [axis.get_xlim() for axis in figure.axes]
            plots.plt.close(figure)

        with patch.object(plots, "save_figure", side_effect=capture):
            aggregates = plots.force_figure(data, Path("unused-no-output"))
        for panel, keys in ((0, ("ippl", "fastpm")),
                            (1, ("raw", "prediction", "residual"))):
            lower, upper = captured["limits"][panel]
            for row in aggregates:
                for key in keys:
                    self.assertLessEqual(lower, row[key], (row["fixture"], key))
                    self.assertGreaterEqual(upper, row[key], (row["fixture"], key))


## @cond CLI_DISPATCH
if __name__ == "__main__":
    unittest.main()
## @endcond
