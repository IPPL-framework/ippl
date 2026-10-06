## @file test_compare_fixture_files.py
# @brief Small synthetic fixture comparison tests; no MPI or physical tolerances.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
# Tests assert declared invariants using isolated fixtures; no scientific tolerances are relaxed by documentation.
"""Small synthetic fixture comparison tests; no MPI or physical tolerances."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare_fixture_files as compare


## @brief Regression suite for FixtureComparison.
# @see cosmology_tools
class FixtureComparisonTests(unittest.TestCase):
    ## @brief Create isolated fixtures and temporary artifact paths for regression checks.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def setUp(self):
        ## @var temporary
        # @brief Retained temporary state owned by this instance; see the initialization and workflow contract.
        self.temporary = tempfile.TemporaryDirectory()
        ## @var root
        # @brief Campaign or artifact root following this module's ownership contract.
        self.root = Path(self.temporary.name)
        ## @var left
        # @brief First ID-matched state or Fourier array; the metric specifies denominator and normalization.
        ## @var right
        # @brief Second ID-matched state or Fourier array; the metric specifies denominator and normalization.
        self.left, self.right = self.root / "local.csv", self.root / "remote.csv"
        ## @var frame
        # @brief Particle DataFrame with the exact columns/ID/finite-state contract checked by this routine.
        self.frame = pd.DataFrame({"id": np.arange(8, dtype=np.uint64), "x": np.arange(8, dtype=float),
            "y": 2., "z": 3., "px": 1., "py": 0., "pz": 0., "mass": 1.})
        self.write(self.frame, self.left)
        self.write(self.frame, self.right)

    ## @brief Release only the temporary fixtures owned by this test.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def tearDown(self):
        self.temporary.cleanup()

    ## @brief Write the documented module workflow.
    # @see cosmology_tools
    #
    # @param frame Particle DataFrame with the exact columns/ID/finite-state contract checked by this routine.
    # @param path Input/output filesystem path; freshness and hash requirements are defined by this routine.
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    @staticmethod
    def write(frame, path):
        frame.to_csv(path, index=False, float_format="%.17g")

    ## @brief Verify identical bytes and values report no physical pass flag.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_identical_bytes_and_values_report_no_physical_pass_flag(self):
        result = compare.compare_fixtures(self.left, self.right, 10.)
        self.assertTrue(result["agreement"]["csv_bytes_identical"])
        self.assertTrue(result["agreement"]["sorted_numeric_values_equal"])
        self.assertTrue(result["agreement"]["sorted_numeric_bits_identical"])
        self.assertEqual(result["original"]["csv_sha256"], compare.sha256(self.left))
        self.assertEqual(result["original"]["canonical_value_sha256"], result["comparison"]["canonical_value_sha256"])
        self.assertEqual(result["position_difference"]["vector_rms"], 0.)
        self.assertEqual(result["momentum_difference"]["relative_vector_rms"], 0.)
        self.assertNotIn("passed", result)
        self.assertNotIn("tolerance", result)
        self.assertIn("not assessed", result["physical_qualification"])

    ## @brief Verify row order and decimal format do not affect sorted values.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_row_order_and_decimal_format_do_not_affect_sorted_values(self):
        self.frame.iloc[::-1].to_csv(self.right, index=False, float_format="%.17e")
        result = compare.compare_fixtures(self.left, self.right, 10.)
        self.assertFalse(result["agreement"]["csv_bytes_identical"])
        self.assertTrue(result["agreement"]["sorted_numeric_bits_identical"])
        self.assertEqual(result["original"]["canonical_value_sha256"], result["comparison"]["canonical_value_sha256"])

    ## @brief Verify signed zero distinguishes bits not numeric values.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_signed_zero_distinguishes_bits_not_numeric_values(self):
        changed = self.frame.copy()
        changed.loc[0, "py"] = -0.
        self.write(changed, self.right)
        result = compare.compare_fixtures(self.left, self.right, 10.)
        self.assertTrue(result["agreement"]["sorted_numeric_values_equal"])
        self.assertFalse(result["agreement"]["sorted_numeric_bits_identical"])
        self.assertEqual(result["agreement"]["different_bitwise_components"], 1)
        self.assertEqual(result["agreement"]["different_numeric_components"], 0)
        self.assertEqual(result["momentum_difference"]["vector_rms"], 0.)

    ## @brief Verify periodic position rms and explicit cell normalization.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_periodic_position_rms_and_explicit_cell_normalization(self):
        changed = self.frame.copy()
        changed.x += 21.  # Two boxes plus a genuine displacement of +1.
        changed.y -= 20.
        self.write(changed, self.right)
        result = compare.compare_fixtures(self.left, self.right, 10., cell_grid=5)
        self.assertEqual(result["position_difference"]["vector_rms"], 1.)
        self.assertEqual(result["position_difference"]["maximum_particle_norm"], 1.)
        self.assertEqual(result["position_difference"]["vector_rms_cells"], .5)
        self.assertEqual(result["position_difference"]["nonzero_component_count"], 8)
        self.assertEqual(result["geometry"]["cell_normalization"], "explicit --cell-grid")
        default = compare.compare_fixtures(self.left, self.right, 10.)
        self.assertEqual(default["geometry"]["cell_grid"], 2)
        self.assertEqual(default["position_difference"]["vector_rms_cells"], .2)

    ## @brief Verify momentum is unwrapped and original is denominator.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_momentum_is_unwrapped_and_original_is_denominator(self):
        changed = self.frame.copy()
        changed.px += .25
        changed.py += .5
        self.write(changed, self.right)
        result = compare.compare_fixtures(self.left, self.right, 10.)
        expected = np.sqrt(.25**2 + .5**2)
        self.assertAlmostEqual(result["momentum_difference"]["vector_rms"], expected)
        self.assertEqual(result["momentum_difference"]["original_vector_rms"], 1.)
        self.assertAlmostEqual(result["momentum_difference"]["relative_vector_rms"], expected)
        changed.px += 10.
        self.write(changed, self.right)
        result = compare.compare_fixtures(self.left, self.right, 10.)
        self.assertAlmostEqual(result["momentum_difference"]["vector_rms"], np.sqrt(10.25**2 + .5**2))

    ## @brief Verify zero momentum reference explicitly undefined.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_zero_momentum_reference_explicitly_undefined(self):
        frame = self.frame.copy()
        frame[["px", "py", "pz"]] = 0.
        self.write(frame, self.left)
        for difference in (0., 1.):
            frame.px = difference
            self.write(frame, self.right)
            result = compare.compare_fixtures(self.left, self.right, 10.)
            self.assertIsNone(result["momentum_difference"]["relative_vector_rms"])
            self.assertEqual(result["momentum_difference"]["normalization_status"], "undefined_zero_original_momentum")
            self.assertEqual(result["momentum_difference"]["vector_rms"], difference)

    ## @brief Verify invalid ids masses and finite contract rejected.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_invalid_ids_masses_and_finite_contract_rejected(self):
        for field, value in (("id", -1), ("id", "1.5"), ("id", str(2**64)),
                             ("id", 1), ("mass", 2.), ("px", float("inf")), ("x", float("nan"))):
            with self.subTest(field=field, value=value):
                frame = self.frame.copy()
                if field == "id":
                    frame["id"] = frame.id.astype(object)
                frame.loc[0, field] = value
                self.write(frame, self.right)
                with self.assertRaises(ValueError):
                    compare.compare_fixtures(self.left, self.right, 10.)

    ## @brief Verify count columns and geometry rejected.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_count_columns_and_geometry_rejected(self):
        self.write(self.frame.iloc[:-1], self.right)
        with self.assertRaisesRegex(ValueError, "perfect cube"):
            compare.compare_fixtures(self.left, self.right, 10.)
        self.write(self.frame.iloc[:1], self.right)
        with self.assertRaisesRegex(ValueError, "same particle count"):
            compare.compare_fixtures(self.left, self.right, 10.)
        self.write(self.frame.drop(columns="mass"), self.right)
        with self.assertRaisesRegex(ValueError, "exact columns"):
            compare.compare_fixtures(self.left, self.right, 10.)
        self.write(self.frame, self.right)
        for box in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                compare.compare_fixtures(self.left, self.right, box)
        for grid in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                compare.compare_fixtures(self.left, self.right, 10., grid)

    ## @brief Verify roundtrip retains adjacent float values.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_roundtrip_retains_adjacent_float_values(self):
        changed = self.frame.copy()
        changed.loc[0, "px"] = np.nextafter(1., 2.)
        self.write(changed, self.right)
        result = compare.compare_fixtures(self.left, self.right, 10.)
        self.assertEqual(result["agreement"]["different_numeric_components"], 1)
        self.assertEqual(result["momentum_difference"]["maximum_absolute_component"], np.spacing(1.))
        self.assertAlmostEqual(result["momentum_difference"]["vector_rms"], np.spacing(1.) / np.sqrt(8), delta=1e-31)

    ## @brief Verify report refuses overwrite and leaves inputs untouched.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_report_refuses_overwrite_and_leaves_inputs_untouched(self):
        before = [compare.sha256(path) for path in (self.left, self.right)]
        result = compare.compare_fixtures(self.left, self.right, 10.)
        output = self.root / "comparison.json"
        compare.write_report(result, output)
        original_bytes = output.read_bytes()
        with self.assertRaises(FileExistsError):
            compare.write_report(result, output)
        with self.assertRaises(FileExistsError):
            compare.write_report(result, self.left)
        self.assertEqual(output.read_bytes(), original_bytes)
        self.assertEqual([compare.sha256(path) for path in (self.left, self.right)], before)
        self.assertEqual(json.loads(output.read_text())["schema"], "cosmology-fixture-comparison-v1")

    ## @brief Verify cli difference is a completed comparison not scientific failure.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_cli_difference_is_a_completed_comparison_not_scientific_failure(self):
        changed = self.frame.copy()
        changed.px += 100.
        self.write(changed, self.right)
        output = self.root / "comparison.json"
        arguments = ["compare_fixture_files.py", str(self.left), str(self.right), "--box-size", "10", "--output", str(output)]
        with patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
            self.assertEqual(compare.main(), 0)
            self.assertEqual(compare.main(), 1)  # Existing report cannot be replaced.
        self.assertFalse(json.loads(output.read_text())["agreement"]["sorted_numeric_values_equal"])


## @cond CLI_DISPATCH
if __name__ == "__main__":
    unittest.main()
## @endcond
