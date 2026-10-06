## @file test_analyze_zeldovich_benchmark.py
# @brief Tests for periodic CIC power normalization and compression integrity helpers.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
# @see cosmology_model cosmology_numerics cosmology_spectra
# Tests assert declared invariants using isolated fixtures; no scientific tolerances are relaxed by documentation.
"""Tests for periodic CIC power normalization and compression integrity helpers."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_zeldovich_benchmark as analysis


## @brief Regression suite for CICPowers.
# @see cosmology_tools
class CICPowers(unittest.TestCase):
    ## @brief Verify uniform cell centered lattice has no resolved density modes.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_uniform_cell_centered_lattice_has_no_resolved_density_modes(self):
        n = 8
        ids = np.arange(n**3)
        positions = (np.column_stack((ids % n, ids // n % n, ids // n**2)) + .5)
        spectrum = analysis.cic_power(positions, particle_grid=n, mesh_grid=n,
                                      box_size=float(n), cutoff=3)
        self.assertEqual([row["shell"] for row in spectrum], [1, 2, 3])
        self.assertTrue(all(abs(row["P_cic_deconvolved_raw"]) < 1e-27 for row in spectrum))
        self.assertTrue(all(row["P_shot_subtracted"] < 0 for row in spectrum))

    ## @brief Verify periodic wrapping and integrity of power rows.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_periodic_wrapping_and_integrity_of_power_rows(self):
        n, box = 8, 10.0
        ids = np.arange(n**3)
        positions = (np.column_stack((ids % n, ids // n % n, ids // n**2)) + .5) * box / n
        shifted = positions.copy()
        shifted[::3] += box
        base = analysis.cic_power(positions, particle_grid=n, mesh_grid=n,
                                  box_size=box, cutoff=3)
        wrapped = analysis.cic_power(shifted, particle_grid=n, mesh_grid=n,
                                     box_size=box, cutoff=3)
        np.testing.assert_allclose([row["P_cic_deconvolved_raw"] for row in wrapped],
                                   [row["P_cic_deconvolved_raw"] for row in base],
                                   rtol=0, atol=1e-27)
        self.assertTrue(all(row["full_lattice_mode_count"] > 0 for row in wrapped))


## @cond CLI_DISPATCH
if __name__ == "__main__":
    unittest.main()
## @endcond
