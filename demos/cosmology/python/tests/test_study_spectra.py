## @file test_study_spectra.py
# @brief Synthetic independent density-estimator tests; no simulations or MPI.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
# @see cosmology_model cosmology_numerics cosmology_spectra
# Tests assert declared invariants using isolated fixtures; no scientific tolerances are relaxed by documentation.
"""Synthetic independent density-estimator tests; no simulations or MPI."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import study_spectra as spectra


## @brief Evaluate the lattice helper in the documented module workflow.
# @see cosmology_tools
#
# @param n Mesh/lattice size per Cartesian dimension in this routine's integer-grid convention.
# @param box Positive periodic comoving box side in Mpc/h.
# @param shift Prescribed translation in the test/measurement coordinate convention; it is not fitted.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def lattice(n, box=1., shift=.5):
    axis = (np.arange(n) + shift) * box / n
    return np.column_stack([part.ravel() for part in np.meshgrid(axis, axis, axis, indexing="ij")])


## @brief Evaluate the scalar direct helper in the documented module workflow.
# @see cosmology_tools
#
# @param positions Finite particle position array of shape (Nparticles,3), in comoving Mpc/h.
# @param box Positive periodic comoving box side in Mpc/h.
# @param modes Signed integer Fourier-mode array with three Cartesian components per mode.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def scalar_direct(positions, box, modes):
    # Deliberately avoid the production helper's matrix products and FFTs.
    return np.asarray([sum(np.exp(-2j * np.pi * sum(int(m[d])*float(x[d]) / box
                       for d in range(3))) for x in positions) / len(positions) for m in modes])


## @brief Regression suite for Spectrum.
# @see cosmology_tools
class SpectrumTests(unittest.TestCase):
    ## @brief Verify modes are spherical unique pairs and ordered consistently.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_modes_are_spherical_unique_pairs_and_ordered_consistently(self):
        modes = spectra.unique_modes(12)
        low = modes[np.sum(modes*modes, axis=1) <= 16]
        np.testing.assert_array_equal(low, spectra.unique_modes(4))
        self.assertEqual(len(low), 128)
        self.assertTrue(np.all(np.linalg.norm(modes, axis=1) <= 12))
        tuples = {tuple(m) for m in modes}
        self.assertEqual(len(tuples), len(modes))
        for m in tuples:
            self.assertNotIn(tuple(-v for v in m), tuples)
        self.assertNotIn((0, 0, 0), tuples)

    ## @brief Verify cic weights axes and periodic mass.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_cic_weights_axes_and_periodic_mass(self):
        grid, box = 8, 8.
        positions = np.array([[1.25, 2.5, 3.75], [-6.75, 10.5, 11.75]])
        counts = spectra.deposit_cic(positions, box, grid)
        self.assertAlmostEqual(counts.sum(), 2.)
        self.assertAlmostEqual(counts[1, 2, 3], 2 * .75 * .5 * .25)
        self.assertAlmostEqual(counts[2, 3, 4], 2 * .25 * .5 * .75)
        self.assertEqual(np.count_nonzero(counts), 8)
        np.testing.assert_array_equal(positions[1], [-6.75, 10.5, 11.75])

    ## @brief Verify shifted node has one nonzero deposit.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_shifted_node_has_one_nonzero_deposit(self):
        shift = np.array([.5, .5, .5])
        position = np.array([[2.5, 3.5, 4.5]])
        counts = spectra.deposit_cic(position, 8., 8, shift)
        self.assertEqual(counts[2, 3, 4], 1.)
        self.assertEqual(np.count_nonzero(counts), 1)

    ## @brief Verify half grid interlacing origin phase analytically.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_half_grid_interlacing_origin_phase_analytically(self):
        n, box = 32, 7.
        position = np.array([[3., 5., 7.]]) * box / n
        modes = np.array([[1, 0, 0], [0, 2, 0], [0, 0, -3], [2, -3, 1]])
        phase = np.exp(-2j*np.pi * (modes @ position[0]) / box)
        raw_expected = .5 * (1 + np.prod(np.cos(np.pi * modes / n), axis=1)) * phase
        raw = spectra.cic_coefficients(position, box, modes, n, deconvolve=False)
        np.testing.assert_allclose(raw, raw_expected, rtol=0, atol=2e-15)
        corrected = spectra.cic_coefficients(position, box, modes, n)
        np.testing.assert_allclose(corrected, raw_expected / spectra.cic_window(modes, n), atol=2e-15)

    ## @brief Verify unshifted grid and known window not cell count normalized.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_unshifted_grid_and_known_window_not_cell_count_normalized(self):
        position = np.array([[0., 0., 0.]] * 3)
        modes = np.array([[1, 0, 0], [2, 1, -3]])
        raw = spectra.cic_coefficients(position, 1., modes, 16, interlaced=False, deconvolve=False)
        np.testing.assert_allclose(raw, 1., atol=1e-15)
        corrected = spectra.cic_coefficients(position, 1., modes, 16, interlaced=False)
        np.testing.assert_allclose(corrected, 1 / spectra.cic_window(modes, 16), atol=1e-15)

    ## @brief Verify uniform lattice and zero signal floor.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_uniform_lattice_and_zero_signal_floor(self):
        positions = lattice(16)
        analysis = spectra.analyze_spectrum(positions, 1., grid=32, max_mode=4)
        np.testing.assert_allclose(analysis["coefficients"], 0., atol=2e-16)
        np.testing.assert_allclose(analysis["direct_low_coefficients"], 0., atol=2e-14)
        self.assertTrue(analysis["diagnostics"]["direct_gate_passed"])
        self.assertIsNone(analysis["diagnostics"]["direct_complex_relative"])
        self.assertTrue(analysis["diagnostics"]["mass_gate_passed"])

    ## @brief Verify direct complex sign axis order and wrapping.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_direct_complex_sign_axis_order_and_wrapping(self):
        positions = np.array([[.125, .2, .4], [.25, .3, .1], [.75, .9, .2]])
        modes = np.array([[1, 0, 0], [0, -2, 0], [1, 2, -1], [3, 0, 2]])
        expected = scalar_direct(positions, 1., modes)
        np.testing.assert_allclose(spectra.direct_coefficients(positions, 1., modes), expected, atol=2e-15)
        shifted_images = positions + np.array([[2, -1, 3], [0, 4, -2], [-4, 5, 1]])
        np.testing.assert_allclose(spectra.direct_coefficients(shifted_images, 1., modes), expected, atol=1e-14)

    ## @brief Verify hermiticity with nontrivial origins.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_hermiticity_with_nontrivial_origins(self):
        positions = np.array([[.13, .27, .31], [.87, .64, .79]])
        modes = np.array([[1, -2, 3], [-1, 2, -3]])
        for result in (spectra.direct_coefficients(positions, 1., modes),
                       spectra.cic_coefficients(positions, 1., modes, 32)):
            self.assertAlmostEqual(abs(result[0] - np.conj(result[1])), 0., places=14)

    ## @brief Verify rigid translation and positive dephase.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_rigid_translation_and_positive_dephase(self):
        positions = np.array([[.11, .23, .37], [.41, .61, .83], [.79, .89, .93]])
        modes = spectra.unique_modes(4)
        shift = np.array([.37, .23, .41]) / 64
        before = spectra.direct_coefficients(positions, 1., modes)
        after = spectra.direct_coefficients((positions + shift) % 1., 1., modes)
        corrected = spectra.dephase(after, modes, 1., shift)
        np.testing.assert_allclose(corrected, before, atol=3e-15)
        self.assertGreater(np.linalg.norm(after - before), .1)
        np.testing.assert_allclose(abs(after)**2, abs(before)**2, atol=3e-15)

    ## @brief Verify integer analysis cell translation preserves cic alias error.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_integer_analysis_cell_translation_preserves_cic_alias_error(self):
        positions = np.random.default_rng(911).uniform(size=(129, 3))
        modes = spectra.unique_modes(4)
        shift = np.array([3, -5, 7]) / 32
        before = spectra.cic_coefficients(positions, 1., modes, 32)
        after = spectra.cic_coefficients(positions + shift, 1., modes, 32)
        np.testing.assert_allclose(spectra.dephase(after, modes, 1., shift), before, atol=3e-16)

    ## @brief Verify cic alias bound is attained by node particle.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_cic_alias_bound_is_attained_by_node_particle(self):
        modes = spectra.unique_modes(4)
        for interlaced in (True, False):
            measured = spectra.cic_coefficients([[0., 0., 0.]], 1., modes, 32, interlaced)
            bound = spectra.cic_alias_bound(modes, 32, interlaced)
            np.testing.assert_allclose(measured.real - 1, bound, atol=8e-16)
            np.testing.assert_allclose(measured.imag, 0., atol=1e-16)
        self.assertTrue(np.all(spectra.cic_alias_bound(modes, 32)
                               < spectra.cic_alias_bound(modes, 32, False)))

    ## @brief Verify alias bound encloses arbitrary particle error.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_alias_bound_encloses_arbitrary_particle_error(self):
        positions = np.random.default_rng(2003).uniform(-2., 3., size=(311, 3))
        modes = spectra.unique_modes(12)[::41]
        direct = scalar_direct(positions, 1., modes)
        for interlaced in (True, False):
            measured = spectra.cic_coefficients(positions, 1., modes, 32, interlaced)
            bound = spectra.cic_alias_bound(modes, 32, interlaced)
            self.assertTrue(np.all(np.abs(measured - direct) <= bound + 2e-14))

    ## @brief Verify dense smooth jitter direct validation and high band label.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_dense_smooth_jitter_direct_validation_and_high_band_label(self):
        positions = lattice(24)
        positions += (.08 / (2*np.pi)) * np.sin(2*np.pi*positions)
        positions += np.random.default_rng(4407).uniform(-.08/24, .08/24, positions.shape)
        measured = spectra.analyze_spectrum(positions, 1.)
        self.assertTrue(measured["diagnostics"]["direct_gate_passed"])
        self.assertTrue(measured["diagnostics"]["mass_gate_passed"])
        self.assertIn("characterization", measured["diagnostics"]["higher_band_status"])
        selected = np.array([[1, 0, 0], [0, 2, 0], [1, -1, 2], [0, 0, 3]])
        indices = [np.flatnonzero(np.all(measured["direct_low_modes"] == m, axis=1))[0] for m in selected]
        np.testing.assert_allclose(measured["direct_low_coefficients"][indices],
                                   scalar_direct(positions, 1., selected), atol=3e-15)
        self.assertEqual(len(measured["shells"]), 7)
        self.assertEqual(len(measured["direct_low_shells"]), 3)

    ## @brief Verify analysis dephases both returned coefficient sets.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_analysis_dephases_both_returned_coefficient_sets(self):
        positions = np.random.default_rng(511).uniform(size=(31, 3))
        shift = [.37/64, .23/64, .41/64]
        raw = spectra.analyze_spectrum(positions, 1., grid=32, max_mode=4)
        aligned = spectra.analyze_spectrum(positions, 1., grid=32, max_mode=4, translation=shift)
        for key, mode_key in (("coefficients", "modes"), ("direct_low_coefficients", "direct_low_modes")):
            np.testing.assert_allclose(aligned[key], spectra.dephase(raw[key], raw[mode_key], 1., shift))
        self.assertEqual(raw["diagnostics"]["direct_absolute_error"], aligned["diagnostics"]["direct_absolute_error"])

    ## @brief Verify measurement gate detects inadequate analysis grid.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_measurement_gate_detects_inadequate_analysis_grid(self):
        # Preserve the legacy CIC failure rather than change its threshold.
        modes = spectra.unique_modes(4)
        direct = spectra.direct_coefficients([[0., 0., 0.]], 1., modes)
        legacy = spectra.cic_coefficients([[0., 0., 0.]], 1., modes, 32)
        self.assertGreater(np.linalg.norm(legacy-direct) / np.linalg.norm(direct), 1e-3)
        self.assertEqual(spectra.DirectRelativeLimit, 1e-3)

    ## @brief Verify measurement gate detects amplitude corruption.
    # @see cosmology_tools
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
    def test_measurement_gate_detects_amplitude_corruption(self):
        original = spectra._pcs_extract
        def corrupt(*args):
            result, masses = original(*args)
            return 1.01 * result, masses
        with patch.object(spectra, "_pcs_extract", side_effect=corrupt):
            measured = spectra.analyze_spectrum([[0., 0., 0.]], 1., max_mode=4)
        self.assertFalse(measured["diagnostics"]["direct_gate_passed"])
        self.assertTrue(measured["diagnostics"]["mass_gate_passed"])

    ## @brief Verify pcs four node weights axes and periodic mass.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_pcs_four_node_weights_axes_and_periodic_mass(self):
        positions = np.array([[1.25, 2.5, 3.75], [-6.75, 10.5, 11.75]])
        counts = spectra.deposit_pcs(positions, 8., 8)
        f = np.array([.25, .5, .75])
        self.assertAlmostEqual(counts.sum(), 2.)
        self.assertAlmostEqual(counts[0, 1, 2], 2*np.prod((1-f)**3 / 6))
        self.assertAlmostEqual(counts[3, 4, 5], 2*np.prod(f**3 / 6))
        self.assertAlmostEqual(counts[1, 2, 3], 2*np.prod((4-6*f*f+3*f**3) / 6))
        self.assertEqual(np.count_nonzero(counts), 64)
        self.assertTrue(np.all(counts >= 0))

    ## @brief Verify pcs half grid origin phase and single mode response.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_pcs_half_grid_origin_phase_and_single_mode_response(self):
        n, box = 32, 7.
        position = np.array([[3., 5., 7.]]) * box / n
        modes = np.array([[1, 0, 0], [0, 2, 0], [0, 0, -3], [2, -3, 1]])
        t = np.pi * modes / n
        phase = np.exp(-2j*np.pi * (modes @ position[0]) / box)
        ordinary = np.prod((2 + np.cos(2*t)) / 3, axis=1)
        shifted = np.prod(np.cos(t) * (5+np.cos(t)**2) / 6, axis=1)
        expected = .5 * (ordinary + shifted) * phase
        raw = spectra.pcs_coefficients(position, box, modes, n, deconvolve=False)
        np.testing.assert_allclose(raw, expected, rtol=0, atol=2e-15)
        corrected = spectra.pcs_coefficients(position, box, modes, n)
        np.testing.assert_allclose(corrected, expected / spectra.pcs_window(modes, n), atol=2e-15)
        ordinary_raw = spectra.pcs_coefficients(position, box, modes, n,
                                                interlaced=False, deconvolve=False)
        np.testing.assert_allclose(ordinary_raw, ordinary*phase, atol=2e-15)

    ## @brief Verify pcs alias formula against explicit image sums.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_pcs_alias_formula_against_explicit_image_sums(self):
        # Independent numerical sum of the fourth-power image weights, not
        # a second use of the closed trigonometric formula.
        grid = 32
        modes = np.array([[1, 0, 0], [4, -3, 2], [12, 2, -5]])
        images = np.arange(-2048, 2049)
        ordinary, alternating = [], []
        for mode in modes:
            a, b = [], []
            for component in mode:
                if component == 0:
                    a.append(1.)
                    b.append(1.)
                else:
                    t = component / grid
                    weights = (t / (t + images))**4
                    a.append(weights.sum())
                    b.append((weights * np.where(images % 2, -1., 1.)).sum())
            ordinary.append(np.prod(a) - 1)
            alternating.append(.5 * (np.prod(a) + np.prod(b)) - 1)
        np.testing.assert_allclose(spectra.pcs_alias_bound(modes, grid, False), ordinary,
                                   rtol=0, atol=3e-12)
        np.testing.assert_allclose(spectra.pcs_alias_bound(modes, grid), alternating,
                                   rtol=0, atol=2e-12)

    ## @brief Verify pcs alias bound is attained and encloses particle errors.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_pcs_alias_bound_is_attained_and_encloses_particle_errors(self):
        modes = spectra.unique_modes(4)
        for interlaced in (True, False):
            measured = spectra.pcs_coefficients([[0., 0., 0.]], 1., modes, 32, interlaced)
            bound = spectra.pcs_alias_bound(modes, 32, interlaced)
            np.testing.assert_allclose(measured.real - 1, bound, atol=1e-15)
            # Symmetry cancellation includes deposition, FFT and physical
            # origin phases; allow a small multiple of double machine epsilon.
            np.testing.assert_allclose(measured.imag, 0., atol=8*np.finfo(float).eps)
            positions = np.random.default_rng(2003).uniform(-2., 3., size=(311, 3))
            direct = scalar_direct(positions, 1., modes[::9])
            measured = spectra.pcs_coefficients(positions, 1., modes[::9], 32, interlaced)
            self.assertTrue(np.all(np.abs(measured-direct) <= bound[::9] + 2e-14))
        self.assertTrue(np.all(spectra.pcs_alias_bound(modes, 32) < spectra.cic_alias_bound(modes, 32)))

    ## @brief Verify pcs preserves phase under grid cell translation.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_pcs_preserves_phase_under_grid_cell_translation(self):
        positions = np.random.default_rng(911).uniform(size=(129, 3))
        modes = spectra.unique_modes(4)
        shift = np.array([3, -5, 7]) / 32
        before = spectra.pcs_coefficients(positions, 1., modes, 32)
        after = spectra.pcs_coefficients(positions + shift, 1., modes, 32)
        np.testing.assert_allclose(spectra.dephase(after, modes, 1., shift), before, atol=3e-16)

    ## @brief Verify pcs actual gaussian initial measurement without relaxed gate.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_pcs_actual_gaussian_initial_measurement_without_relaxed_gate(self):
        from gaussian_fixture import make_gaussian_fixture
        for n, cutoff, redshift in ((16, 6, 99), (32, 12, 49), (32, 12, 99)):
            with self.subTest(particles=n, redshift=redshift):
                frame, _ = make_gaussian_fixture(n, redshift, 20261003, cutoff=cutoff)
                positions = frame[["x", "y", "z"]].to_numpy()
                analysis = spectra.analyze_spectrum(positions, 168.75, max_mode=cutoff)
                diagnostic = analysis["diagnostics"]
                self.assertEqual(diagnostic["assignment"], "PCS")
                self.assertEqual(diagnostic["direct_relative_limit"], 1e-3)
                self.assertEqual(diagnostic["direct_absolute_floor"], 1e-12)
                self.assertTrue(diagnostic["direct_gate_passed"])
                self.assertTrue(diagnostic["mass_gate_passed"])
                self.assertLessEqual(diagnostic["direct_complex_relative"], 1e-3)
                if n == 16:
                    # This is the preserved pipeline-smoke failure: coherent
                    # particle-lattice aliases defeated CIC at the same grid.
                    legacy = spectra.cic_coefficients(positions, 168.75,
                                                       analysis["direct_low_modes"], 128)
                    direct = analysis["direct_low_coefficients"]
                    self.assertGreater(np.linalg.norm(legacy-direct)/np.linalg.norm(direct), 1e-3)

    ## @brief Verify shell power volume normalization and pair counts.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_shell_power_volume_normalization_and_pair_counts(self):
        modes = np.array([[1, 0, 0], [0, 1, 0], [2, 0, 0]])
        coefficients = np.array([.1 + .2j, .3 + .4j, .2j])
        rows = spectra.shell_statistics(coefficients, modes, 10.)
        self.assertEqual([r["pairs"] for r in rows], [2, 1])
        self.assertAlmostEqual(rows[0]["delta_power_sum"], .3)
        self.assertAlmostEqual(rows[0]["power_mean"], 150.)
        self.assertAlmostEqual(rows[1]["power_mean"], 40.)
        self.assertAlmostEqual(rows[0]["k_mean"], 2*np.pi/10.)
        for invalid_modes in ([[1, 0, 0], [1, 0, 0]], [[1, 0, 0], [-1, 0, 0]]):
            with self.assertRaises(ValueError):
                spectra.shell_statistics([1., 1.], invalid_modes, 1.)

    ## @brief Verify nonfinite empty and wrong geometry rejected.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_nonfinite_empty_and_wrong_geometry_rejected(self):
        modes = [[1, 0, 0]]
        for positions in ([], [[np.nan, 0, 0]], [[1, 2]], [[np.inf, 0, 0]]):
            with self.assertRaises(ValueError):
                spectra.cic_coefficients(positions, 1., modes, 32)
        for grid in (3, 7, 32., True):
            with self.assertRaises(ValueError):
                spectra.cic_coefficients([[0, 0, 0]], 1., modes, grid)
        for box in (0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                spectra.direct_coefficients([[0, 0, 0]], box, modes)
        for invalid_modes in ([], [[0, 0, 0]], [[.5, 0, 0]], [[np.nan, 0, 0]], [[16, 0, 0]]):
            with self.assertRaises(ValueError):
                spectra.cic_coefficients([[0, 0, 0]], 1., invalid_modes, 32)
        with self.assertRaises(ValueError):
            spectra.analyze_spectrum([[0, 0, 0]], 1., max_mode=3, direct_max_mode=4)


## @cond CLI_DISPATCH
if __name__ == "__main__":
    unittest.main()
## @endcond
