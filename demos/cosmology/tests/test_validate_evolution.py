"""Independent synthetic/corruption tests for matched-particle evolution analysis.

No simulation executable or external reference is required.  A folded map is
used solely to test the ordering diagnostic, never as an analytic solution of
the post-shell-crossing gravitational evolution.
"""

from pathlib import Path
import math
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_evolution as validation


def lattice_frame(n=8, box=168.75):
    ids = np.arange(n**3, dtype=np.uint64)
    q = (np.column_stack((ids % n, ids//n % n, ids//(n*n)))+.5)*box/n
    frame = pd.DataFrame(q, columns=["x", "y", "z"])
    frame.insert(0, "id", ids)
    frame[["px", "py", "pz"]] = 0.0
    return frame


def synthetic_factor_table(steps=8, ai=.02, af=.2):
    """Complete log schedule with independently analytic positive EdS factors.

    The schedule guard is background-independent; factor quadrature accuracy
    remains a separate runner check against the requested cosmology.
    """
    endpoint = np.geomspace(ai, af, steps+1)
    a0, a1 = endpoint[:-1], endpoint[1:]
    half = np.sqrt(a0*a1)
    return pd.DataFrame({"step": np.arange(steps), "a0": a0, "ah": half, "a1": a1,
                         "drift": 2*(a0**-.5-a1**-.5),
                         "canonical_kick0": 2*(np.sqrt(half)-np.sqrt(a0)),
                         "canonical_kick1": 2*(np.sqrt(a1)-np.sqrt(half))})


class EvolutionAnalysisTests(unittest.TestCase):
    def test_fixture_momenta_are_one_shared_float32_quantization(self):
        box = validation.Parameters["box_size"]
        ai = validation.Parameters["a_initial"]
        omega = validation.Parameters["omega_m"]
        _, rate = validation.growth_reference(ai, omega)
        canonicalFactor = ai**2*math.sqrt(omega/ai**3+1-omega)*rate
        for kind in ("pancake", "coupled3d"):
            frame, description = validation.make_fixture(kind, particle_grid=8)
            np.testing.assert_array_equal(frame.id, np.arange(8**3))
            self.assertTrue((frame.mass == 1).all())
            q = lattice_frame(8, box)[["x", "y", "z"]].to_numpy()
            displacement = validation.periodic_difference(frame[["x", "y", "z"]].to_numpy(), q, box)
            raw = canonicalFactor*displacement
            actual = frame[["px", "py", "pz"]].to_numpy()
            np.testing.assert_array_equal(actual, actual.astype(np.float32).astype(np.float64))
            self.assertLess(np.linalg.norm(actual-raw)/np.linalg.norm(raw), np.finfo(np.float32).eps)
            self.assertGreater(description["initial_minimum_map_eigenvalue"], 0)
            self.assertLess(description["momentum_quantization_relative_rms"], np.finfo(np.float32).eps)
            self.assertTrue(np.isfinite(frame[["x", "y", "z", "px", "py", "pz"]]).all().all())

    def test_fixture_linear_density_coefficients_and_normalization(self):
        """Recover -i k.s_hat directly on q; do not fit mode amplitude or phase."""
        n, box = 16, validation.Parameters["box_size"]
        q = lattice_frame(n, box)[["x", "y", "z"]].to_numpy()
        di = validation.growth_reference(validation.Parameters["a_initial"], validation.Parameters["omega_m"])[0]
        df = validation.growth_reference(validation.Parameters["a_final"], validation.Parameters["omega_m"])[0]
        for kind in ("pancake", "coupled3d"):
            frame, description = validation.make_fixture(kind, n)
            displacement = validation.periodic_difference(frame[["x", "y", "z"]].to_numpy(), q, box)
            recovered = []
            for mode, finalAmplitude, phase in description["modes"]:
                wave = np.asarray(mode)*2*np.pi/box
                coefficient = np.mean(displacement*np.exp(-1j*(q@wave))[:, None], axis=0)
                delta = -1j*np.dot(wave, coefficient)
                expected = .5*finalAmplitude*di/df*np.exp(1j*phase)
                self.assertLess(abs(delta-expected), 3e-14)
                recovered.append(delta)
            recoveredRms = math.sqrt(2*np.sum(np.abs(recovered)**2))*df/di
            self.assertAlmostEqual(recoveredRms, description["linear_final_density_rms"], places=12)
            self.assertAlmostEqual(recoveredRms, 1 if kind == "coupled3d" else 1.5/math.sqrt(2), places=12)

    def test_particle_density_transform_normalization_and_sign(self):
        box = 10.0
        # Two point particles provide an exact, unrelated oracle for the
        # forward sign and 1/Nparticle normalization (not 1/Nmesh^3).
        positions = np.asarray([[0., 0., 0.], [box/4, 0., 0.]])
        modes = np.asarray([[1, 0, 0], [2, 0, 0], [-1, 0, 0]])
        actual = validation.density_modes(positions, box, modes)
        np.testing.assert_allclose(actual, [.5-.5j, 0, .5+.5j], atol=2e-16)
        permuted = validation.density_modes(positions[::-1], box, modes)
        np.testing.assert_allclose(permuted, actual, atol=0, rtol=0)

    def test_global_translation_preserves_power_but_changes_phase(self):
        frame, _ = validation.make_fixture("pancake", 8)
        box = validation.Parameters["box_size"]
        positions = frame[["x", "y", "z"]].to_numpy()
        modes = validation.resolved_modes("pancake")
        offset = np.asarray([box/8, .037*box, .093*box])
        left = validation.density_modes(positions, box, modes)
        right = validation.density_modes((positions+offset) % box, box, modes)
        np.testing.assert_allclose(right, left*np.exp(-2j*np.pi*(modes@offset)/box), atol=1e-15)
        metrics = validation.density_comparison(left, right)
        self.assertAlmostEqual(metrics["power_ratio"], 1, places=12)
        self.assertGreater(metrics["complex_relative"], .7)
        self.assertLess(metrics["correlation"], .75)

    def test_complex_error_normalization_and_signed_correlation(self):
        coefficients = np.asarray([1+2j, 3-4j])
        amplitude = validation.density_comparison(coefficients, 2*coefficients)
        self.assertAlmostEqual(amplitude["power_left"], 30)
        self.assertAlmostEqual(amplitude["power_right"], 120)
        self.assertAlmostEqual(amplitude["power_ratio"], .25)
        self.assertAlmostEqual(amplitude["complex_relative"], 1/math.sqrt(2))
        self.assertAlmostEqual(amplitude["correlation"], 1)
        phase = validation.density_comparison(coefficients, 1j*coefficients)
        self.assertAlmostEqual(phase["complex_relative"], math.sqrt(2))
        self.assertAlmostEqual(phase["correlation"], 0)
        reversedPhase = validation.density_comparison(coefficients, -coefficients)
        self.assertAlmostEqual(reversedPhase["correlation"], -1)

    def test_uniform_and_tiny_power_are_not_falsely_normalized(self):
        box = validation.Parameters["box_size"]
        uniform = lattice_frame(8, box)
        delta = validation.density_modes(uniform[["x", "y", "z"]].to_numpy(), box,
                                        validation.resolved_modes("coupled3d"))
        self.assertLess(np.linalg.norm(delta), 2e-14)
        for left, right in ((np.zeros(4), np.zeros(4)), (1e-20*np.ones(4), 2e-20*np.ones(4)),
                            (np.zeros(4), np.ones(4))):
            result = validation.density_comparison(left, right)
            self.assertFalse(result["normalization_defined"])
            self.assertIsNone(result["power_ratio"])
            self.assertIsNone(result["complex_relative"])
            self.assertIsNone(result["correlation"])
            self.assertTrue(math.isfinite(result["absolute_difference"]))
        # One vanishing code must remain visible as a mismatch, not be hidden
        # by a normalized-ratio NaN or a claimed correlation of one.
        self.assertEqual(validation.density_comparison(np.zeros(4), np.ones(4))["absolute_difference"], 2)

    def test_unique_modes_exclude_dc_and_double_counting(self):
        modes = validation.resolved_modes("coupled3d")
        modeSet = set(map(tuple, modes))
        self.assertEqual(len(modeSet), len(modes))
        self.assertNotIn((0, 0, 0), modeSet)
        for mode in modes:
            self.assertGreater(next(value for value in mode if value), 0)
            self.assertLessEqual(np.dot(mode, mode), 16)
            self.assertNotIn(tuple(-mode), modeSet)
        # Count the full lattice ball independently, without the implementation
        # loop or lexicographic representative-selection rule.
        z, y, x = np.mgrid[-4:5, -4:5, -4:5]
        full = ((x*x+y*y+z*z > 0) & (x*x+y*y+z*z <= 16)).sum()
        self.assertEqual(2*len(modes), full)
        np.testing.assert_array_equal(validation.resolved_modes("pancake"),
                                      [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])

    def test_periodic_phase_space_and_wrong_momentum_units(self):
        right, _ = validation.make_fixture("pancake", 8)
        left = right.copy()
        box = validation.Parameters["box_size"]
        left["x"] = (left.x+.002*box) % box
        left[["px", "py", "pz"]] *= .2
        result = validation.phase_space_metrics(left, right, box, 32)
        self.assertAlmostEqual(result["position_rms"], .002*box, places=12)
        self.assertAlmostEqual(result["position_cells"], .002*32, places=12)
        self.assertAlmostEqual(result["momentum_relative"], .8, places=12)
        uniform = lattice_frame(8, box)
        self.assertIsNone(validation.phase_space_metrics(uniform, uniform, box, 32)["momentum_relative"])

    def test_fold_detection_is_not_a_postcrossing_dynamics_oracle(self):
        n, box = 16, validation.Parameters["box_size"]
        for amplitude, folded in ((0., False), (.8, False), (1.5, True)):
            frame = lattice_frame(n, box)
            qx = frame.x.to_numpy()
            frame.x = (qx-amplitude*box/(2*np.pi)*np.sin(2*np.pi*qx/box+.17)) % box
            result = validation.planar_ordering(frame, n, box)
            self.assertEqual(result["minimum_sampled_jacobian"] < 0, folded)
            self.assertEqual(result["negative_interval_fraction"] > 0, folded)

    def test_fold_outside_first_row_is_not_missed(self):
        n, box = 16, validation.Parameters["box_size"]
        frame = lattice_frame(n, box)
        indices = np.arange(n, 2*n)
        qx = frame.loc[indices, "x"].to_numpy()
        frame.loc[indices, "x"] = (qx-1.5*box/(2*np.pi)*np.sin(2*np.pi*qx/box+.17)) % box
        result = validation.planar_ordering(frame, n, box)
        self.assertLess(result["minimum_sampled_jacobian"], 0)
        self.assertGreater(result["negative_interval_fraction"], 0)

    def test_refinement_order_and_precision_classification(self):
        # A fixed C*dt^2 error gives a successive-difference ratio of four.
        quadratic = validation.refinement(abs(1/64**2-1/128**2),
                                          abs(1/128**2-1/256**2), 1e-9, 2, 6)
        self.assertTrue(quadratic["passed"])
        self.assertAlmostEqual(quadratic["ratio"], 4)
        self.assertAlmostEqual(quadratic["measured_order"], 2)
        self.assertFalse(validation.refinement(.1, .11, 1e-9, 1.5)["passed"])
        limited = validation.refinement(1e-8, 2e-8, 1e-7, 2, 6)
        self.assertEqual(limited["status"], "precision_limited")
        self.assertIsNone(limited["measured_order"])
        zeroFine = validation.refinement(1., 0., 1e-7, 2, 6)
        self.assertIsNone(zeroFine["measured_order"])

    def test_nonfinite_and_incompatible_fourier_inputs_are_rejected(self):
        for left, right in (([np.nan], [1]), ([1], [np.inf]), (np.ones(2), np.ones((2, 1))),
                            ([], [])):
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                validation.density_comparison(left, right)
        with self.assertRaises(ValueError):
            validation.density_modes(np.asarray([[np.nan, 0., 0.]]), 1, np.asarray([[1, 0, 0]]))
        with self.assertRaises(ValueError):
            validation.resolved_modes("unknown")

    def test_invalid_phase_space_and_ids_are_rejected(self):
        valid = lattice_frame(8)
        for column, value in (("px", np.nan), ("x", np.inf), ("id", 1)):
            invalid = valid.copy()
            invalid.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                validation.phase_space_metrics(invalid, invalid, 168.75, 32)
        unsorted = valid.iloc[::-1].reset_index(drop=True)
        with self.assertRaises(ValueError):
            validation.phase_space_metrics(unsorted, unsorted, 168.75, 32)
        with self.assertRaises(ValueError):
            validation.planar_ordering(unsorted, 8, 168.75)

    def test_invalid_refinement_arguments_are_rejected(self):
        arguments = [(-1., 1., 0., 1.5, None), (1., np.nan, 0., 1.5, None),
                     (1., .2, np.nan, 1.5, None), (1., .2, -1., 1.5, None),
                     (1., .2, 0., -1., None), (1., .2, 0., 2., 1.)]
        for values in arguments:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validation.refinement(*values)

    def test_factor_schedule_accepts_complete_geometric_schedule(self):
        data = synthetic_factor_table()
        self.assertIsNone(validation.validate_factor_schedule(data, 8, .02, .2))
        # Harmless binary64 endpoint/midpoint evaluation differences must fit
        # the explicitly declared schedule budget, not require bitwise equality.
        data["ah"] *= 1+5e-14
        self.assertIsNone(validation.validate_factor_schedule(data, 8, .02, .2))

    def test_factor_schedule_rejects_missing_and_duplicate_steps(self):
        complete = synthetic_factor_table()
        missing = complete.drop(index=3).reset_index(drop=True)
        duplicate = complete.copy()
        duplicate.loc[3, "step"] = 2
        fractional = complete.copy()
        fractional["step"] = fractional["step"].astype(float)
        fractional.loc[3, "step"] = 3.5
        for label, data in (("missing", missing), ("duplicate", duplicate), ("fractional", fractional)):
            with self.subTest(label=label), self.assertRaises(ValueError):
                validation.validate_factor_schedule(data, 8, .02, .2)

    def test_factor_schedule_rejects_wrong_intervals_and_midpoints(self):
        complete = synthetic_factor_table()
        wrongEndpoint = complete.copy()
        wrongEndpoint.loc[3, "a1"] *= 1.0001
        arithmeticMidpoint = complete.copy()
        arithmeticMidpoint["ah"] = (arithmeticMidpoint.a0+arithmeticMidpoint.a1)/2
        shiftedRange = synthetic_factor_table(ai=.021, af=.201)
        for label, data in (("endpoint", wrongEndpoint), ("arithmetic_midpoint", arithmeticMidpoint),
                            ("shifted_range", shiftedRange)):
            with self.subTest(label=label), self.assertRaises(ValueError):
                validation.validate_factor_schedule(data, 8, .02, .2)

    def test_factor_schedule_rejects_nonfinite_and_nonpositive_values(self):
        for column in synthetic_factor_table().columns:
            for invalid in (np.nan, np.inf, -np.inf):
                data = synthetic_factor_table()
                data[column] = data[column].astype(float)
                data.loc[2, column] = invalid
                with self.subTest(column=column, invalid=invalid), self.assertRaises(ValueError):
                    validation.validate_factor_schedule(data, 8, .02, .2)
        for column in ("a0", "ah", "a1", "drift", "canonical_kick0", "canonical_kick1"):
            for invalid in (0., -1.):
                data = synthetic_factor_table()
                data.loc[2, column] = invalid
                with self.subTest(column=column, invalid=invalid), self.assertRaises(ValueError):
                    validation.validate_factor_schedule(data, 8, .02, .2)

    def test_factor_schedule_rejects_missing_or_extra_columns(self):
        complete = synthetic_factor_table()
        for data in (complete.drop(columns="canonical_kick1"), complete.assign(unexpected=1)):
            with self.assertRaises(ValueError):
                validation.validate_factor_schedule(data, 8, .02, .2)

    def test_full_and_quick_case_coverage(self):
        full = validation.cases()
        self.assertEqual(len(full), 16)
        self.assertEqual(len({case.name for case in full}), 16)
        for kind in ("pancake", "coupled3d"):
            subset = [case for case in full if case.fixture == kind]
            self.assertEqual({case.mesh for case in subset}, {16, 32, 64})
            self.assertEqual({case.steps for case in subset}, {64, 128, 256})
            self.assertEqual({case.ranks for case in subset}, {1, 2, 3, 4})
        quick = validation.cases(True)
        self.assertEqual(len(quick), 4)
        self.assertEqual({case.ranks for case in quick}, {1, 2})


if __name__ == "__main__":
    unittest.main()
