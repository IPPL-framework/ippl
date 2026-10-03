"""Independent analytic pancake oracle tests; no simulation or MPI required."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_pancake as validation


class PancakeTests(unittest.TestCase):
    def setUp(self):
        self.parameters = {**validation.Parameters, "np": 16}
        self.parameters["amplitude"] = validation.initial_amplitude(0.8, self.parameters)
        self.ids = np.arange(16**3, dtype=np.uint64)
        self.ai, self.af = 1/50, 1/10

    def snapshot(self, a, parameters=None):
        parameters = self.parameters if parameters is None else parameters
        solution = validation.analytic_solution(self.ids, parameters, a)
        values = np.column_stack((solution["positions"], solution["momentum"]))
        frame = pd.DataFrame(values, columns=("x", "y", "z", "px", "py", "pz"))
        frame.insert(0, "id", self.ids)
        return frame

    def test_finite_amplitude_solution_initial_and_final(self):
        for a in (self.ai, self.af):
            metrics = validation.trajectory_metrics(self.snapshot(a), self.parameters, a)
            self.assertLess(metrics["displacement_relative_error"], 1e-14)
            self.assertLess(metrics["momentum_relative_error"], 1e-14)
            self.assertLess(metrics["net_momentum_relative"], 1e-14)
            self.assertGreater(metrics["minimum_sampled_planar_jacobian"], 0)
        final = validation.analytic_solution(self.ids, self.parameters, self.af)
        self.assertAlmostEqual(final["amplitude"], 0.8, places=14)
        self.assertAlmostEqual(final["minimum_continuum_jacobian"], 0.2, places=14)
        self.assertAlmostEqual(final["continuum_peak_density_contrast"], 4.0, places=13)

    def test_lattice_id_axis_order_and_sign(self):
        box = self.parameters["box_size"]
        q = validation.lattice(np.asarray([0, 1, 16, 256]), 16, box)
        np.testing.assert_allclose(q/(box/16), [[.5, .5, .5], [1.5, .5, .5],
                                                [.5, 1.5, .5], [.5, .5, 1.5]])
        solution = validation.analytic_solution(self.ids, self.parameters, self.af)
        self.assertLess(solution["displacement"][0, 0], 0)
        self.assertEqual(solution["displacement"][0, 1], 0)
        np.testing.assert_allclose(validation.periodic_difference(np.asarray([[0., box, -box]]),
                                                                 np.zeros((1, 3)), box), 0)

    def test_eds_canonical_momentum_has_correct_scale_factor(self):
        parameters = {**self.parameters, "Omega_m": 1.0}
        solution = validation.analytic_solution(self.ids, parameters, self.af)
        # EdS D=a, f=1 and a^2 E=a^(1/2), not a^(3/2) or a^(-1/2).
        np.testing.assert_allclose(solution["momentum"], np.sqrt(self.af)*solution["displacement"],
                                   rtol=2e-10, atol=1e-14)
        frame = self.snapshot(self.af)
        frame[["px", "py", "pz"]] *= self.af
        metrics = validation.trajectory_metrics(frame, self.parameters, self.af)
        self.assertAlmostEqual(metrics["momentum_relative_error"], 0.9, places=12)

    def test_oblique_planar_symmetry_and_transverse_corruption(self):
        parameters = {**self.parameters, "mode_y": 1}
        frame = self.snapshot(self.af, parameters)
        metrics = validation.trajectory_metrics(frame, parameters, self.af)
        self.assertLess(metrics["transverse_displacement_relative"], 1e-14)
        self.assertLess(metrics["transverse_momentum_relative"], 1e-14)
        self.assertGreater(metrics["minimum_sampled_planar_jacobian"], 0.2)
        frame["pz"] += 0.01
        corrupted = validation.trajectory_metrics(frame, parameters, self.af)
        self.assertGreater(corrupted["transverse_momentum_relative"], 1e-3)
        self.assertGreater(corrupted["net_momentum_relative"], 1e-3)

    def test_amplitude_error_is_measured_not_fitted_away(self):
        solution = validation.analytic_solution(self.ids, self.parameters, self.af)
        frame = self.snapshot(self.af)
        frame[["x", "y", "z"]] = (solution["q"]+1.1*solution["displacement"]) % self.parameters["box_size"]
        frame[["px", "py", "pz"]] *= 1.1
        metrics = validation.trajectory_metrics(frame, self.parameters, self.af)
        self.assertAlmostEqual(metrics["displacement_relative_error"], 0.1, places=13)
        self.assertAlmostEqual(metrics["momentum_relative_error"], 0.1, places=13)
        budget = validation.error_budgets(64, 0.8, (1, 0, 0))
        self.assertGreater(metrics["displacement_relative_error"], budget["displacement"])
        self.assertGreater(metrics["momentum_relative_error"], budget["momentum"])

    def test_no_post_crossing_oracle(self):
        for amplitude in (0.0, 1.0, 1.2):
            with self.assertRaisesRegex(ValueError, "requires"):
                validation.initial_amplitude(amplitude, self.parameters)
        parameters = {**self.parameters, "amplitude": .3}
        with self.assertRaisesRegex(ValueError, "post-crossing"):
            validation.analytic_solution(self.ids, parameters, self.af)

    def test_density_fundamental_is_not_linear_deformation_amplitude(self):
        measured = validation.continuum_density_modes(0.8)
        # Independent convergent Bessel power series J_n(x).
        import math
        expected = []
        for harmonic in (1, 2, 3, 4):
            x = harmonic*.8
            expected.append(2*sum((-1)**j*(x/2)**(2*j+harmonic)
                                  / (math.factorial(j)*math.factorial(j+harmonic))
                                  for j in range(30)))
        np.testing.assert_allclose(measured.real, expected, rtol=2e-14, atol=1e-14)
        np.testing.assert_allclose(measured.imag, 0, atol=2e-16)
        self.assertGreater(abs(measured[0]-.8)/.8, .07)

    def test_particle_integrity_failures_are_rejected(self):
        frame = self.snapshot(self.af)
        frame.loc[1, "id"] = 0
        with self.assertRaisesRegex(ValueError, "IDs"):
            validation.trajectory_metrics(frame, self.parameters, self.af)
        frame = self.snapshot(self.af)
        frame.loc[0, "px"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validation.trajectory_metrics(frame, self.parameters, self.af)


if __name__ == "__main__":
    unittest.main()
