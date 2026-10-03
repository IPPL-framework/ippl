#!/usr/bin/env python3
"""Analytical tests of the frozen-force oracle; no simulation binaries needed."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_frozen_force as vf


class FrozenForceAnalysis(unittest.TestCase):
    def test_cic_partition_adjoint_and_wrapping(self):
        n, box = 8, 10.
        rng = np.random.default_rng(20261003)
        points = rng.uniform(-3 * box, 4 * box, (n**3, 3))
        delta = vf.deposit(points, n, box)
        self.assertLess(abs(delta.sum()), 1e-12)
        field = rng.normal(size=(n, n, n, 3))
        np.testing.assert_allclose(vf.gather(field, points, box).sum(axis=0),
                                   ((delta + 1)[..., None] * field).sum(axis=(0, 1, 2)),
                                   atol=2e-12, rtol=1e-13)
        np.testing.assert_allclose(vf.wrap(np.array([[0., box, -box], [31., -21., 2.]]), box),
                                   [[0, 0, 0], [1, 9, 2]])

    def test_translated_uniform_lattice(self):
        n, box = 8, 32.
        points = (np.indices((n, n, n)).reshape(3, -1).T + [.71, 1.21, -.13]) * box / n
        delta = vf.deposit(points, n, box)
        np.testing.assert_allclose(delta, 0, atol=5e-15)
        np.testing.assert_allclose(vf.mesh_force(delta, box, .31), 0, atol=5e-15)

    def test_analytical_axis_and_oblique_modes(self):
        n, box, omega, amplitude = 16, 168.75, .31, .12
        grid = np.indices((n, n, n)).transpose(1, 2, 3, 0) * box / n
        for mode in ([1, 0, 0], [1, 2, -3]):
            k = np.asarray(mode) * 2 * np.pi / box
            phase = grid @ k
            delta = amplitude * np.cos(phase)
            exact = -1.5 * omega * amplitude * np.sin(phase)[..., None] * k / (k @ k)
            np.testing.assert_allclose(vf.mesh_force(delta, box, omega), exact, atol=1e-14)
            np.testing.assert_allclose(vf.mesh_force(delta, box, omega, "fastpm"), exact,
                                       atol=2e-7, rtol=5e-7)

    def test_omega_and_length_scaling(self):
        rng = np.random.default_rng(5117)
        delta = rng.normal(size=(8, 8, 8))
        for convention in ("ippl", "fastpm"):
            field = vf.mesh_force(delta, 64., .25, convention)
            np.testing.assert_allclose(vf.mesh_force(delta, 128., .5, convention), 4 * field,
                                       atol=1e-13, rtol=2e-14)

    def test_native_nyquist_difference_is_not_filtered(self):
        n = 16
        ix, _, iz = np.indices((n, n, n))
        delta = np.cos(np.pi * ix + 2 * np.pi * iz / n)
        ippl = vf.mesh_force(delta, 100., .31)
        native = vf.mesh_force(delta, 100., .31, "fastpm")
        self.assertLess(vf.rms(ippl[..., 0]), 1e-14)
        self.assertGreater(vf.rms(native[..., 0]), .1)
        self.assertLess(vf.rms(vf.common_band(native - ippl)), 1e-14)

    def test_common_band_particle_fixture(self):
        fixture = next(item for item in vf.fixtures() if item["name"] == "common_band")
        delta = vf.deposit(fixture["positions"], fixture["n"], fixture["box"])
        self.assertGreater(vf.rms(delta), .001)
        self.assertLess(vf.rms(delta - vf.common_band(delta)), 2e-14)

    def test_reject_wrong_sign_and_amplitude(self):
        field = np.arange(30.).reshape(10, 3) * .1
        checks = vf.Checks()
        checks.compare("correct", field, field, 1e-9, 5e-6)
        checks.compare("wrong_sign", -field, field, 1e-9, 5e-6)
        checks.compare("wrong_amplitude", 1.001 * field, field, 1e-9, 5e-6)
        self.assertEqual([row["passed"] for row in checks.rows], [True, False, False])


if __name__ == "__main__":
    unittest.main()
