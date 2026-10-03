"""Fast analytic tests of the independent cross-generator analysis (no MPI)."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_zarija as validation


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.geometry = validation.Geometry(16, 168.75)
        self.a = 1/50
        self.omega = 0.31

    def synthetic(self, code, transverse=False, momentumScale=1.0):
        geometry = self.geometry
        ids = np.arange(geometry.n**3)
        q = validation.lattice(ids, geometry, code)
        wave = 2*np.pi/geometry.box*np.array([2, -1, 3])
        phase = q@wave
        direction = np.array([1., 2., 0.]) if transverse else wave/np.dot(wave, wave)
        psi = -0.013*np.sin(phase)[:, None]*direction
        growth, rate = validation.growth_reference(self.a, self.omega)
        expansion = np.sqrt(self.omega/self.a**3+1-self.omega)
        positions = (q+growth*psi) % geometry.box
        positions[positions >= geometry.box] = 0.0
        return validation.Snapshot(ids, positions,
                                   momentumScale*self.a**2*expansion*rate*growth*psi)

    def test_longitudinal_origin_and_axis_order(self):
        coefficients = []
        for code in ("ippl", "zarija"):
            delta, metrics = validation.recover(self.synthetic(code), self.geometry, code, self.a, self.omega)
            self.assertLess(metrics["longitudinal_relative_error"], 1e-10)
            self.assertLess(metrics["momentum_relative_error"], 1e-10)
            self.assertLess(metrics["displacement_dc_relative_error"], 1e-10)
            self.assertAlmostEqual(delta[3, -1, 2].real, 0.013/2, places=12)
            self.assertAlmostEqual(delta[3, -1, 2].imag, 0.0, places=12)
            coefficients.append(delta)
        np.testing.assert_allclose(coefficients[0], coefficients[1], rtol=1e-9, atol=2e-13)

    def test_bad_momentum_is_detected(self):
        _, metrics = validation.recover(self.synthetic("ippl", momentumScale=1.05),
                                        self.geometry, "ippl", self.a, self.omega)
        self.assertGreater(metrics["momentum_relative_error"], 0.049)

    def test_transverse_displacement_is_detected(self):
        _, metrics = validation.recover(self.synthetic("ippl", transverse=True),
                                        self.geometry, "ippl", self.a, self.omega)
        self.assertGreater(metrics["longitudinal_relative_error"], 0.99)

    def test_common_mask_counts_unique_pairs(self):
        geometry = validation.Geometry(32, 168.75)
        self.assertEqual(geometry.unique.sum(), 14895)
        self.assertEqual(geometry.common.sum(), 29790)
        self.assertFalse(geometry.common[16].any())
        self.assertFalse(geometry.common[:, 16].any())
        self.assertFalse(geometry.common[:, :, 16].any())

    def test_gaussian_amplitude_error_is_detected(self):
        rng = np.random.default_rng(41601)
        z = (rng.normal(size=120000)+1j*rng.normal(size=120000))/np.sqrt(2)
        self.assertTrue(all(item["passed"] for item in validation.gaussian_metrics(z)))
        checks = validation.gaussian_metrics(1.1*z)
        self.assertFalse(next(item["passed"] for item in checks if item["statistic"] == "mean_power"))

    def test_binary_layout_and_velocity_units(self):
        snapshot = self.synthetic("zarija")
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)/"particles"
            count = len(snapshot.ids)//2
            for rank in (0, 1):
                source = slice(rank*count, (rank+1)*count)
                records = np.empty(count, dtype=validation.BinaryDtype)
                records["id"] = snapshot.ids[source]
                for axis, name in enumerate(("x", "y", "z")):
                    records[name] = snapshot.positions[source, axis]
                    records["v"+name] = snapshot.momentum[source, axis]*100/self.a**2
                records.tofile(Path(str(prefix)+f".bin.{rank}"))
            recovered = validation.read_zarija(prefix, 2, self.geometry, self.a)
            np.testing.assert_array_equal(recovered.positions, snapshot.positions)
            np.testing.assert_allclose(recovered.momentum, snapshot.momentum, rtol=3e-16)
            # Deliberately corrupt an ID without changing the file size.
            path = Path(str(prefix)+".bin.1")
            data = np.fromfile(path, dtype=validation.BinaryDtype)
            data["id"][0] = 0
            data.tofile(path)
            with self.assertRaisesRegex(ValueError, "IDs/order"):
                validation.read_zarija(prefix, 2, self.geometry, self.a)

    def test_wrong_binary_precision_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)/"particles"
            Path(str(prefix)+".bin.0").write_bytes(bytes(28*self.geometry.n**3))
            with self.assertRaisesRegex(ValueError, "ABI/file size"):
                validation.read_zarija(prefix, 1, self.geometry, self.a)

    def test_table_normalizes_first_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"transfer.tf"
            path.write_text("0.00001 20000000 20000000\n0.1 10000000 10000000\n10 10000 10000\n")
            wave = np.asarray([0.0, 0.000005, 0.001, 0.1, 1.0])
            first = validation.table_spectrum(wave, validation.Parameters, path)
            path.write_text("0.00001 2 2\n0.1 1 1\n10 0.001 0.001\n")
            second = validation.table_spectrum(wave, validation.Parameters, path)
            np.testing.assert_allclose(first, second, rtol=2e-14)
            self.assertEqual(first[0], 0.0)
            self.assertTrue(np.isfinite(first).all())


if __name__ == "__main__":
    unittest.main()
