#!/usr/bin/env python3
"""Independent synthesis, statistics, units, and provenance checks; no simulations."""

import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gaussian_fixture as fixture


def direct_displacement(realization, positions):
    """Explicit mode sum with xyz positions, independent of FFT/index convention."""
    wave = realization.modes * (2*np.pi/realization.parameters["box_size"])
    amplitude = (1j*realization.coefficients[:, None]*wave
                 / np.sum(wave*wave, axis=1)[:, None])
    return np.real(np.exp(1j*positions @ wave.T) @ amplitude)


def recover_coefficients(realization, n, displacement):
    """delta=-div psi using a forward DFT and explicit half-cell dephasing."""
    wave = realization.modes * (2*np.pi/realization.parameters["box_size"])
    indices = tuple(realization.modes[:, component] % n for component in (2, 1, 0))
    psi = np.asarray([np.fft.fftn(displacement[:, component].reshape(n, n, n),
                                 norm="forward")[indices] for component in range(3)]).T
    return (-1j*np.sum(wave*psi, axis=1)
            * np.exp(-1j*np.pi*realization.modes.sum(axis=1)/n))


def independent_spectrum(k, parameters):
    """Dense Simpson in log(k), independent of imported Gauss--Legendre normalization."""
    def shape(wave):
        q = wave/(parameters["Omega_m"]*parameters["hubble"])
        transfer = (np.log(1+2.34*q)/(2.34*q)
                    / (1+3.89*q+(16.1*q)**2+(5.46*q)**3+(6.71*q)**4)**.25)
        return wave**parameters["n_s"]*transfer**2

    logK = np.linspace(np.log(1e-8), np.log(10), 65537)
    wave = np.exp(logK)
    x = 8*wave
    window = np.ones_like(x)
    small = x < .02
    window[small] = 1-x[small]**2/10+x[small]**4/280-x[small]**6/15120
    window[~small] = 3*(np.sin(x[~small])-x[~small]*np.cos(x[~small]))/x[~small]**3
    integrand = wave**3*shape(wave)*window**2/(2*np.pi**2)
    rawVariance = (logK[1]-logK[0])/3*(integrand[0]+integrand[-1]
                   + 4*integrand[1:-1:2].sum()+2*integrand[2:-1:2].sum())
    return parameters["Sigma_8"]**2/rawVariance*shape(k)


class GaussianFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.realization = fixture.make_realization(20261003)

    def test_physical_mode_rng_is_hermitian_and_has_fixed_encoding(self):
        seed, mode = 17, (2, -1, 0)
        # Independent byte-level oracle for the documented RNG contract.
        digest = hashlib.sha256(b"IPPL-band-limited-Gaussian-v1\0"
                                + struct.pack("<Qiii", seed, *mode)).digest()
        u = ((int.from_bytes(digest[:8], "little") >> 12)+.5)/2**52
        v = ((int.from_bytes(digest[8:16], "little") >> 12)+.5)/2**52
        expected = np.sqrt(-np.log(u))*np.exp(2j*np.pi*v)
        value = fixture.mode_gaussian(seed, mode)
        self.assertAlmostEqual(abs(value-expected), 0, delta=3e-16)
        self.assertEqual(fixture.mode_gaussian(seed, tuple(-m for m in mode)), value.conjugate())
        self.assertNotEqual(value, fixture.mode_gaussian(seed+1, mode))
        self.assertNotEqual(value, fixture.mode_gaussian(seed, (1, -2, 0)))

    def test_gaussian_ensemble_moments_without_realization_normalization(self):
        count = 8192
        values = np.asarray([fixture.mode_gaussian(seed, (1, 2, -3)) for seed in range(count)])
        others = np.asarray([fixture.mode_gaussian(seed, (2, 1, -3)) for seed in range(count)])
        # Fixed six-standard-error limits; independent complex coefficients
        # should have zero means, E|g|²=1, real/imag variance=1/2, E[g²]=0.
        self.assertLess(abs(values.mean()), 6/np.sqrt(count))
        self.assertLess(abs(np.mean(abs(values)**2)-1), 6/np.sqrt(count))
        self.assertLess(abs(np.mean(values.real**2)-.5), 6/np.sqrt(2*count))
        self.assertLess(abs(np.mean(values.imag**2)-.5), 6/np.sqrt(2*count))
        self.assertLess(abs(np.mean(values**2)), 6*np.sqrt(2/count))
        self.assertLess(abs(np.mean(values*others.conj())), 6/np.sqrt(count))
        # Gaussian amplitudes must not have been replaced by fixed amplitudes.
        self.assertGreater(float(np.var(abs(values)**2)), .8)

    def test_spherical_band_dc_nyquist_and_exact_hermitian_pairs(self):
        field = self.realization
        self.assertEqual(len(field.modes), 7152)
        self.assertTrue(np.all(np.linalg.norm(field.modes, axis=1) <= 12))
        self.assertFalse(np.any(np.all(field.modes == 0, axis=1)))
        self.assertFalse(np.any(np.abs(field.modes) == 16))
        byMode = dict(zip(map(tuple, field.modes), field.coefficients))
        for mode, value in byMode.items():
            self.assertEqual(byMode[tuple(-x for x in mode)], value.conjugate())
        self.assertFalse(field.modes.flags.writeable)
        self.assertFalse(field.coefficients.flags.writeable)

    def test_shared_modes_independent_of_cutoff_and_baryon_metadata(self):
        small = fixture.make_realization(20261003, cutoff=3)
        byMode = dict(zip(map(tuple, self.realization.modes), self.realization.coefficients))
        self.assertTrue(np.array_equal(small.coefficients,
                        np.asarray([byMode[tuple(mode)] for mode in small.modes])))
        noBaryons = fixture.make_realization(20261003, cosmology={"Omega_bar": 0})
        self.assertTrue(np.array_equal(noBaryons.coefficients, self.realization.coefficients))
        self.assertEqual(noBaryons.coefficient_sha256, self.realization.coefficient_sha256)

    def test_bbks_normalization_against_independent_simpson_quadrature(self):
        field = self.realization
        k = 2*np.pi/field.parameters["box_size"]*np.linalg.norm(field.modes, axis=1)
        reference = independent_spectrum(k, field.parameters)
        np.testing.assert_allclose(field.power, reference, rtol=2e-8, atol=0)
        doubled = fixture.make_realization(20261003, cosmology={"Sigma_8": 1.64})
        np.testing.assert_array_equal(doubled.coefficients, 2*field.coefficients)
        np.testing.assert_array_equal(doubled.power, 4*field.power)

    def test_actual_density_coefficient_has_power_over_volume_normalization(self):
        field = self.realization
        mode = (2, -1, 0)
        index = int(np.flatnonzero(np.all(field.modes == mode, axis=1))[0])
        digest = hashlib.sha256(b"IPPL-band-limited-Gaussian-v1\0"
                                + struct.pack("<Qiii", 20261003, *mode)).digest()
        u = ((int.from_bytes(digest[:8], "little") >> 12)+.5)/2**52
        v = ((int.from_bytes(digest[8:16], "little") >> 12)+.5)/2**52
        k = 2*np.pi/168.75*np.sqrt(5)
        power = independent_spectrum(k, field.parameters)
        expected = np.sqrt(-np.log(u)*power/168.75**3)*np.exp(2j*np.pi*v)
        self.assertAlmostEqual(abs(field.coefficients[index]-expected)/abs(expected),
                               0, delta=1e-8)

    def test_direct_single_oblique_mode_has_correct_sign_axis_phase_amplitude(self):
        modes = np.asarray([[-1, 2, -1], [1, -2, 1]], dtype=np.int32)
        # A complex cosine with an arbitrary phase exercises every convention.
        coefficients = np.asarray([.1-.07j, .1+.07j])
        field = fixture.GaussianRealization(0, 3, fixture.DefaultCosmology,
                                            modes, coefficients, np.ones(2), "test")
        n = 8
        q = fixture.lattice_positions(n, 168.75)
        sampled = fixture.sample_displacement(field, n)
        expected = direct_displacement(field, q)
        np.testing.assert_allclose(sampled, expected, rtol=2e-13, atol=5e-15)
        np.testing.assert_allclose(recover_coefficients(field, n, sampled), coefficients,
                                   rtol=2e-15, atol=1e-16)

    def test_np32_and_np64_recover_identical_continuous_coefficients(self):
        field = self.realization
        for n in (32, 64):
            with self.subTest(n=n):
                displacement = fixture.sample_displacement(field, n)
                np.testing.assert_allclose(recover_coefficients(field, n, displacement),
                                           field.coefficients, rtol=3e-13, atol=4e-18)
                # Check unrelated physical positions by direct sum, not a DFT.
                ids = np.asarray([0, 19, n**2+3*n+7, n**3-1])
                q = fixture.lattice_positions(n, 168.75)[ids]
                np.testing.assert_allclose(displacement[ids], direct_displacement(field, q),
                                           rtol=5e-13, atol=5e-13)

    def test_deformation_recovers_density_and_is_symmetric(self):
        field = fixture.make_realization(42, cutoff=2)
        n = 8
        derivative = fixture.sample_deformation(field, n)
        np.testing.assert_array_equal(derivative, derivative.transpose(0, 2, 1))
        sampledDelta = -np.trace(derivative, axis1=1, axis2=2).reshape(n, n, n)
        fft = np.fft.fftn(sampledDelta, norm="forward")
        index = tuple(field.modes[:, component] % n for component in (2, 1, 0))
        recovered = fft[index]*np.exp(-1j*np.pi*field.modes.sum(axis=1)/n)
        np.testing.assert_allclose(recovered, field.coefficients, rtol=4e-15, atol=3e-18)

    def test_eds_growth_and_canonical_momentum_against_analytic_formula(self):
        field = fixture.make_realization(17, cutoff=2, cosmology={"Omega_m": 1})
        n, redshift = 8, 49
        frame, metadata = fixture.make_gaussian_fixture(n, redshift, 17, cutoff=2,
                                                       cosmology={"Omega_m": 1})
        a = 1/(1+redshift)
        displacement = fixture.sample_displacement(field, n)
        np.testing.assert_allclose(frame[["x", "y", "z"]],
            np.remainder(fixture.lattice_positions(n, 168.75)+a*displacement, 168.75),
            rtol=0, atol=4e-14)
        # EdS: E=a^-3/2, D=a, f=1 => p=a^3/2 psi0.
        expected = (a**1.5*displacement).astype(np.float32).astype(np.float64)
        np.testing.assert_array_equal(frame[["px", "py", "pz"]].to_numpy(), expected)
        self.assertAlmostEqual(metadata["D_initial"], a, delta=2e-15)
        self.assertAlmostEqual(metadata["f_initial"], 1, delta=2e-12)

    def test_redshift_changes_growth_not_continuous_phases_or_field(self):
        frames, records = [], []
        for redshift in (49, 99):
            frame, metadata = fixture.make_gaussian_fixture(8, redshift, 9, cutoff=2)
            frames.append(frame)
            records.append(metadata)
        self.assertEqual(records[0]["coefficient_sha256"], records[1]["coefficient_sha256"])
        q = fixture.lattice_positions(8, 168.75)
        displacements = []
        for frame, metadata in zip(frames, records):
            delta = frame[["x", "y", "z"]].to_numpy()-q
            delta -= 168.75*np.rint(delta/168.75)
            displacements.append(delta/metadata["D_initial"])
            factor = metadata["a_initial"]**2*metadata["E_initial"]*metadata["f_initial"]
            raw = factor*delta
            np.testing.assert_allclose(frame[["px", "py", "pz"]], raw,
                                       rtol=7e-8, atol=1e-14)
        np.testing.assert_allclose(*displacements, rtol=2e-11, atol=2e-12)

    def test_parseval_finite_band_variances_not_forced_to_sigma8(self):
        field = self.realization
        metadata = field.metadata()
        expected = np.sum(field.power)/168.75**3
        realized = np.sum(abs(field.coefficients)**2)
        self.assertAlmostEqual(metadata["finite_band_density_variance_expected_z0"], expected,
                               delta=4e-15*expected)
        self.assertAlmostEqual(metadata["finite_band_density_variance_realized_z0"], realized)
        self.assertNotEqual(expected, realized)
        self.assertNotEqual(metadata["finite_band_sigma8_squared_expected_z0"], .82**2)
        self.assertNotEqual(metadata["finite_band_sigma8_squared_realized_z0"], .82**2)
        n = 32
        mesh = np.zeros((n, n, n), dtype=np.complex128)
        index = tuple(field.modes[:, component] % n for component in (2, 1, 0))
        mesh[index] = field.coefficients*np.exp(1j*np.pi*field.modes.sum(axis=1)/n)
        density = np.fft.ifftn(mesh, norm="forward").real
        self.assertAlmostEqual(float(np.mean(density*density)), realized, delta=1e-14)
        np.testing.assert_array_equal(fixture.top_hat_window([0]), [1])

    def test_fixture_contract_determinism_hash_json_and_quantization(self):
        frame, metadata = fixture.make_gaussian_fixture(8, 49, 123, cutoff=2)
        repeated, repeatedMetadata = fixture.make_gaussian_fixture(8, 49, 123, cutoff=2)
        pd.testing.assert_frame_equal(frame, repeated)
        self.assertEqual(metadata, repeatedMetadata)
        self.assertEqual(tuple(frame.columns), fixture.FixtureColumns)
        np.testing.assert_array_equal(frame.id, np.arange(8**3))
        np.testing.assert_array_equal(frame.mass, 1.)
        self.assertTrue(np.all((frame[["x", "y", "z"]] >= 0)
                               & (frame[["x", "y", "z"]] < 168.75)))
        momentum = frame[["px", "py", "pz"]].to_numpy()
        np.testing.assert_array_equal(momentum, momentum.astype(np.float32).astype(np.float64))
        self.assertGreater(metadata["minimum_sampled_initial_map_eigenvalue"], 0)
        self.assertGreater(metadata["minimum_sampled_initial_map_jacobian"], 0)
        self.assertFalse(metadata["realized_variance_fitted"])
        self.assertEqual(metadata["phase_space_sha256"], fixture.phase_space_sha256(frame))
        json.dumps(metadata, allow_nan=False)
        frame.loc[0, "x"] += .1
        self.assertNotEqual(metadata["phase_space_sha256"], fixture.phase_space_sha256(frame))

    def test_campaign_sampling_and_starting_epochs_have_unfolded_sampled_maps(self):
        hashes = set()
        for n in (32, 64):
            for redshift in (49, 99):
                with self.subTest(n=n, redshift=redshift):
                    frame, metadata = fixture.make_gaussian_fixture(n, redshift, 20261003)
                    self.assertEqual(len(frame), n**3)
                    self.assertGreater(metadata["minimum_sampled_initial_map_eigenvalue"], .8)
                    self.assertGreater(metadata["minimum_sampled_initial_map_jacobian"], .7)
                    self.assertLess(metadata["momentum_rounding_relative_l2"], np.finfo(np.float32).eps)
                    hashes.add(metadata["coefficient_sha256"])
        self.assertEqual(hashes, {self.realization.coefficient_sha256})

    def test_invalid_parameters_fail_without_resampling(self):
        for parameters in ({"Omega_m": 0}, {"Omega_bar": .9}, {"Omega_r": .01},
                           {"Sigma_8": float("nan")}, {"box_size": -1}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                fixture.make_realization(1, cosmology=parameters)
        for kwargs in ({"seed": -1}, {"seed": 2**64}, {"seed": True},
                       {"seed": 1, "cutoff": 13}, {"seed": 1, "cutoff": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fixture.make_realization(**kwargs)
        for n in (16, 25, 24, 32.0):
            with self.subTest(n=n), self.assertRaises(ValueError):
                fixture.sample_displacement(self.realization, n)
        for redshift in (-1, np.inf, np.nan):
            with self.subTest(redshift=redshift), self.assertRaises(ValueError):
                fixture.make_gaussian_fixture(8, redshift, 1, cutoff=2)
        with self.assertRaisesRegex(ValueError, "nonpositive"):
            fixture.make_gaussian_fixture(8, 0, 1, cutoff=2, cosmology={"Sigma_8": 100})


if __name__ == "__main__":
    unittest.main()
