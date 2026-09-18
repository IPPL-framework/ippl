"""Independent physical identities and CLI regression checks; run with unittest."""

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.special import k0, k1

from chdr_1d import (BeamCase, ELEMENTARY_CHARGE, EPSILON_0, LIGHT_SPEED,
                     MU_0, PI, VACUUM_IMPEDANCE, bunchSpectra, main,
                     reconstructFields, solveMode, spectralRadiation)


class HalfSpaceTests(unittest.TestCase):
    def setUp(self):
        self.case = BeamCase()

    def test_interface_conditions_and_maxwell(self):
        # Include a propagating mode, ky=0, and an evanescent dielectric mode.
        for energy in [0.1, 60.0]:
            case = replace(self.case, energyMeV=energy)
            for ky in [0.0, 100.0, 500.0, -100.0]:
                with self.subTest(energy=energy, ky=ky):
                    mode = solveMode(case, 16e9, ky)
                    vacuumE = mode.incidentE + mode.reflectedE
                    vacuumH = mode.incidentH + mode.reflectedH
                    assert_allclose(vacuumE[1:], mode.transmittedE[1:], rtol=2e-11, atol=1e-32)
                    assert_allclose(vacuumH, mode.transmittedH, rtol=2e-11, atol=1e-32)
                    assert_allclose(vacuumE[0], case.epsilonR * mode.transmittedE[0], rtol=2e-11, atol=1e-32)
                    wave = np.array([-mode.normalWave, ky, mode.kz])
                    omega = 2 * PI * 16e9
                    assert_allclose(np.cross(wave, mode.transmittedE), omega * MU_0 * mode.transmittedH, rtol=2e-11, atol=1e-28)
                    assert_allclose(np.cross(wave, mode.transmittedH), -omega * EPSILON_0 * case.epsilonR * mode.transmittedE, rtol=2e-11, atol=1e-28)

    def test_outgoing_and_evanescent_branches(self):
        propagating = solveMode(self.case, 16e9, 0)
        evanescent = solveMode(self.case, 16e9, 1000)
        self.assertGreater(propagating.normalWave.real, 0)
        self.assertGreater(evanescent.normalWave.imag, 0)
        self.assertLess(abs(np.exp(-1j * evanescent.normalWave * -0.002)), 1)
        self.assertAlmostEqual(self.case.angleDeg, 46.7476, places=4)

    def test_vacuum_and_below_threshold_radiation_vanish(self):
        for case in [replace(self.case, epsilonR=1), replace(self.case, energyMeV=0.1)]:
            assert_allclose(spectralRadiation(case, 16e9), 0, atol=0)
        vacuum = solveMode(replace(self.case, epsilonR=1), 16e9, 100)
        assert_allclose(vacuum.reflectedE, 0, atol=1e-32)
        assert_allclose(vacuum.reflectedH, 0, atol=1e-32)

    def test_gap_dependence_of_each_mode(self):
        for ky in [0, 100, 1000]:
            first = solveMode(self.case, 16e9, ky)
            secondCase = replace(self.case, gapMm=3)
            second = solveMode(secondCase, 16e9, ky)
            factor = math.exp(-first.kappa * (secondCase.gap - self.case.gap))
            assert_allclose(second.transmittedE, factor * first.transmittedE, rtol=1e-12, atol=1e-32)
            assert_allclose(second.transmittedH, factor * first.transmittedH, rtol=1e-12, atol=1e-32)

    def test_energy_flux_matches_work_on_electron(self):
        for energy, gap, frequency in [(0.5, 0.2, 1e9), (60, 1, 16e9), (60, 3, 80e9)]:
            with self.subTest(energy=energy, gap=gap, frequency=frequency):
                result = spectralRadiation(replace(self.case, energyMeV=energy, gapMm=gap), frequency, 1e-9)
                self.assertGreater(result[0], 0)
                assert_allclose(result[0], result[1], rtol=2e-9, atol=0)
                self.assertLess(result[2] / result[0], 1e-8)

    def test_scale_invariance_and_quadrature_convergence(self):
        reference = spectralRadiation(self.case, 16e9, 1e-10)
        scaled = spectralRadiation(replace(self.case, gapMm=self.case.gapMm / 2), 32e9, 1e-10)
        assert_allclose(scaled[:2], 2 * reference[:2], rtol=1e-9, atol=0)
        coarse = spectralRadiation(self.case, 16e9, 1e-6)
        assert_allclose(coarse[:2], reference[:2], rtol=2e-6, atol=0)

    def test_reconstructed_vacuum_fields_against_bessel_solution(self):
        # Analytic time-Fourier Lorenz potential q*K0(alpha*r)/(2*pi*eps0*v).
        # This independently tests the ky integral and its absolute 2*pi factor.
        case = replace(self.case, epsilonR=1)
        frequency = 16e9
        x = np.array([-0.002, 0.0004, 0.002])
        y = np.array([0.0007, -0.0003, 0.0005])
        z = np.array([0.0, 0.002, -0.003])
        electric, magnetic = reconstructFields(case, frequency, x, y, z, 1e-10)
        omega = 2 * PI * frequency
        alpha = omega / (case.gamma * case.velocity)
        radius = np.hypot(x - case.gap, y)
        factor = -ELEMENTARY_CHARGE / (2 * PI * EPSILON_0 * case.velocity)
        potential = factor * k0(alpha * radius)
        radial = factor * alpha * k1(alpha * radius)
        expectedE = np.stack([radial * (x - case.gap) / radius, radial * y / radius,
                             -1j * omega / (case.gamma**2 * case.velocity) * potential], axis=-1)
        expectedH = EPSILON_0 * case.velocity * np.stack([-expectedE[:, 1], expectedE[:, 0], np.zeros(3)], axis=-1)
        phase = np.exp(1j * omega / case.velocity * z)[:, None]
        assert_allclose(electric, expectedE * phase, rtol=2e-7, atol=1e-30)
        # TE/TM recombination leaves roundoff in the analytically zero Hz.
        # Bound it relative to the independently known nonzero field scale.
        zeroTolerance = 5e-12 * np.max(np.abs(expectedH))
        assert_allclose(magnetic, expectedH * phase, rtol=2e-7, atol=zeroTolerance)

    def test_field_quadrature_and_transverse_symmetry(self):
        fineE, fineH = reconstructFields(self.case, 16e9, [-0.002, -0.002], [0.0008, -0.0008], relativeTolerance=1e-10)
        assert_allclose(fineE[0], fineE[1] * np.array([1, -1, 1]), rtol=1e-10, atol=1e-30)
        assert_allclose(fineH[0], fineH[1] * np.array([-1, 1, -1]), rtol=1e-10, atol=1e-32)
        coarseE, coarseH = reconstructFields(self.case, 16e9, -0.002, 0.0008, relativeTolerance=1e-7)
        assert_allclose(coarseE, fineE[0], rtol=2e-6, atol=1e-30)
        assert_allclose(coarseH, fineH[0], rtol=2e-6, atol=1e-32)

    def test_gaussian_duration_and_one_electron_limit(self):
        frequency = 1 / (2 * PI * self.case.sigmaTime)
        ff, incoherent, coherent, total = bunchSpectra(self.case, np.array([frequency]), np.array([1.0]))
        self.assertAlmostEqual(float(ff[0]), math.exp(-1), places=14)
        assert_allclose(total, incoherent + coherent)
        longer = bunchSpectra(replace(self.case, pulsePs=20), np.array([frequency]), np.array([1.0]))
        self.assertLess(longer[-1][0], total[0])
        singleCase = replace(self.case, chargeNc=ELEMENTARY_CHARGE * 1e9)
        single = bunchSpectra(singleCase, np.array([frequency]), np.array([2.0]))
        assert_allclose(single[-1], 2.0, rtol=1e-14)
        doubled = bunchSpectra(replace(self.case, chargeNc=10), np.array([frequency]), np.array([1.0]))
        assert_allclose(doubled[1], 2 * incoherent)
        n = self.case.electronCount
        assert_allclose(doubled[2] / coherent, (2 * n) * (2 * n - 1) / (n * (n - 1)))

    def test_invalid_parameters(self):
        for field in ["gapMm", "energyMeV", "pulsePs", "chargeNc", "epsilonR"]:
            for value in [0, -1, float("nan"), float("inf")]:
                with self.assertRaises(ValueError):
                    replace(self.case, **{field: value})
        with self.assertRaises(ValueError):
            reconstructFields(self.case, 16e9, self.case.gap)

    def test_cli_parameter_sweep_and_metadata(self):
        with tempfile.TemporaryDirectory(prefix="chdr-test-") as folder:
            main(["--gap-mm", "0.5", "2", "--pulse-ps", "5", "20", "--points", "4", "--no-fields", "--no-plots", "--output", folder])
            cases = list(Path(folder).glob("case_*"))
            self.assertEqual(len(cases), 4)
            for caseDir in cases:
                data = np.genfromtxt(caseDir / "spectrum.csv", delimiter=",", names=True)
                self.assertEqual(len(data), 4)
                self.assertTrue(np.all(data["bunch_total_J_per_m_per_Hz"] >= 0))
                metadata = json.loads((caseDir / "parameters.json").read_text())
                self.assertLess(metadata["flux_work_max_relative_difference"], 1e-8)
                self.assertIn(metadata["parameters"]["pulsePs"], [5, 20])


if __name__ == "__main__":
    unittest.main()
