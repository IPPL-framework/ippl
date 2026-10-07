#!/usr/bin/env python3
## @file test_native_ic_validation.py
# @brief Independent native-IC comparison oracle and corrupt-artifact rejection tests.
# @ingroup cosmology_python
# @details Tiny canonical fixtures exercise ID permutations, duplicate rejection,
# periodic residuals, x-fast half-cell Fourier recovery, 1LPT velocity units,
# fixed coefficients across grids, and failed/inconsistent preflight artifacts.
# These are host NumPy tests for the analysis harness; compiled CUDA/MPI native
# generation and 64/128 evolution are qualified separately by retained runs.
"""Run on Merlin; no production simulations or large FFTs are launched here."""

import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gaussian_fixture as gaussian
import quijote_io
import validate_native_ic as validator
from validate_linear import growth_reference


## @brief Build a small independent sample of the fixed Gaussian continuum field.
# @param grid Small even particle-lattice side for an isolated test fixture.
# @param seed Fixed Gaussian physical-mode realization seed.
# @param cutoff Positive spherical integer-mode ceiling below the fixture Nyquist.
# @return Canonical phase-space array and its background/count header.
def makeState(grid=8, seed=1, cutoff=2):
    realization = gaussian.make_realization(seed, cutoff=cutoff)
    a, box, omega = .01, 168.75, .31
    growth, rate = growth_reference(a, omega)
    displacement = growth * gaussian.sample_displacement(realization, grid)
    positions = (validator.lattice(grid, box) + displacement) % box
    momentum = (a*a*math.sqrt(omega/a**3 + 1-omega)*rate*displacement).astype(np.float32).astype(np.float64)
    return {"phase": np.column_stack((positions, momentum)),
            "header": {"total_count": grid**3, "a": a, "box_mpc_h": box, "omega_m": omega,
                       "omega_lambda": 1-omega, "hubble": .675, "particle_mass_msun_h": 1.}}


## @brief Serialize arbitrary ID subsets into exact canonical unordered shards.
# @param path Temporary canonical binary output path owned by the test.
# @param state Independent ID-ordered phase-space fixture from makeState.
# @param ids Explicit possibly permuted particle IDs assigned to this unordered shard.
# @return None; writes the exact v1 header and selected records.
def writeShard(path, state, ids):
    ids = np.asarray(ids, dtype=np.uint64)
    header = state["header"]
    records = np.empty(len(ids), dtype=quijote_io.RecordDtype)
    records["id"] = ids
    records["position"] = state["phase"][ids, :3]
    records["momentum"] = state["phase"][ids, 3:]
    with Path(path).open("wb") as stream:
        stream.write(quijote_io.HeaderStruct.pack(quijote_io.Magic, len(ids), header["total_count"],
                     header["a"], header["box_mpc_h"], header["omega_m"], header["omega_lambda"],
                     header["hubble"], header["particle_mass_msun_h"], 0))
        stream.write(records.tobytes())


## @brief Numerical oracle checks independent of any C++ native implementation.
class NativeOracleTests(unittest.TestCase):
    ## @brief Set up a small common seeded field once for independent tests.
    @classmethod
    def setUpClass(cls):
        ## @var state
        # @brief Shared tiny ID-ordered Gaussian initial state used only as immutable test input.
        cls.state = makeState()

    ## @brief Half-cell dephasing recovers identical target coefficients across grids.
    def test_same_continuous_mode_contract_across_grids(self):
        first = validator.initialAudit(self.state, 8, 1, 2, "float32")
        second = validator.initialAudit(makeState(16), 16, 1, 2, "float32")
        self.assertTrue(first["passed"])
        self.assertTrue(second["passed"])
        self.assertEqual(first["coefficient_sha256"], second["coefficient_sha256"])
        self.assertLess(first["coefficient_relative_l2"], 1e-10)
        self.assertLess(second["coefficient_relative_l2"], 1e-10)

    ## @brief Reversing both displacement and momentum preserves p proportional Psi but changes phases.
    def test_wrong_phase_sign_is_rejected(self):
        altered = copy.deepcopy(self.state)
        q = validator.lattice(8, 168.75)
        displacement = validator.minimumImage(altered["phase"][:, :3], q, 168.75)
        altered["phase"][:, :3] = (q - displacement) % 168.75
        altered["phase"][:, 3:] *= -1
        audit = validator.initialAudit(altered, 8, 1, 2, "float32")
        self.assertFalse(audit["passed"])
        self.assertGreater(audit["coefficient_relative_l2"], 1.)

    ## @brief A wrong seed cannot pass by matching ensemble shell amplitudes alone.
    def test_wrong_seed_is_rejected(self):
        audit = validator.initialAudit(self.state, 8, 2, 2, "float32")
        self.assertFalse(audit["passed"])

    ## @brief A coherent DC translation with matching momentum fails the zero-displacement-mean gate.
    def test_uniform_displacement_is_rejected(self):
        altered = copy.deepcopy(self.state)
        q = validator.lattice(8, 168.75)
        displacement = validator.minimumImage(altered["phase"][:, :3], q, 168.75)
        displacement[:, 0] += 1e-4
        _, rate = growth_reference(.01, .31)
        altered["phase"][:, :3] = (q + displacement) % 168.75
        altered["phase"][:, 3:] = (.01**2 * math.sqrt(.31/.01**3 + .69) * rate * displacement).astype(np.float32).astype(np.float64)
        audit = validator.initialAudit(altered, 8, 1, 2, "float32")
        self.assertFalse(audit["passed"])
        self.assertGreater(audit["displacement_dc_relative"], audit["coefficient_limit"])

    ## @brief Claimed float32 momentum must be exactly representable in that format.
    def test_false_float32_precision_claim_is_rejected(self):
        altered = copy.deepcopy(self.state)
        altered["phase"][0, 3] += 1e-14
        with self.assertRaisesRegex(ValueError, "float32 representable"):
            validator.initialAudit(altered, 8, 1, 2, "float32")

    ## @brief Position residuals use periodic minimum images, not raw boundary jumps.
    def test_periodic_difference_and_identity(self):
        metrics = validator.compareStates(self.state, self.state, 8)
        self.assertEqual(metrics["position_rms_mpc_h"], 0)
        self.assertEqual(metrics["momentum_relative_rms"], 0)
        np.testing.assert_allclose(validator.minimumImage(np.array([[.1, 0, 0]]),
                                                         np.array([[168.65, 0, 0]]), 168.75), [[.2, 0, 0]], atol=3e-14)

    ## @brief Explicit engineering budgets require present, finite metrics and valid maxima.
    def test_metric_limits_do_not_accept_missing_values(self):
        checks = validator.applyLimits({"a": None, "b": .2}, {"a": 1., "b": .1})
        self.assertFalse(checks["a"]["passed"])
        self.assertFalse(checks["b"]["passed"])
        with self.assertRaisesRegex(ValueError, "Unknown metric"):
            validator.applyLimits({}, {"missing": 1})
        with self.assertRaisesRegex(ValueError, "Invalid metric"):
            validator.applyLimits({"a": 0}, {"a": True})


## @brief Canonical multi-shard integrity and bounded validation-memory checks.
class NativeWireTests(unittest.TestCase):
    ## @brief Reassemble permuted even/odd MPI shards exactly by their uint64 IDs.
    def test_permuted_shards_reassemble_identically(self):
        state = makeState()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for rank, ids in enumerate((np.arange(0, 512, 2)[::-1], np.arange(1, 512, 2)[::-1])):
                writeShard(directory / f"particles_initial_rank{rank}.bin", state, ids)
            (directory / "snapshots.csv").write_text("name,step,a,z,ranks,format\ninitial,0,0.01,99,2,binary\n")
            loaded = validator.loadState(directory, "initial", 8, 512)
            np.testing.assert_array_equal(loaded["phase"], state["phase"])
            self.assertEqual(len(loaded["inputs"]), 2)

    ## @brief Duplicate or missing IDs are rejected even when total record counts agree.
    def test_duplicate_id_is_rejected(self):
        state = makeState()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.bin"
            ids = np.arange(512)
            ids[2] = 1
            writeShard(path, state, ids)
            with self.assertRaisesRegex(ValueError, "duplicated"):
                validator.loadState(path, "initial", 8, 512)

    ## @brief Truncation is rejected before numerical comparison.
    def test_truncated_binary_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "truncated.bin"
            writeShard(path, makeState(), np.arange(512))
            path.write_bytes(path.read_bytes()[:-8])
            with self.assertRaises(ValueError):
                validator.loadState(path, "initial", 8, 512)

    ## @brief The explicit particle-memory ceiling prevents accidental billion-particle arrays.
    def test_memory_cap_is_checked_before_state_allocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.bin"
            writeShard(path, makeState(), np.arange(512))
            with self.assertRaisesRegex(ValueError, "memory cap"):
                validator.loadState(path, "initial", 8, 64)


## @brief Preflight artifact failure injection; no solver or simulation is launched.
class NativePreflightTests(unittest.TestCase):
    ## @brief Materialize a minimal valid native receipt using the agreed artifact contract.
    def setUp(self):
        ## @var temporary
        # @brief Lifetime owner for this test case's isolated temporary artifact directory.
        self.temporary = tempfile.TemporaryDirectory()
        ## @var directory
        # @brief Temporary native-run directory containing independently authored artifact fixtures.
        self.directory = Path(self.temporary.name)
        ## @var value
        # @brief Mutable IC acceptance receipt copied or corrupted explicitly by individual tests.
        self.value = {"schema": "ippl-native-ic-check-v1", "passed": True, "native_1lpt": True,
                      "ic_mode": "gaussian", "failures": [],
                      "config": {"np": 8, "seed": 1, "ic_rng": "mode_hash_v1", "ic_mode_cutoff": 2,
                                 "ic_momentum_precision": "float32", "a_initial": .01, "box_mpc_h": 168.75},
                      "checks": {name: True for name in
                                 ("particle_count", "finite_wrapped_phase_space", "unit_pm_weights", "physical_particle_mass",
                                  "expected_id_signatures", "native_lagrangian_lattice", "native_momentum_relation",
                                  "native_finite_modes", "native_dc_zero", "native_cutoff_and_nyquist_zero",
                                  "native_inverse_reality", "native_declared_mode_coefficients", "native_selected_mode_recovery")},
                      "selected_modes": [{"integer_mode": [1, 0, 0], "expected_initial_real": .01,
                                          "expected_initial_imaginary": .02, "recovered_real": .01, "recovered_imaginary": .02,
                                          "absolute_error": 0., "tolerance": 1e-10, "passed": True,
                                          "declared_initial_real": .01, "declared_initial_imaginary": .02,
                                          "declared_absolute_error": 0., "declared_tolerance": 1e-10, "declared_passed": True}]}
        (self.directory / "metadata.txt").write_text("np=8\nseed=1\nic_mode_cutoff=2\nic_rng=mode_hash_v1\nic_mode=gaussian\nic_momentum_precision=float32\n")
        (self.directory / "pk_initial.csv").write_text("# Native linear input field; no shot-noise subtraction.\nshell_index,k_mean_h_mpc,modes_full,modes_independent,p_linear_initial_mpc_h_cubed,p_target_initial_mpc_h_cubed,ratio_to_target,gaussian_fractional_sigma\n1,0.05,26,13,0.1,0.11,0.909090909,0.2773501\n")
        self.writeReceipt()

    ## @brief Retain modified injection bytes exactly for the parser under test.
    def writeReceipt(self):
        (self.directory / "ic_check.json").write_text(json.dumps(self.value))

    ## @brief Remove only this test's isolated temporary directory.
    def tearDown(self):
        self.temporary.cleanup()

    ## @brief Accept a complete well-formed native artifact set.
    def test_complete_receipt_is_accepted(self):
        result = validator.preflightAudit(self.directory, 8, 1, 2, "float32")
        self.assertEqual(result["value"]["schema"], "ippl-native-ic-check-v1")

    ## @brief A failed subcheck cannot be hidden by a stale top-level passed flag.
    def test_failed_subcheck_rejects_stale_top_level_pass(self):
        self.value["checks"]["native_momentum_relation"] = False
        self.writeReceipt()
        with self.assertRaisesRegex(ValueError, "momentum"):
            validator.preflightAudit(self.directory, 8, 1, 2, "float32")

    ## @brief External 2LPT receipts must not be mistaken for native 1LPT acceptance.
    def test_external_ic_is_not_claimed_native(self):
        self.value["native_1lpt"] = False
        self.writeReceipt()
        with self.assertRaisesRegex(ValueError, "native Gaussian"):
            validator.preflightAudit(self.directory, 8, 1, 2, "float32")

    ## @brief Nonfinite spectrum diagnostics cannot pass finite JSON status metadata.
    def test_nonfinite_spectrum_is_rejected(self):
        path = self.directory / "pk_initial.csv"
        path.write_text(path.read_text().replace("0.05", "nan"))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            validator.preflightAudit(self.directory, 8, 1, 2, "float32")


if __name__ == "__main__":
    unittest.main()
