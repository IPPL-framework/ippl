## @file test_evolution_refinement.py
# @brief Synthetic provenance and numerical-analysis tests; no MPI or simulations.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
# @see cosmology_model cosmology_numerics cosmology_spectra
# Tests assert declared invariants using isolated fixtures; no scientific tolerances are relaxed by documentation.
"""Synthetic provenance and numerical-analysis tests; no MPI or simulations."""
from copy import deepcopy
from argparse import Namespace
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_evolution_refinement as followup

## @var original
# @brief Original shared input or retained reference record; it is not modified to fit the comparison.
original = followup.original


## @brief Read the snapshot for the documented module workflow.
# @see cosmology_tools
#
# @param offset Prescribed synthetic perturbation used to exercise a comparison/refinement gate.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def snapshot(offset):
    frame = pd.DataFrame({"id": np.arange(8, dtype=np.uint64),
                          "x": np.arange(8, dtype=float) + offset, "y": 1., "z": 1.,
                          "px": 1. + offset, "py": 0., "pz": 0.})
    return frame


## @brief Evaluate the spectrum record helper in the documented module workflow.
# @see cosmology_tools
#
# @param checkpoint Synchronized saved epoch index, with zero denoting imported initial state.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def spectrum_record(checkpoint=4):
    a = original.Parameters["a_initial"] * np.exp(checkpoint * np.log(
        original.Parameters["a_final"] / original.Parameters["a_initial"]) / 8)
    n = len(original.resolved_modes("coupled3d"))
    return {"checkpoint": checkpoint, "a": float(a), "real": [.1] * n, "imag": [.2] * n}


## @brief Regression suite for Refinement.
# @see cosmology_tools
class RefinementTests(unittest.TestCase):
    ## @brief Verify three level differences use finer momentum reference.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_three_level_differences_use_finer_momentum_reference(self):
        frames = [snapshot(offset) for offset in (.008, .002, .0005)]
        spectra = [np.array([1. + offset + .2j]) for offset in (.008, .002, .0005)]
        row, checks = followup.assess_refinement("ippl", 8, frames, spectra, original.Parameters)
        self.assertEqual(row["steps"], [128, 256, 512])
        self.assertAlmostEqual(row["phase_space_differences"][0]["momentum_relative"], .006 / 1.002)
        self.assertAlmostEqual(row["phase_space_differences"][1]["momentum_relative"], .0015 / 1.0005)
        self.assertAlmostEqual(row["orders"]["position_cells"]["ratio"], 4.)
        self.assertTrue(all(check["passed"] for check in checks))
        self.assertEqual(len(checks), 5)
        self.assertEqual(checks[-1]["limit"], original.Limits["time_finest_momentum_relative"])
        self.assertIn("not errors", row["interpretation"])

    ## @brief Verify finest budget is not waived by second order ratio.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_finest_budget_is_not_waived_by_second_order_ratio(self):
        frames = [snapshot(offset) for offset in (.08, .02, .005)]
        spectra = [np.array([1. + offset + .2j]) for offset in (.08, .02, .005)]
        _, checks = followup.assess_refinement("fastpm", 8, frames, spectra, original.Parameters)
        self.assertTrue(all(check["passed"] for check in checks[:3]))
        self.assertFalse(checks[-1]["passed"])
        self.assertEqual(checks[-1]["limit"], .002)

    ## @brief Verify precross upper ratio and postcross lower only.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_precross_upper_ratio_and_postcross_lower_only(self):
        frames = [snapshot(offset) for offset in (.1, .001, 0.)]
        spectra = [np.array([1. + offset + .2j]) for offset in (.1, .001, 0.)]
        _, early = followup.assess_refinement("ippl", 4, frames, spectra, original.Parameters)
        _, late = followup.assess_refinement("ippl", 8, frames, spectra, original.Parameters)
        self.assertEqual(len(early), 3)
        self.assertTrue(all(not check["passed"] for check in early))
        self.assertTrue(all(check["passed"] for check in late))

    ## @brief Verify original precision floors retained.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_original_precision_floors_retained(self):
        frames = [snapshot(offset) for offset in (3e-9, 2e-9, 1e-9)]
        spectra = [np.array([1. + offset + .2j]) for offset in (3e-9, 2e-9, 1e-9)]
        _, checks = followup.assess_refinement("ippl", 4, frames, spectra, original.Parameters)
        for check, quantity in zip(checks, ("position_cells", "momentum_relative", "complex_relative")):
            self.assertTrue(check["passed"])
            self.assertEqual(check["status"], "precision_limited")
            self.assertEqual(check["precision_floor"], original.Limits["time_precision_" + quantity])

    ## @brief Verify incomplete refinement and missing signal rejected.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_incomplete_refinement_and_missing_signal_rejected(self):
        frames = [snapshot(0)] * 3
        spectra = [np.array([1j])] * 3
        with self.assertRaises(ValueError):
            followup.assess_refinement("ippl", 8, frames[:2], spectra, original.Parameters)
        with self.assertRaises(ValueError):
            followup.assess_refinement("ippl", 7, frames, spectra, original.Parameters)
        with self.assertRaises((ValueError, TypeError)):
            followup.assess_refinement("ippl", 8, frames, [np.array([0j])] * 3, original.Parameters)

    ## @brief Verify hash mismatch rejected.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source"
            path.write_bytes(b"original")
            hashes = {str(path): original.sha256(path)}
            followup.verify_hashes(hashes)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                followup.verify_hashes(hashes)
        with self.assertRaises(ValueError):
            followup.verify_hashes({})

    ## @brief Verify both compressed and content hashes required.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_both_compressed_and_content_hashes_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.csv.gz"
            content = b"id,x,y,z,px,py,pz\n0,0,0,0,1,0,0\n"
            path.write_bytes(gzip.compress(content, mtime=0))
            record = {"path": str(path), "sha256": original.sha256(path),
                      "csv_sha256": hashlib.sha256(content).hexdigest(),
                      "csv_bytes": len(content), "compressed_bytes": path.stat().st_size}
            followup.verify_archive(record)
            bad = dict(record, csv_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "content mismatch"):
                followup.verify_archive(bad)
            bad = dict(record, csv_bytes=len(content) + 1)
            with self.assertRaises(ValueError):
                followup.verify_archive(bad)
            path.write_bytes(gzip.compress(content + b"\n", mtime=0))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                followup.verify_archive(record)

    ## @brief Verify saved fourier data validate order length epoch and finiteness.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_saved_fourier_data_validate_order_length_epoch_and_finiteness(self):
        run = {"density_modes": [spectrum_record()]}
        result = followup.saved_spectrum(run, 4, original.Parameters)
        np.testing.assert_array_equal(result, [.1 + .2j] * len(original.resolved_modes("coupled3d")))
        bad_records = []
        for key, value in (("real", [.1]), ("imag", [float("nan")] * len(result)), ("a", .1)):
            entry = spectrum_record()
            entry[key] = value
            bad_records.append([entry])
        bad_records.extend([[], [spectrum_record(), spectrum_record()]])
        for records in bad_records:
            with self.subTest(records=len(records)):
                with self.assertRaises(ValueError):
                    followup.saved_spectrum({"density_modes": records}, 4, original.Parameters)

    ## @brief Verify exact shared input read not regenerated or rewritten.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_exact_shared_input_read_not_regenerated_or_rewritten(self):
        frame = snapshot(0)
        frame["px"] = float(np.float32(.1))
        frame["mass"] = 1
        frame = frame.sample(frac=1, random_state=17)
        parameters = dict(original.Parameters, particle_grid=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            frame.to_csv(path, index=False, float_format="%.17g")
            before = original.sha256(path)
            with patch.object(original, "make_fixture", side_effect=AssertionError("Must not regenerate input")):
                actual = followup.load_input(path, parameters)
            self.assertEqual(original.sha256(path), before)
            pd.testing.assert_frame_equal(actual, frame.sort_values("id").reset_index(drop=True),
                                          check_dtype=False, check_exact=True)
            frame["px"] = .1  # Not the exact widened float32 value.
            frame.to_csv(path, index=False, float_format="%.17g")
            with self.assertRaisesRegex(ValueError, "shared-momentum"):
                followup.load_input(path, parameters)

    ## @brief Verify modified limits or parent failure set rejected without launch.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_modified_limits_or_parent_failure_set_rejected_without_launch(self):
        failures = [{"name": f"coupled3d/{code}/8/time/finest_momentum", "passed": False,
                     "value": .0027, "limit": .002} for code in followup.Codes]
        parent = {"schema": "ippl-fastpm-evolution-v1", "complete": True, "quick": False,
                  "passed": False, "parameters": deepcopy(original.Parameters),
                  "limits": followup.canonical(original.Limits), "checks": deepcopy(failures),
                  "failed_checks": deepcopy(failures)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            changed = deepcopy(parent)
            changed["limits"]["time_finest_momentum_relative"] = .003
            path.write_text(json.dumps(changed))
            with patch.object(original, "launch", side_effect=AssertionError("Must not launch")):
                with self.assertRaisesRegex(ValueError, "unchanged parameters and limits"):
                    followup.inspect_parent(path)
                changed = deepcopy(parent)
                changed["failed_checks"] = []
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "failure set"):
                    followup.inspect_parent(path)
            self.assertEqual(parent["failed_checks"], failures)

    ## @brief Exercise orchestration with synthetic snapshots, never a subprocess.
    # @see cosmology_tools
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
    def test_followup_launch_set_is_exactly_two_and_parent_failure_is_retained(self):
        """Exercise orchestration with synthetic snapshots, never a subprocess."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preserved = root / "preserved"
            preserved.mkdir()
            input_path = preserved / "coupled3d-initial.csv"
            snapshot(0).assign(mass=1).to_csv(input_path, index=False)
            parent_path = preserved / "results.json"
            parent_path.write_text("{}\n")
            executables = {code: root / (code + "-exe") for code in followup.Codes}
            manifest = root / "build-manifest.txt"
            for path in [*executables.values(), manifest]:
                path.write_bytes(b"synthetic provenance only\n")
            fixture = {"path": str(input_path), "sha256": original.sha256(input_path)}
            failures = [{"name": f"coupled3d/{code}/8/time/finest_momentum", "passed": False,
                         "value": .0027, "limit": .002} for code in followup.Codes]
            parent = {"path": parent_path, "sha256": original.sha256(parent_path),
                      "executables": executables, "manifest": manifest, "artifacts": {},
                      "fixture": snapshot(0).assign(mass=1), "runs": {}, "spectra": {},
                      "report": {"hashes_before": {str(input_path): fixture["sha256"]},
                                 "complete": True, "passed": False, "failed_checks": deepcopy(failures),
                                 "fixtures": {"coupled3d": fixture}}}
            for steps, offset in ((128, .008), (256, .002)):
                case = original.Case("coupled3d", 32, steps)
                for code in followup.Codes:
                    parent["runs"][(case, code)] = {"name": case.name + "_" + code,
                        **case.__dict__, "code": code, "output": str(preserved / (case.name + "_" + code)),
                        "input_sha256": fixture["sha256"]}
                    for checkpoint in followup.Checkpoints:
                        parent["spectra"][(case, code, checkpoint)] = np.full(
                            len(original.resolved_modes("coupled3d")), 1. + offset + .2j)
            args = Namespace(ippl_exe=None, fastpm_exe=None, fastpm_manifest=None,
                             output_dir=root / "new", mpiexec="forbidden", mpi_arg=[],
                             numproc_flag="-n", timeout=1)
            campaign = followup.RefinementCampaign(args, parent)
            requested = []
            def fake_run_case(case, code):
                requested.append((case, code))
                campaign.report["runs"].append({"name": case.name + "_" + code})
                campaign.outputs[(case, code)] = campaign.root / (case.name + "_" + code)
                for checkpoint in range(9):
                    campaign.spectra[(case, code, checkpoint)] = np.full(
                        len(original.resolved_modes("coupled3d")), 1.0005 + .2j)
            def fake_read(directory, checkpoint, ranks, particle_grid):
                offset = .008 if "_t128_" in directory else .002 if "_t256_" in directory else .0005
                return snapshot(offset)
            with (patch.object(campaign, "run_case", side_effect=fake_run_case),
                  patch.object(original, "read_snapshot", side_effect=fake_read),
                  patch.object(original.Campaign, "prepare", side_effect=AssertionError("No prepare")),
                  patch.object(original, "launch", side_effect=AssertionError("No launch"))):
                self.assertEqual(campaign.run(), 0)
            self.assertEqual(requested, [(original.Case("coupled3d", 32, 512), code)
                                         for code in followup.Codes])
            self.assertTrue(campaign.report["passed"])
            self.assertTrue(campaign.report["complete"])
            self.assertFalse(campaign.report["parent"]["passed"])
            self.assertFalse(campaign.report["parent"]["failures_superseded"])
            self.assertEqual(campaign.report["parent"]["failed_checks"], failures)
            self.assertEqual(parent["report"]["failed_checks"], failures)
            self.assertEqual(campaign.report["fixtures"]["coupled3d"]["path"], str(input_path))
            self.assertEqual(original.sha256(parent_path), parent["sha256"])


## @cond CLI_DISPATCH
if __name__ == "__main__":
    unittest.main()
## @endcond
