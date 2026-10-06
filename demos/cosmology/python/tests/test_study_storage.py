## @file test_study_storage.py
# @brief Temporary, mocked execution tests. No MPI, simulation, or previous data use.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
# Tests assert declared invariants using isolated fixtures; no scientific tolerances are relaxed by documentation.
"""Temporary, mocked execution tests. No MPI, simulation, or previous data use."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import study_storage as storage


## @brief Regression suite for Storage.
# @see cosmology_tools
class StorageTests(unittest.TestCase):
    ## @brief Create isolated fixtures and temporary artifact paths for regression checks.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def setUp(self):
        ## @var temporary
        # @brief Retained temporary state owned by this instance; see the initialization and workflow contract.
        self.temporary = tempfile.TemporaryDirectory()
        ## @var directory
        # @brief Artifact directory following this module's ownership/freshness contract.
        self.directory = Path(self.temporary.name)
        ## @var source
        # @brief Retained source state owned by this instance; see the initialization and workflow contract.
        self.source = self.directory / "source.py"
        ## @var executable
        # @brief Selected native/application executable path; its bytes and declared build provenance must match.
        self.executable = self.directory / "adapter"
        ## @var input
        # @brief Retained input state owned by this instance; see the initialization and workflow contract.
        self.input = self.directory / "input.csv"
        for path in (self.source, self.executable, self.input):
            path.write_text("unchanged test provenance\n")
        ## @var provenance
        # @brief Nonempty path-to-hash registry defining the exact source/build/input identity.
        self.provenance = {str(path): storage.sha256(path) for path in (self.source, self.executable)}
        ## @var store
        # @brief Retained store state owned by this instance; see the initialization and workflow contract.
        self.store = storage.StudyStorage(self.directory / "storage", self.provenance, reserve_bytes=0)
        ## @var command
        # @brief Argument-vector command; no shell interpolation is required by the Python launcher.
        self.command = ["mpi-not-executed", "-n", "3", str(self.executable), "2", "4", "10", ".3",
                        ".02", ".2", "8", "2", str(self.input), str(self.store.output_dir("case"))]
        ## @var launches
        # @brief Retained launches state owned by this instance; see the initialization and workflow contract.
        self.launches = []
        ## @var original_frames
        # @brief Retained original frames state owned by this instance; see the initialization and workflow contract.
        self.original_frames = {}
        ## @var free
        # @brief Retained free state owned by this instance; see the initialization and workflow contract.
        self.free = patch.object(storage.shutil, "disk_usage", return_value=SimpleNamespace(free=10**12))
        self.free.start()

    ## @brief Release only the temporary fixtures owned by this test.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def tearDown(self):
        self.store.close()
        self.free.stop()
        self.temporary.cleanup()

    ## @brief Evaluate the fake launch helper in the documented module workflow.
    # @see cosmology_tools
    #
    # @param command Argument-vector command; no shell interpolation is required by the Python launcher.
    # @param log_path Path retaining combined subprocess output and failure evidence.
    # @param timeout Finite positive subprocess timeout in seconds.
    # @param environment Explicit subprocess environment; it does not by itself qualify an execution backend.
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def fake_launch(self, command, log_path, timeout, environment):
        self.launches.append((deepcopy(command), deepcopy(environment)))
        output = Path(command[-1])
        output.mkdir()
        log_path.write_text("mock completed adapter\n")
        (output / "metadata.txt").write_text("ranks=3\nthreads=1\n")
        (output / "checkpoints.csv").write_text("checkpoint,step,a\n0,0,0.02\n1,4,0.063\n2,8,0.2\n")
        (output / "factors.csv").write_text("retained native factor evidence\n")
        for checkpoint in range(3):
            ids = np.arange(8, dtype=np.uint64)
            phase = np.arange(48, dtype=np.float64).reshape(8, 6) / 19 + checkpoint / 7
            phase[0, 0] = np.nextafter(1., 2.)
            phase[1, 1] = -0.
            frame = pd.DataFrame(phase, columns=storage.Columns[1:])
            frame.insert(0, "id", ids)
            self.original_frames[checkpoint] = frame
            for rank, part in enumerate(np.array_split(np.array([7, 0, 6, 1, 5, 2, 4, 3]), 3)):
                frame.iloc[part].to_csv(output / f"particles_checkpoint{checkpoint:04d}_rank{rank}.csv",
                                       index=False, float_format="%.17g")

    ## @brief Execute the documented module workflow.
    # @see cosmology_tools
    #
    # @param kwargs Explicit keyword overrides for an isolated test/configuration fixture.
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
    def execute(self, **kwargs):
        return self.store.execute("case", self.command, particle_grid=2, ranks=3, checkpoints=2, **kwargs)

    ## @brief Evaluate the completed run helper in the documented module workflow.
    # @see cosmology_tools
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
    def completed_run(self):
        with patch.object(self.store, "_launch", side_effect=self.fake_launch):
            return self.execute()

    ## @brief Verify numerical archive exact values ids and original csv provenance.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_numerical_archive_exact_values_ids_and_original_csv_provenance(self):
        record = self.completed_run()
        self.assertEqual(record["state"], "complete")
        self.assertEqual(record["scientific_acceptance"], "not assessed")
        self.assertNotIn("passed", record)
        self.assertEqual(len(record["snapshots"]), 3)
        self.assertEqual(len(self.launches), 1)
        self.assertEqual(self.launches[0][1], storage.FixedEnvironment)
        output = self.store.output_dir("case")
        self.assertFalse(list(output.glob("*.csv")) == [])  # Metadata tables retained.
        self.assertFalse(list(output.glob("particles_*.csv")))
        for checkpoint in range(3):
            frame = self.store.read_snapshot("case", checkpoint)
            expected = self.original_frames[checkpoint]
            self.assertEqual(frame.id.dtype, np.dtype("uint64"))
            self.assertEqual(frame[storage.Columns[1:]].to_numpy().dtype, np.dtype("float64"))
            self.assertEqual(frame[storage.Columns[1:]].to_numpy().tobytes(),
                             expected[storage.Columns[1:]].to_numpy().tobytes())
            entry = record["snapshots"][checkpoint]
            self.assertEqual(sum(source["rows"] for source in entry["csv_sources"]), 8)
            self.assertTrue(all(len(source["sha256"]) == 64 for source in entry["csv_sources"]))
            self.assertEqual(storage.sha256(entry["path"]), entry["sha256"])
        self.assertTrue((output / "factors.csv").exists())
        self.assertTrue((output / "metadata.txt").exists())
        self.assertTrue(Path(record["log_path"]).exists())

    ## @brief Verify completed resume verifies without launch.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_completed_resume_verifies_without_launch(self):
        before = self.completed_run()
        self.store.close()
        self.store = storage.StudyStorage(self.directory / "storage", self.provenance,
                                          reserve_bytes=0, resume=True)
        with patch.object(self.store, "_launch", side_effect=AssertionError("Must not relaunch")):
            after = self.execute()
        self.assertEqual(before, after)

    ## @brief Verify disk block is persistent and resumable not success or failure.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_disk_block_is_persistent_and_resumable_not_success_or_failure(self):
        with (patch.object(storage.shutil, "disk_usage", return_value=SimpleNamespace(free=1)),
              patch.object(self.store, "_launch", side_effect=AssertionError("Must not launch"))):
            with self.assertRaises(storage.DiskSpaceBlocked):
                self.execute()
        run = json.loads(self.store.manifest_path.read_text())["runs"]["case"]
        self.assertEqual(run["state"], "blocked_disk")
        self.assertEqual(run["resume_phase"], "launch")
        self.assertNotIn("storage_complete", run)
        self.assertNotIn("passed", run)
        self.assertFalse(self.store.output_dir("case").exists())
        self.assertEqual(self.completed_run()["state"], "complete")

    ## @brief Verify peak estimate covers all raw archives workspace and reserve.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_peak_estimate_covers_all_raw_archives_workspace_and_reserve(self):
        n, checkpoints = 64, 8
        self.assertGreater(storage.peak_bytes(n, checkpoints), n**3 * (checkpoints + 1) * (180 + 56))
        self.assertEqual(storage.GiB, 1024**3)
        for values in ((0, 8), (2, 0), (2.5, 8)):
            with self.assertRaises(ValueError):
                storage.peak_bytes(*values)

    ## @brief Verify input command environment and executable changes reject resume.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_input_command_environment_and_executable_changes_reject_resume(self):
        self.completed_run()
        original_command = list(self.command)
        for index, changed in ((-9, "8"), (-4, "16"), (0, "other-mpi")):
            self.command = list(original_command)
            self.command[index] = changed
            with self.assertRaisesRegex(ValueError, "descriptor"):
                self.execute()
        self.command = original_command
        with self.assertRaisesRegex(ValueError, "one-thread"):
            self.execute(env={"OMP_NUM_THREADS": "2"})
        with self.assertRaisesRegex(ValueError, "descriptor"):
            self.execute(env={"EXPLICIT_OPTION": "different"})
        self.input.write_text("changed input\n")
        with self.assertRaisesRegex(ValueError, "descriptor"):
            self.execute()

    ## @brief Verify source or archive or metadata tampering is not a completed result.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_source_or_archive_or_metadata_tampering_is_not_a_completed_result(self):
        record = self.completed_run()
        metadata = self.store.output_dir("case") / "metadata.txt"
        before = metadata.read_bytes()
        metadata.write_text("changed metadata\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.execute()
        self.assertEqual(self.store.manifest["runs"]["case"]["state"], "integrity_failed")
        self.assertFalse(self.store.manifest["runs"]["case"]["storage_complete"])
        metadata.write_bytes(before)
        with patch.object(self.store, "_launch", side_effect=AssertionError("No relaunch")):
            self.assertEqual(self.execute()["state"], "complete")
        Path(record["snapshots"][0]["path"]).write_bytes(b"bad archive")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.execute()
        self.source.write_text("changed source\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.execute()

    ## @brief Verify unrelated data never deleted and run paths are bounded.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_unrelated_data_never_deleted_and_run_paths_are_bounded(self):
        outside = self.directory / "unrelated.csv"
        outside.write_text("user data\n")
        record = {"csv_sources": [{"path": str(outside), "sha256": storage.sha256(outside)}]}
        with self.assertRaisesRegex(ValueError, "unowned"):
            self.store._remove_recorded_csvs(record, self.store.output_dir("case"))
        self.assertEqual(outside.read_text(), "user data\n")
        for name in ("../escape", "a/b", "", ".", "/tmp/foo"):
            with self.assertRaises(ValueError):
                self.store.output_dir(name)
        self.store.output_dir("case").mkdir()
        with self.assertRaisesRegex(ValueError, "Unowned"):
            self.execute()

    ## @brief Verify failed execution is retained and never automatically rerun.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_failed_execution_is_retained_and_never_automatically_rerun(self):
        def fail(*arguments):
            self.fake_launch(*arguments)
            raise subprocess.CalledProcessError(2, self.command)
        with patch.object(self.store, "_launch", side_effect=fail):
            with self.assertRaises(subprocess.CalledProcessError):
                self.execute()
        self.assertEqual(self.store.manifest["runs"]["case"]["state"], "execution_failed")
        self.assertEqual(len(list(self.store.output_dir("case").glob("particles_*.csv"))), 9)
        with patch.object(self.store, "_launch", side_effect=AssertionError("No rerun")):
            with self.assertRaisesRegex(RuntimeError, "Incomplete execution"):
                self.execute()

    ## @brief Verify interrupted archive resumes without relaunch.
    # @see cosmology_tools
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
    def test_interrupted_archive_resumes_without_relaunch(self):
        original_remove = self.store._remove_recorded_csvs
        calls = []
        def interrupted(record, output):
            calls.append(record["checkpoint"])
            if len(calls) == 1:
                raise RuntimeError("Simulated interruption after committed archive")
            return original_remove(record, output)
        with (patch.object(self.store, "_launch", side_effect=self.fake_launch),
              patch.object(self.store, "_remove_recorded_csvs", side_effect=interrupted)):
            with self.assertRaisesRegex(RuntimeError, "Simulated interruption"):
                self.execute()
        self.assertEqual(self.store.manifest["runs"]["case"]["state"], "archive_failed")
        self.assertTrue((self.store.output_dir("case") / "particles_checkpoint0000.npz").exists())
        self.store.close()
        self.store = storage.StudyStorage(self.directory / "storage", self.provenance, reserve_bytes=0, resume=True)
        with patch.object(self.store, "_launch", side_effect=AssertionError("No rerun")):
            self.assertEqual(self.execute()["state"], "complete")
        self.assertFalse(list(self.store.output_dir("case").glob("particles_*.csv")))

    ## @brief Verify partial archive write resumes preserving csv until verified.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_partial_archive_write_resumes_preserving_csv_until_verified(self):
        def partial_write(stream, **arrays):
            stream.write(b"incomplete npz")
            raise OSError("Simulated interrupted write")
        with (patch.object(self.store, "_launch", side_effect=self.fake_launch),
              patch.object(storage.np, "savez_compressed", side_effect=partial_write)):
            with self.assertRaises(OSError):
                self.execute()
        output = self.store.output_dir("case")
        self.assertEqual(len(list(output.glob("particles_*.csv"))), 9)
        self.assertTrue((output / "particles_checkpoint0000.npz.partial").exists())
        with patch.object(self.store, "_launch", side_effect=AssertionError("No rerun")):
            self.assertEqual(self.execute()["state"], "complete")
        self.assertFalse(list(output.glob("*.partial")))

    ## @brief Verify changed csv after commit is retained and rejected.
    # @see cosmology_tools
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
    def test_changed_csv_after_commit_is_retained_and_rejected(self):
        original_remove = self.store._remove_recorded_csvs
        def changed(record, output):
            path = Path(record["csv_sources"][0]["path"])
            path.write_text(path.read_text() + "\n")
            return original_remove(record, output)
        with (patch.object(self.store, "_launch", side_effect=self.fake_launch),
              patch.object(self.store, "_remove_recorded_csvs", side_effect=changed)):
            with self.assertRaisesRegex(ValueError, "CSV changed"):
                self.execute()
        self.assertEqual(len(list(self.store.output_dir("case").glob("particles_*.csv"))), 9)

    ## @brief Verify invalid ids or missing rank keep csv and block archive.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_invalid_ids_or_missing_rank_keep_csv_and_block_archive(self):
        def bad(*arguments):
            self.fake_launch(*arguments)
            path = self.store.output_dir("case") / "particles_checkpoint0000_rank0.csv"
            frame = pd.read_csv(path)
            frame.loc[0, "id"] = frame.loc[1, "id"]
            frame.to_csv(path, index=False)
        with patch.object(self.store, "_launch", side_effect=bad):
            with self.assertRaisesRegex(ValueError, "sorted complete IDs"):
                self.execute()
        self.assertEqual(len(list(self.store.output_dir("case").glob("particles_*.csv"))), 9)
        missing = self.store.output_dir("case") / "particles_checkpoint0000_rank2.csv"
        missing.unlink()  # This temporary test explicitly owns the fixture.
        with self.assertRaisesRegex(ValueError, "rank CSV set"):
            self.execute()

    ## @brief Verify second writer and nonempty new storage rejected.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_second_writer_and_nonempty_new_storage_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Another process"):
            storage.StudyStorage(self.store.root, self.provenance, resume=True)
        with self.assertRaisesRegex(ValueError, "must be empty"):
            storage.StudyStorage(self.store.root, self.provenance)

    ## @brief Verify empty rank archives with complete global ids.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_empty_rank_archives_with_complete_global_ids(self):
        def empty_rank(*arguments):
            self.fake_launch(*arguments)
            output = self.store.output_dir("case")
            for checkpoint, frame in self.original_frames.items():
                for rank, indices in enumerate(([], [7, 1, 0], [2, 3, 4, 5, 6])):
                    frame.iloc[indices].to_csv(output / f"particles_checkpoint{checkpoint:04d}_rank{rank}.csv",
                                              index=False, float_format="%.17g")
        with patch.object(self.store, "_launch", side_effect=empty_rank):
            record = self.execute()
        for snapshot in record["snapshots"]:
            self.assertEqual([source["rows"] for source in snapshot["csv_sources"]], [0, 3, 5])
            np.testing.assert_array_equal(self.store.read_snapshot("case", snapshot["checkpoint"]).id,
                                          np.arange(8, dtype=np.uint64))

    ## @brief Verify disk block during archival resumes without reexecution.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_disk_block_during_archival_resumes_without_reexecution(self):
        values = iter([storage.peak_bytes(2, 2) + 1, 1])
        with (patch.object(storage.shutil, "disk_usage", side_effect=lambda _: SimpleNamespace(free=next(values))),
              patch.object(self.store, "_launch", side_effect=self.fake_launch)):
            with self.assertRaises(storage.DiskSpaceBlocked):
                self.execute()
        run = self.store.manifest["runs"]["case"]
        self.assertEqual(run["state"], "blocked_disk")
        self.assertEqual(run["resume_phase"], "archive")
        self.assertEqual(len(list(self.store.output_dir("case").glob("particles_*.csv"))), 9)
        with patch.object(self.store, "_launch", side_effect=AssertionError("No reexecution")):
            self.assertEqual(self.execute()["state"], "complete")

    ## @brief Verify unknown state never launches.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_unknown_state_never_launches(self):
        with patch.object(storage.shutil, "disk_usage", return_value=SimpleNamespace(free=1)):
            with self.assertRaises(storage.DiskSpaceBlocked):
                self.execute()
        self.store.manifest["runs"]["case"]["state"] = "unrecognized"
        with patch.object(self.store, "_launch", side_effect=AssertionError("No launch")):
            with self.assertRaisesRegex(ValueError, "Unknown persisted"):
                self.execute()


## @cond CLI_DISPATCH
if __name__ == "__main__":
    unittest.main()
## @endcond
