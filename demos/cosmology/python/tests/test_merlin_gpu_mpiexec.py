## @file test_merlin_gpu_mpiexec.py
# @brief Pure/mocked launcher checks. No MPI, GPU, Slurm command, or network call.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
# Tests assert declared invariants using isolated fixtures; no scientific tolerances are relaxed by documentation.
"""Pure/mocked launcher checks. No MPI, GPU, Slurm command, or network call."""

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "merlin"))
import gpu_mpiexec as launcher


## @var Slurm
# @brief Named Slurm protocol/schema value; the source initializer records its exact contents.
Slurm = {"SLURM_JOB_ID": "12345", "SLURM_JOB_NUM_NODES": "1", "SLURM_NNODES": "1",
         "SLURM_CPUS_ON_NODE": "4", "SLURM_NTASKS": "4", "SLURM_CPUS_PER_TASK": "1"}


## @brief Regression suite for Launcher.
# @see cosmology_tools
class LauncherTests(unittest.TestCase):
    ## @brief Create isolated fixtures and temporary artifact paths for regression checks.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def setUp(self):
        ## @var temporary
        # @brief Retained temporary state owned by this instance; see the initialization and workflow contract.
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        ## @var root
        # @brief Campaign or artifact root following this module's ownership contract.
        self.root = Path(self.temporary.name).resolve()
        ## @var files
        # @brief Retained files state owned by this instance; see the initialization and workflow contract.
        self.files = {}
        for name in ("mpiexec", "python", "gpu_rank.py", "Cosmology", "CompareCosmologyForce",
                     "CompareCosmologyEvolution", "FastPMForce", "FastPMEvolution", "provenance.txt"):
            path = self.root / name
            path.write_text("mock artifact " + name)
            path.chmod(0o755)
            self.files[name] = str(path)
        ## @var config
        # @brief Explicit validated launcher/campaign configuration; machine identity and hashes are checked separately.
        self.config = {"mpiexec": self.files["mpiexec"], "python": self.files["python"],
            "gpu_helper": self.files["gpu_rank.py"], "evidence_dir": str(self.root / "evidence"),
            "gpu_executables": [self.files[name] for name in sorted(launcher.GpuNames)],
            "cpu_executables": [self.files[name] for name in sorted(launcher.CpuNames)],
            "sha256": {path: launcher.sha256(path) for path in self.files.values()}}
        ## @var allocationPath
        # @brief Retained allocationPath state owned by this instance; see the initialization and workflow contract.
        self.allocationPath = self.root / "run-allocated-gpus.csv"
        self.allocationPath.write_text("".join(f"NVIDIA A100-SXM4-40GB, GPU-{token}, 00000000:{rank+1:02x}:00.0, Disabled\n"
            for rank, token in enumerate(("aaaa", "bbbb", "cccc", "dddd"))))
        self.config["allocation_evidence"] = str(self.allocationPath)
        ## @var configPath
        # @brief Absolute strict-launcher configuration JSON path.
        self.configPath = self.root / "config.json"
        self.write_config()

    ## @brief Write config.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def write_config(self):
        self.configPath.write_text(json.dumps(self.config))

    ## @brief Evaluate the binding helper in the documented module workflow.
    # @see cosmology_tools
    #
    # @param rank MPI rank or local-rank integer specified by the launcher/test.
    # @param ranks Positive MPI rank count; all expected snapshot shards must exist.
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
    def binding(self, rank, ranks=2):
        return {"schema": "ippl-gpu-binding-v1", "host": "merlin-test-node", "rank": rank,
            "local_rank": rank, "local_size": ranks, "world_size": ranks,
            "runtime_device_count": 1, "visible_device_ordinal": 0,
            "allocated_tokens": ["GPU-aaaa", "GPU-bbbb", "GPU-cccc", "GPU-dddd"],
            "visible_token": ["GPU-aaaa", "GPU-bbbb", "GPU-cccc", "GPU-dddd"][rank],
            "pci": f"0000:{rank+1:02x}:00.0", "executable": self.files["Cosmology"],
            "requested_executable": self.files["Cosmology"], "helper_path": self.files["gpu_rank.py"],
            "helper_sha256": self.config["sha256"][self.files["gpu_rank.py"]]}

    ## @brief Validate the documented module workflow.
    # @see cosmology_tools
    #
    # @param records Retained evidence records under the calling validator's schema and ordering.
    # @param ranks Positive MPI rank count; all expected snapshot shards must exist.
    # @param kind Fixture selector from the module's declared supported cases.
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def validate(self, records, ranks=2, kind="gpu"):
        launcher.validate_bindings(records, ranks, kind, self.config,
                                   self.files["Cosmology"], self.files["Cosmology"],
                                   launcher.allocation_evidence(self.allocationPath))

    ## @brief Evaluate the mocked launch helper in the documented module workflow.
    # @see cosmology_tools
    #
    # @param records Retained evidence records under the calling validator's schema and ordering.
    # @param code Solver identifier (IPPL, native FastPM or GADGET) selected by the protocol.
    # @param target Selected executable or runtime binding target as defined by this launcher/test.
    # @param extra Explicit additional test/command arguments; no hidden runtime options are inferred.
    # @param arguments Parsed command-line options; see main/--help and the module workflow contract.
    # @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
    def mocked_launch(self, records, *, code=0, target="Cosmology", extra=b"", arguments=None):
        raw = b"ordinary merged output\n" + b"".join(
            b"GPU_BINDING " + json.dumps(row).encode() + b"\n" for row in records) + extra
        process = Mock(stdout=io.BytesIO(raw), returncode=code)
        process.wait.return_value = code
        process.poll.return_value = code
        arguments = arguments or [self.files[target], "--a", "path with spaces", "--config", "target-owned", "-n", "99"]
        capture = io.StringIO()
        with (patch.dict(os.environ, Slurm, clear=True), patch.object(launcher.subprocess, "Popen", return_value=process) as popen,
              redirect_stdout(capture)):
            result = launcher.launch(self.configPath, 2, arguments)
        manifests = [json.loads(path.read_text()) for path in sorted((self.root/"evidence").glob("*.json"))]
        return result, manifests[-1], popen, capture.getvalue(), raw

    ## @brief Verify prefix order equals and target arguments remain untouched.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_prefix_order_equals_and_target_arguments_remain_untouched(self):
        target = [self.files["Cosmology"], "--config", "/target-config", "-n", "99", "space value"]
        for prefix in (["--config", str(self.configPath), "-n", "2"],
                       ["-n", "2", "--config="+str(self.configPath)]):
            path, ranks, remaining = launcher.parse_arguments(prefix+target)
            self.assertEqual(path, self.configPath)
            self.assertEqual(ranks, 2)
            self.assertEqual(remaining, target)

    ## @brief Verify reject arbitrary mpi flags invalid ranks and relative paths.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_reject_arbitrary_mpi_flags_invalid_ranks_and_relative_paths(self):
        prefix = ["--config", str(self.configPath)]
        target = [self.files["Cosmology"]]
        invalid = (["-n", "0"], ["-n", "5"], ["-n", "02"], ["-n", "-1"],
                   ["--oversubscribe", "-n", "1"], ["-n", "1", "-x", "SECRET"],
                   ["-n", "1", "-n", "1"], ["-n", "1", "--config=/another"])
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(launcher.LaunchError):
                launcher.parse_arguments(prefix+options+target)
        for args in (["--config=relative", "-n", "1", *target],
                     [*prefix, "-n", "1", "relative-target"], prefix):
            with self.assertRaises(launcher.LaunchError):
                launcher.parse_arguments(args)

    ## @brief Verify hash coverage allows extra provenance and rejects mutation.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_hash_coverage_allows_extra_provenance_and_rejects_mutation(self):
        config, target, kind, digest = launcher.load_config(self.configPath, [self.files["Cosmology"]])
        self.assertEqual((target, kind), (self.files["Cosmology"], "gpu"))
        self.assertEqual(digest, launcher.sha256(self.configPath))
        self.assertIn(self.files["provenance.txt"], config["sha256"])
        for artifact in ("mpiexec", "python", "gpu_rank.py", "Cosmology", "FastPMEvolution"):
            missing = deepcopy(self.config)
            del missing["sha256"][self.files[artifact]]
            self.configPath.write_text(json.dumps(missing))
            with self.subTest(artifact=artifact), self.assertRaises(launcher.LaunchError):
                launcher.load_config(self.configPath, [self.files["Cosmology"]])
        self.write_config()
        Path(self.files["FastPMForce"]).write_text("changed even though not selected")
        with self.assertRaisesRegex(launcher.LaunchError, "SHA256"):
            launcher.load_config(self.configPath, [self.files["Cosmology"]])

    ## @brief Verify allowlists and absolute paths are enforced.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_allowlists_and_absolute_paths_are_enforced(self):
        with self.assertRaisesRegex(launcher.LaunchError, "allowlisted"):
            launcher.load_config(self.configPath, [self.files["provenance.txt"]])
        self.config["gpu_executables"].append(self.files["FastPMForce"])
        self.write_config()
        with self.assertRaises(launcher.LaunchError):
            launcher.load_config(self.configPath, [self.files["Cosmology"]])
        self.config["gpu_executables"].pop()
        self.config["evidence_dir"] = "relative-evidence"
        self.write_config()
        with self.assertRaises(launcher.LaunchError):
            launcher.load_config(self.configPath, [self.files["Cosmology"]])

    ## @brief Verify symlinked venv python path is preserved but hashes resolve.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_symlinked_venv_python_path_is_preserved_but_hashes_resolve(self):
        alias = self.root / "venv-python"
        alias.symlink_to(self.files["python"])
        self.config["python"] = str(alias)
        digest = self.config["sha256"].pop(self.files["python"])
        self.config["sha256"][str(alias)] = digest
        self.write_config()
        config, _, _, _ = launcher.load_config(self.configPath, [self.files["Cosmology"]])
        command = launcher.build_command(config, 2, "gpu", [self.files["Cosmology"]])
        self.assertEqual(command[7], str(alias))
        self.assertEqual(config["sha256"][self.files["python"]], digest)
        self.assertNotIn(str(alias), config["sha256"])
        self.config["sha256"][self.files["python"]] = "0"*64
        self.write_config()
        with self.assertRaisesRegex(launcher.LaunchError, "Conflicting"):
            launcher.load_config(self.configPath, [self.files["Cosmology"]])

    ## @brief Verify launcher has executable mode.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_launcher_has_executable_mode(self):
        self.assertEqual(Path(launcher.__file__).stat().st_mode & 0o777, 0o755)

    ## @brief Verify exact slurm cpu and node cap.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_exact_slurm_cpu_and_node_cap(self):
        self.assertEqual(launcher.allocation(Slurm), Slurm)
        for key, value in (("SLURM_CPUS_ON_NODE", "8"), ("SLURM_NTASKS", "1"),
                           ("SLURM_CPUS_PER_TASK", "4"), ("SLURM_NNODES", "2"),
                           ("SLURM_JOB_ID", "")):
            with self.subTest(key=key), self.assertRaises(launcher.LaunchError):
                launcher.allocation({**Slurm, key: value})
        with self.assertRaises(launcher.LaunchError):
            launcher.allocation({})

    ## @brief Verify allocation evidence rejects wrong model mig duplicates and count.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_allocation_evidence_rejects_wrong_model_mig_duplicates_and_count(self):
        original = self.allocationPath.read_text()
        valid = launcher.allocation_evidence(self.allocationPath)
        self.assertEqual(len(valid["devices"]), 4)
        self.assertEqual(valid["sha256"], launcher.sha256(self.allocationPath))
        variants = (original.replace("A100", "A1000"), original.replace("Disabled", "Enabled", 1),
                    original.replace("GPU-bbbb", "GPU-aaaa"), original.replace("00000000:02", "00000000:01"),
                    "\n".join(original.splitlines()[:3])+"\n", "name,uuid,pci,mig\n"+original)
        for contents in variants:
            self.allocationPath.write_text(contents)
            with self.subTest(contents=contents[:40]), self.assertRaises(launcher.LaunchError):
                launcher.allocation_evidence(self.allocationPath)

    ## @brief Verify observed pci must be in preflight allocation.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_observed_pci_must_be_in_preflight_allocation(self):
        records = [self.binding(rank) for rank in range(2)]
        records[1]["pci"] = "0000:05:00.0"
        with self.assertRaisesRegex(launcher.LaunchError, "outside"):
            self.validate(records)

    ## @brief Verify fixed command and gpu helper cpu direct dispatch.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_fixed_command_and_gpu_helper_cpu_direct_dispatch(self):
        arguments = [self.files["Cosmology"], "8", "filename with spaces", "--config=target-option"]
        prefix = [self.files["mpiexec"], "--bind-to", "none", "--map-by", "slot", "-n", "2"]
        self.assertEqual(launcher.build_command(self.config, 2, "gpu", arguments),
                         prefix+[self.files["python"], "-B", self.files["gpu_rank.py"]]+arguments)
        self.assertEqual(launcher.build_command(self.config, 2, "cpu", arguments), prefix+arguments)

    ## @brief Verify two host threads preserved only for one gpu rank.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_two_host_threads_preserved_only_for_one_gpu_rank(self):
        self.assertEqual(launcher.thread_environment({"OMP_NUM_THREADS": "2"}, 1, "gpu")["OMP_NUM_THREADS"], "2")
        self.assertEqual(launcher.thread_environment({"OMP_NUM_THREADS": "2"}, 4, "cpu")["OMP_NUM_THREADS"], "1")
        for ranks in (2, 4):
            with self.subTest(ranks=ranks), self.assertRaises(launcher.LaunchError):
                launcher.thread_environment({"OMP_NUM_THREADS": "2"}, ranks, "gpu")
        with self.assertRaises(launcher.LaunchError):
            launcher.thread_environment({"OMP_NUM_THREADS": "8"}, 1, "gpu")
        raw = b"GPU_BINDING " + json.dumps(self.binding(0, 1)).encode() + b"\n"
        process = Mock(stdout=io.BytesIO(raw), returncode=0)
        process.wait.return_value = process.poll.return_value = 0
        with (patch.dict(os.environ, {**Slurm, "OMP_NUM_THREADS": "2"}, clear=True),
              patch.object(launcher.subprocess, "Popen", return_value=process) as popen,
              redirect_stdout(io.StringIO())):
            self.assertEqual(launcher.launch(self.configPath, 1, [self.files["Cosmology"]]), 0)
        self.assertEqual(popen.call_args.kwargs["env"]["OMP_NUM_THREADS"], "2")
        manifest = json.loads(next((self.root/"evidence").glob("*.json")).read_text())
        self.assertEqual(manifest["thread_environment"]["OMP_NUM_THREADS"], "2")

    ## @brief Verify one to four distinct physical gpu bindings.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_one_to_four_distinct_physical_gpu_bindings(self):
        for ranks in range(1, 5):
            with self.subTest(ranks=ranks):
                self.validate([self.binding(rank, ranks) for rank in reversed(range(ranks))], ranks)

    ## @brief Verify binding rejects missing duplicate host pci mig and wrong artifacts.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_binding_rejects_missing_duplicate_host_pci_mig_and_wrong_artifacts(self):
        records = [self.binding(rank) for rank in range(2)]
        for broken in (records[:1], records+records[:1]):
            with self.assertRaises(launcher.LaunchError):
                self.validate(broken)
        for key, value in (("rank", 0), ("local_rank", 0), ("rank", True), ("host", "another-node"),
                           ("pci", "00000000:01:00.0"), ("pci", "bad"), ("world_size", 3),
                           ("runtime_device_count", 2), ("visible_device_ordinal", 1),
                           ("helper_sha256", "0"*64), ("executable", self.files["FastPMForce"])):
            broken = deepcopy(records)
            broken[1][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(launcher.LaunchError):
                self.validate(broken)
        broken = deepcopy(records)
        for row in broken:
            row["allocated_tokens"][0] = "MIG-abcd"
        broken[0]["visible_token"] = "MIG-abcd"
        with self.assertRaises(launcher.LaunchError):
            self.validate(broken)
        self.validate([], kind="cpu")
        with self.assertRaises(launcher.LaunchError):
            self.validate(records, kind="cpu")

    ## @brief Verify success tees exact bytes and retains process group and manifest.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_success_tees_exact_bytes_and_retains_process_group_and_manifest(self):
        result, record, popen, output, raw = self.mocked_launch([self.binding(rank) for rank in range(2)])
        self.assertEqual(result, 0)
        self.assertEqual(record["status"], "launch_complete")
        self.assertEqual(record["scientific_acceptance"], "not assessed")
        self.assertEqual(Path(record["log"]).read_bytes(), raw)
        self.assertEqual(output.encode(), raw)
        self.assertEqual(record["log_sha256"], launcher.sha256(record["log"]))
        self.assertFalse(popen.call_args.kwargs["start_new_session"])
        self.assertNotIn("process_group", popen.call_args.kwargs)
        self.assertEqual(popen.call_args.kwargs["stderr"], launcher.subprocess.STDOUT)
        for key, value in {**launcher.BlasEnvironment, "OMP_NUM_THREADS": "1"}.items():
            self.assertEqual(popen.call_args.kwargs["env"][key], value)
        self.assertEqual(record["command"][-7:], record["requested_target_arguments"])
        self.assertEqual(record["hashes"][str(self.configPath)], launcher.sha256(self.configPath))
        self.assertEqual(record["allocation_evidence"]["sha256"], launcher.sha256(self.allocationPath))
        self.assertEqual(record["hashes"][str(self.allocationPath)], launcher.sha256(self.allocationPath))

    ## @brief Verify native failure or invalid bindings never become success.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_native_failure_or_invalid_bindings_never_become_success(self):
        result, record, _, _, _ = self.mocked_launch([], code=7)
        self.assertEqual((result, record["return_code"], record["status"]), (7, 7, "execution_failed"))
        result, record, _, _, _ = self.mocked_launch([], extra=b"GPU_BINDING not-json\n")
        self.assertEqual(result, 2)
        self.assertEqual(record["return_code"], 0)
        self.assertEqual(record["status"], "launcher_rejected")
        self.assertTrue(record["binding_errors"])

    ## @brief Verify cpu reference success has no gpu records or helper.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_cpu_reference_success_has_no_gpu_records_or_helper(self):
        result, record, _, _, _ = self.mocked_launch([], target="FastPMEvolution")
        self.assertEqual(result, 0)
        self.assertEqual(record["target_kind"], "cpu")
        self.assertEqual(record["bindings"], [])
        self.assertNotIn(self.files["gpu_rank.py"], record["command"])

    ## @brief Verify launch artifact names are exclusive and preserve previous evidence.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_launch_artifact_names_are_exclusive_and_preserve_previous_evidence(self):
        self.mocked_launch([], target="FastPMForce")
        first = {path: path.read_bytes() for path in (self.root/"evidence").iterdir()}
        self.mocked_launch([], target="FastPMForce")
        self.assertEqual(len(list((self.root/"evidence").glob("*.json"))), 2)
        self.assertEqual(len(list((self.root/"evidence").glob("*.log"))), 2)
        for path, contents in first.items():
            self.assertEqual(path.read_bytes(), contents)

    ## @brief Verify preflight rejection never starts mpi.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_preflight_rejection_never_starts_mpi(self):
        with (patch.dict(os.environ, {}, clear=True), patch.object(launcher.subprocess, "Popen") as popen,
              redirect_stderr(io.StringIO())):
            result = launcher.main(["--config", str(self.configPath), "-n", "1", self.files["Cosmology"]])
        self.assertEqual(result, 2)
        popen.assert_not_called()
        self.assertFalse((self.root/"evidence").exists())


## @cond CLI_DISPATCH
if __name__ == "__main__":
    unittest.main()
## @endcond
