"""Launcher preflight tests; never run modules, builds, MPI, or network calls."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


Launcher = Path(__file__).resolve().parents[1] / "merlin" / "cpu_validation.sh"


class MerlinLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "must-not-be-created"
        self.environment = {key: value for key, value in os.environ.items()
                            if not key.startswith("SLURM_")}

    def tearDown(self):
        self.temporary.cleanup()

    def rejected(self, arguments, additions=None):
        environment = dict(self.environment)
        environment.update(additions or {})
        result = subprocess.run(["bash", str(Launcher), *arguments], env=environment,
                                capture_output=True, text=True, timeout=5)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.output.exists())
        return result

    def mocked_host(self, name):
        hostname = self.root / "hostname"
        hostname.write_text(f"#!/bin/sh\nprintf '%s\\n' '{name}'\n")
        hostname.chmod(0o755)
        return {"PATH": str(self.root) + os.pathsep + self.environment["PATH"]}

    def reaches_source_preflight(self, arguments, additions):
        # A sentinel proves the host/allocation guards accepted the invocation,
        # while stopping before mkdir, module initialization or any real git call.
        git = self.root / "git"
        git.write_text("#!/bin/sh\nprintf 'mock source preflight\\n' >&2\nexit 73\n")
        git.chmod(0o755)
        result = self.rejected(arguments, additions)
        self.assertEqual(result.returncode, 73, result.stdout + result.stderr)
        self.assertIn("mock source preflight", result.stderr)

    def test_shell_syntax(self):
        subprocess.run(["bash", "-n", str(Launcher)], check=True, timeout=5)

    def test_missing_and_relative_output_arguments_rejected(self):
        for arguments in ([], ["relative-output"], ["--login"],
                          ["--login", "relative-output"], ["--unknown", str(self.output)],
                          ["--login", "--login", str(self.output)]):
            with self.subTest(arguments=arguments):
                self.rejected(arguments)

    def test_outside_slurm_rejected_before_creating_output(self):
        self.rejected([str(self.output)])

    def test_wrong_allocation_rejected(self):
        valid = {"SLURM_JOB_ID": "synthetic", "SLURM_JOB_NUM_NODES": "1",
                 "SLURM_NTASKS": "4", "SLURM_CPUS_PER_TASK": "1", "SLURM_CPUS_ON_NODE": "4"}
        for key, value in (("SLURM_JOB_NUM_NODES", "2"), ("SLURM_NTASKS", "1"),
                           ("SLURM_CPUS_PER_TASK", "2"), ("SLURM_CPUS_ON_NODE", "8"),
                           ("SLURM_CPUS_ON_NODE", "2")):
            with self.subTest(key=key, value=value):
                self.rejected([str(self.output)], {**valid, key: value})

    def test_login_host_rejected_even_with_valid_allocation_variables(self):
        result = self.rejected([str(self.output)], {
            "SLURM_JOB_ID": "synthetic", "SLURM_JOB_NUM_NODES": "1", "SLURM_NTASKS": "4",
            "SLURM_CPUS_PER_TASK": "1", "SLURM_CPUS_ON_NODE": "4",
            **self.mocked_host("merlin-l-001")})
        self.assertIn("Refusing a non-compute host", result.stderr)

    def test_explicit_login_mode_accepts_login_host_without_slurm(self):
        self.reaches_source_preflight(["--login", str(self.output)],
                                      self.mocked_host("merlin-l-001"))

    def test_login_mode_rejects_compute_and_unrelated_hosts(self):
        for hostname in ("merlin-c-001", "merlin-g-100", "laptop", "merlin-l"):
            with self.subTest(hostname=hostname):
                result = self.rejected(["--login", str(self.output)], self.mocked_host(hostname))
                self.assertIn("outside a Merlin login host", result.stderr)

    def test_login_mode_rejects_inherited_slurm_allocation(self):
        result = self.rejected(["--login", str(self.output)], {
            **self.mocked_host("merlin-l-001"), "SLURM_JOB_ID": "12345"})
        self.assertIn("inherited Slurm job allocation", result.stderr)

    def test_compute_mode_still_accepts_four_cpu_slurm_allocation(self):
        for hostname in ("merlin-c-001", "merlin-g-100"):
            with self.subTest(hostname=hostname):
                self.reaches_source_preflight([str(self.output)], {
                    **self.mocked_host(hostname), "SLURM_JOB_ID": "12345",
                    "SLURM_JOB_NUM_NODES": "1", "SLURM_NTASKS": "4",
                    "SLURM_CPUS_PER_TASK": "1", "SLURM_CPUS_ON_NODE": "4"})

    def test_login_mode_preserves_existing_evidence_directory(self):
        environment = {**self.environment, **self.mocked_host("merlin-l-001")}
        git = self.root / "git"
        git.write_text("#!/bin/sh\nexit 0\n")
        git.chmod(0o755)
        self.output.mkdir()
        sentinel = self.output / "retained.txt"
        sentinel.write_text("retained evidence")
        result = subprocess.run(["bash", str(Launcher), "--login", str(self.output)],
                                env=environment, capture_output=True, text=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sentinel.read_text(), "retained evidence")
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_login_mode_records_truthful_evidence_without_scheduler_query(self):
        environment = {**self.environment, **self.mocked_host("merlin-l-001")}
        commands = {
            "git": "exit 0",
            "module": "printf 'mock module %s\\n' \"$*\"",
            # Stop immediately before environment creation: all earlier Python
            # version queries are mocked, not executed on a remote machine.
            "python3": 'if [ "$1" = -m ]; then exit 74; fi\nprintf "mock Python\\n"',
            "scontrol": "printf 'UNEXPECTED_SCHEDULER_QUERY\\n' >&2\nexit 75",
        }
        for name in ("gcc", "mpiexec", "mpicc", "cmake"):
            commands[name] = "printf 'mock version\\n'"
        for name, body in commands.items():
            command = self.root / name
            command.write_text("#!/bin/sh\n" + body + "\n")
            command.chmod(0o755)
        result = subprocess.run(["bash", str(Launcher), "--login", str(self.output)],
                                env=environment, capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 74, result.stdout + result.stderr)
        evidence = (self.output / "environment.txt").read_text()
        self.assertIn("execution_mode=login ranks_max=4 build_jobs=4", evidence)
        self.assertIn("slurm_job=none cpu_budget=4 source=explicit_user_authorized_login_mode", evidence)
        self.assertNotIn("UNEXPECTED_SCHEDULER_QUERY", result.stdout + result.stderr + evidence)
        self.assertEqual((self.output / "controller-exit.txt").read_text(),
                         "phase=python_environment exit=74\n")


if __name__ == "__main__":
    unittest.main()
