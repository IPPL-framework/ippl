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

    def test_shell_syntax(self):
        subprocess.run(["bash", "-n", str(Launcher)], check=True, timeout=5)

    def test_missing_and_relative_output_arguments_rejected(self):
        self.rejected([])
        self.rejected(["relative-output"])

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
        hostname = self.root / "hostname"
        hostname.write_text("#!/bin/sh\nprintf 'merlin-l-001\\n'\n")
        hostname.chmod(0o755)
        result = self.rejected([str(self.output)], {
            "SLURM_JOB_ID": "synthetic", "SLURM_JOB_NUM_NODES": "1", "SLURM_NTASKS": "4",
            "SLURM_CPUS_PER_TASK": "1", "SLURM_CPUS_ON_NODE": "4",
            "PATH": str(self.root) + os.pathsep + self.environment["PATH"]})
        self.assertIn("Refusing a non-compute host", result.stderr)


if __name__ == "__main__":
    unittest.main()
