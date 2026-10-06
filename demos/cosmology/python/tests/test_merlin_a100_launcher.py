## @file test_merlin_a100_launcher.py
# @brief A100 allocation guards; no modules or device queries run in these tests.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
# Tests assert declared invariants using isolated fixtures; no scientific tolerances are relaxed by documentation.
"""A100 allocation guards; no modules or device queries run in these tests."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

## @var Launcher
# @brief Named Launcher protocol/schema value; the source initializer records its exact contents.
Launcher = Path(__file__).resolve().parents[2] / 'merlin/a100_validation.sh'


## @brief Regression suite for A100Launcher.
# @see cosmology_tools
class A100LauncherTests(unittest.TestCase):
    ## @brief Verify syntax and unsafe invocation rejected.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_syntax_and_unsafe_invocation_rejected(self):
        subprocess.run(['bash', '-n', str(Launcher)], check=True, timeout=5)
        environment = {key: value for key, value in os.environ.items() if not key.startswith('SLURM_')}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'must-not-be-created'
            for args in ([], ['run', 'relative', str(output)], ['build', directory, str(output)]):
                result = subprocess.run(['bash', str(Launcher), *args], env=environment,
                                        capture_output=True, timeout=5)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(output.exists())
            environment.update(SLURM_JOB_ID='synthetic', SLURM_JOB_NUM_NODES='1', SLURM_NTASKS='4',
                               SLURM_CPUS_PER_TASK='1', SLURM_CPUS_ON_NODE='8')
            result = subprocess.run(['bash', str(Launcher), 'build', directory, str(output)],
                                    env=environment, capture_output=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())


## @cond CLI_DISPATCH
if __name__ == '__main__':
    unittest.main()
## @endcond
