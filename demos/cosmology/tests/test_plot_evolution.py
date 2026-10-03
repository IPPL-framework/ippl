#!/usr/bin/env python3
"""Small mathematical/provenance regression tests for the static summary."""
import gzip
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plot_evolution as plotting


class PlotEvolutionTests(unittest.TestCase):
    def test_identical_nonzero_modes_keep_exact_zero_residual(self):
        result = plotting.direct_metrics([1+2j, 3j], [1+2j, 3j])
        self.assertEqual(result["power_ippl"], 14)
        self.assertEqual(result["complex_relative"], 0)
        self.assertEqual(result["absolute_complex_difference"], 0)

    def test_zero_signal_stays_zero_with_undefined_normalization(self):
        result = plotting.direct_metrics([0j, 0j], [0j, 0j])
        self.assertEqual(result["power_ippl"], 0)
        self.assertEqual(result["power_fastpm"], 0)
        self.assertIsNone(result["complex_relative"])
        self.assertFalse(result["normalization_defined"])

    def test_opposite_phases_have_nonzero_complex_residual(self):
        result = plotting.direct_metrics([1+0j], [-1+0j])
        self.assertEqual(result["power_ippl"], result["power_fastpm"])
        self.assertEqual(result["complex_relative"], 2)

    def test_invalid_modes_rejected(self):
        for left, right in (([], []), ([1], [1, 2]), ([np.nan], [1])):
            with self.assertRaises(ValueError):
                plotting.direct_metrics(left, right)

    def test_failed_campaign_is_preserved_not_rejected_or_passed(self):
        failure = {"name": "retained failure", "passed": False, "value": .0027, "limit": .002}
        report = {"schema": "ippl-fastpm-evolution-v1", "complete": True, "passed": False,
                  "checks": [failure], "failed_checks": [failure]}
        self.assertEqual(plotting.campaign_status(report), [failure])
        self.assertIn("1 failed checks", plotting.status_text({"failed_checks": [failure]}))
        report["passed"] = True
        with self.assertRaises(ValueError):
            plotting.campaign_status(report)

    def test_incomplete_campaign_rejected(self):
        with self.assertRaises(ValueError):
            plotting.campaign_status({"schema": "ippl-fastpm-evolution-v1", "complete": False})

    def test_compressed_snapshot_hash_and_sorted_id_alignment(self):
        csv = b'id,x,y,z,px,py,pz\n1,4,5,6,.4,.5,.6\n0,1,2,3,.1,.2,.3\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'particles_checkpoint0000_rank0.csv.gz'
            path.write_bytes(gzip.compress(csv, mtime=0))
            run = {"output": directory, "ranks": 1, "snapshots": [{"path": str(path),
                   "sha256": plotting.sha256(path), "csv_sha256": hashlib.sha256(csv).hexdigest()}]}
            hashes = {}
            frame = plotting.snapshot(run, 0, 2, hashes)
            self.assertEqual(frame.id.tolist(), [0, 1])
            self.assertEqual(frame.x.tolist(), [1, 4])
            self.assertEqual(hashes[str(path.resolve())], plotting.sha256(path))
            path.write_bytes(path.read_bytes()+b'corrupt')
            with self.assertRaises(ValueError):
                plotting.snapshot(run, 0, 2, {})


if __name__ == '__main__':
    unittest.main()
