#!/usr/bin/env python3
## @file test_quijote_benchmark.py
# @brief Analytical estimator, canonical shard, and dry-run provenance tests.
# @ingroup cosmology_python
# @details Tests cover L^3 power normalization, phase-sensitive cross power,
# independent particle sums, R2C multiplicities, RSD units and chunk bounds.
"""Small synthetic arrays only; no cosmological simulations or catalogue downloads."""
from pathlib import Path
import json
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quijote_analysis as analysis
import quijote_benchmark as benchmark


## @brief Evaluate lattice.
# @param n Synthetic cubic lattice side in particles.
# @param box Positive periodic comoving box side in Mpc/h.
# @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
def lattice(n, box):
    axis = (np.arange(n) + .5) * box / n
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)


## @brief Tiny repeatable particle source for independent estimator tests.
class ArraySource:
    ## @brief Initialize the documented source geometry and chunked particle access.
    # @param positions Finite synthetic particle positions with shape (N,3), in Mpc/h.
    # @param box Positive periodic comoving box side in Mpc/h.
    # @param chunk Positive particle count per synthetic test chunk.
    def __init__(self, positions, box, chunk=17):
        ## @brief Small synthetic position array owned by this test source.
        self.array = positions
        ## @brief Global number of equal-mass particles.
        self.count = len(positions)
        ## @brief Periodic comoving box side in Mpc/h.
        self.box = box
        ## @brief Maximum records in each synthetic position buffer.
        self.chunk = chunk

    ## @brief Yield bounded chunks of synthetic particle positions.
    # @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
    def positions(self):
        for first in range(0, len(self.array), self.chunk):
            yield self.array[first:first + self.chunk]


## @brief Independent wire fixture, deliberately not written through converter code.
# @param path Input or output filesystem path; the calling contract determines freshness and format.
# @param positions Finite synthetic particle positions with shape (N,3), in Mpc/h.
# @param ids Optional explicit zero-based uint64 particle IDs for wire fixtures.
# @param total Optional global count for a fixture representing one rank shard.
# @param ordered Whether the fixture claims complete ID-ordered canonical input.
# @param a Dimensionless scale factor written to the fixture header.
# @param box Positive periodic comoving box side in Mpc/h.
# @param momentum Optional (N,3) canonical momenta p=a*vpec/100 for the synthetic fixture.
# @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
def canonical(path, positions, *, ids=None, total=None, ordered=True, a=1/128, box=1000., momentum=None):
    """Independent wire fixture, deliberately not written through converter code."""
    if ids is None:
        ids = np.arange(len(positions))
    if total is None:
        total = len(positions)
    if momentum is None:
        momentum = np.zeros_like(positions)
    with Path(path).open("wb") as stream:
        stream.write(struct.pack("<8sQQ6dQ48x", b"IPPLPS01", len(positions), total,
                                 a, box, .3175, .6825, .6711, 1e10, int(ordered)))
        for label, position, mom in zip(ids, positions, momentum):
            stream.write(struct.pack("<Q6d", int(label), *position, *mom))
    return Path(path)


## @brief Analytical spectral estimator and physical-unit regression tests.
class EstimatorTests(unittest.TestCase):
    ## @brief Verify chunk deposition mass periodicity and chunk invariance.
    # @return None; unittest assertions report a violated invariant.
    def test_chunk_deposition_mass_periodicity_and_chunk_invariance(self):
        positions = np.array([[1.25, 2.5, 3.75], [-6.75, 10.5, 11.75]])
        one, error = analysis.deposit_cic([positions], 8., 8, 2)
        two, _ = analysis.deposit_cic(iter([positions[:1], positions[1:]]), 8., 8, 2)
        np.testing.assert_array_equal(one, two)
        self.assertLess(error, 1e-14)
        counts = (one + 1) * 2 / 8**3
        self.assertAlmostEqual(counts.sum(), 2.)
        self.assertAlmostEqual(counts[1, 2, 3], 2*.75*.5*.25)
        with self.assertRaises(ValueError):
            analysis.deposit_cic([positions], 8., 8, 3)

    ## @brief Verify direct oracle phase sign and interlaced measurement.
    # @return None; unittest assertions report a violated invariant.
    def test_direct_oracle_phase_sign_and_interlaced_measurement(self):
        positions = lattice(16, 8.)
        positions[:, 0] += .1 * np.sin(2*np.pi*positions[:, 0]/8)
        source = ArraySource(positions, 8., 29)
        modes = np.array([[1, 0, 0], [2, 0, 0], [0, 1, 0]])
        exact = np.array([sum(np.exp(-2j*np.pi*sum(int(m[d])*float(x[d]) / 8 for d in range(3)))
                             for x in positions) / len(positions) for m in modes])
        direct = analysis.direct_coefficients(source.positions(), 8., modes, len(positions))
        np.testing.assert_allclose(direct, exact, atol=1e-14)
        coarse, _ = analysis.fourier_field(source, 64)
        self.assertGreater(np.max(np.abs(coarse[tuple(modes.T)] - exact)), 2e-5)
        del coarse
        # Refine the estimator; do not relax the gate for a weak aliased mode.
        field, errors = analysis.fourier_field(source, 256)
        measured = field[tuple(modes.T)]
        np.testing.assert_allclose(measured, exact, atol=2e-5)
        self.assertLess(max(errors), 1e-13)
        self.assertLess(exact[0].real, 0)

    ## @brief Verify power volume cross phase r2c and shot noise.
    # @return None; unittest assertions report a violated invariant.
    def test_power_volume_cross_phase_r2c_and_shot_noise(self):
        left = np.zeros((8, 8, 5), dtype=complex)
        right = np.zeros_like(left)
        left[0, 0, 1] = .2
        right[0, 0, 1] = .1 * np.exp(1j*np.pi/3)
        edges = np.array([.5, 1.5]) * 2*np.pi/8
        row = analysis.shell_spectra(left, right, 8., edges, 64, 64)[0]
        self.assertEqual(row["modes"], 18)  # six axial + twelve face-diagonal vectors
        self.assertAlmostEqual(row["left_raw"], 8**3 * 2 * .2**2 / 18)
        self.assertAlmostEqual(row["ratio"], 4.)
        self.assertAlmostEqual(row["correlation_raw"], .5)
        scaled = analysis.shell_spectra(left, right, 16., edges/2, 64, 64)[0]
        self.assertAlmostEqual(scaled["left_raw"] / row["left_raw"], 8.)
        self.assertAlmostEqual(scaled["k_mean"] / row["k_mean"], .5)
        sub = analysis.shell_spectra(left, right, 8., edges, 64, 64, "poisson")[0]
        self.assertLess(sub["left_power"], 0.)
        self.assertIsNone(sub["ratio"])
        self.assertAlmostEqual(sub["correlation_raw"], .5)
        self.assertEqual(sub["cross_raw"], row["cross_raw"])

    ## @brief Verify multipoles for line of sight mode.
    # @return None; unittest assertions report a violated invariant.
    def test_multipoles_for_line_of_sight_mode(self):
        field = np.zeros((8, 8, 5), dtype=complex)
        field[0, 0, 1] = .2
        edges = np.array([.5, 1.5]) * 2*np.pi/8
        along = analysis.shell_spectra(field, field, 8., edges, 64, 64, line_of_sight=2)[0]
        across = analysis.shell_spectra(field, field, 8., edges, 64, 64, line_of_sight=0)[0]
        self.assertAlmostEqual(along["left_p2_raw"], 5 * along["left_raw"])
        self.assertAlmostEqual(along["left_p4_raw"], 9 * along["left_raw"])
        self.assertAlmostEqual(across["left_p2_raw"], -2.5 * across["left_raw"])
        self.assertAlmostEqual(across["left_p4_raw"], 27/8 * across["left_raw"])

    ## @brief Verify reject out of band and memory budget.
    # @return None; unittest assertions report a violated invariant.
    def test_reject_out_of_band_and_memory_budget(self):
        field = np.zeros((8, 8, 5), dtype=complex)
        with self.assertRaises(ValueError):
            analysis.shell_spectra(field, field, 8., [0, 4.], 64, 64)
        self.assertGreater(analysis.analysis_memory_bytes(512, 262144, 512**3), 8*1024**3)
        edges = benchmark.physical_edges(.01, .1, .04)
        np.testing.assert_allclose(edges, [.01, .05, .09, .1])


## @brief Canonical-file integrity, execution gate and provenance regression tests.
class FileAndRunnerTests(unittest.TestCase):
    ## @brief Create an isolated tiny canonical fixture and temporary directory.
    # @return None; unittest assertions report a violated invariant.
    def setUp(self):
        ## @brief Task-owned temporary directory containing only synthetic fixtures.
        self.temporary = tempfile.TemporaryDirectory()
        ## @brief Root path of the isolated synthetic test directory.
        self.root = Path(self.temporary.name)
        ## @brief Small finite synthetic 4 cubed particle lattice in Mpc/h.
        self.positions = lattice(4, 1000.)
        ## @brief Complete sorted canonical IC file written by the independent fixture helper.
        self.ic = canonical(self.root / "ic.bin", self.positions)

    ## @brief Remove only the temporary fixture directory owned by this test.
    # @return None; unittest assertions report a violated invariant.
    def tearDown(self):
        self.temporary.cleanup()

    ## @brief Verify chunked source manifest and rsd velocity convention.
    # @return None; unittest assertions report a violated invariant.
    def test_chunked_source_manifest_and_rsd_velocity_convention(self):
        momenta = np.full_like(self.positions, .01)
        source_path = canonical(self.root / "velocity.bin", self.positions, momentum=momenta, a=.5)
        source = analysis.ParticleSource([source_path], chunk_size=7)
        self.assertEqual([len(chunk) for chunk in source.records()], [7]*9 + [1])
        source.validate()
        shifted = np.concatenate(list(source.positions(0)))
        expected = self.positions.copy()
        expected[:, 0] = (expected[:, 0] + .01/(.5**2 * np.sqrt(.3175/.5**3+.6825))) % 1000
        np.testing.assert_allclose(shifted, expected)
        np.testing.assert_array_equal(np.concatenate(list(source.positions())), self.positions)

    ## @brief Verify shard coverage duplicate ids and manifest epoch.
    # @return None; unittest assertions report a violated invariant.
    def test_shard_coverage_duplicate_ids_and_manifest_epoch(self):
        first = canonical(self.root / "particles_final_rank0.bin", self.positions[::2],
                          ids=np.arange(0, 64, 2), total=64, ordered=False, a=1.)
        second = canonical(self.root / "particles_final_rank1.bin", self.positions[1::2],
                           ids=np.arange(1, 64, 2), total=64, ordered=False, a=1.)
        table = self.root / "snapshots.csv"
        table.write_text("name,step,a,z,ranks,format\nfinal,10,1,0,2,binary\n")
        paths, a = analysis.resolve_sources([str(table)])
        source = analysis.ParticleSource(paths, chunk_size=5, expected_a=a)
        self.assertEqual(len(source.validate()), 2)
        canonical(second, self.positions[1::2], ids=np.arange(0, 64, 2), total=64, ordered=False, a=1.)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            analysis.ParticleSource([first, second]).validate()
        with self.assertRaisesRegex(ValueError, "epoch"):
            analysis.ParticleSource([first, second], expected_a=.5)

    ## @brief Verify full compare identical fields and preflight.
    # @return None; unittest assertions report a violated invariant.
    def test_full_compare_identical_fields_and_preflight(self):
        pos = self.positions.copy()
        pos[:, 0] += 10*np.sin(2*np.pi*pos[:, 0]/1000)
        path = canonical(self.root / "perturbed.bin", pos)
        source = analysis.ParticleSource([path], 9)
        with self.assertRaises(MemoryError):
            analysis.compare(source, source, grid=32, edges=[0, .02], memory_limit_bytes=1)
        with patch.object(analysis, "fourier_field") as transform:
            with self.assertRaisesRegex(ValueError, "Nyquist"):
                analysis.compare(source, source, grid=32, edges=[0, 1.], memory_limit_bytes=1024**3)
            transform.assert_not_called()
        report = analysis.compare(source, source, grid=32, edges=[.001, .01, .02],
                                  memory_limit_bytes=1024**3)
        self.assertTrue(report["rows"])
        for row in report["rows"]:
            self.assertAlmostEqual(row["ratio"], 1.)
            self.assertAlmostEqual(row["correlation_raw"], 1.)
        self.assertEqual(report["shot_noise"], "raw")
        changed = canonical(self.root / "later.bin", pos, a=.5)
        with self.assertRaisesRegex(ValueError, "differ in a"):
            analysis.compare(source, analysis.ParticleSource([changed]), grid=32, edges=[0, .02])

    ## @brief Construct an isolated test CLI with no simulation launch by default.
    # @param extra Additional command-line arguments for the isolated dry-run test.
    # @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
    def arguments(self, *extra):
        return benchmark.parser().parse_args(["prepare", "--ic", str(self.ic), "--exe", "/bin/echo",
                                              "--output-dir", str(self.root / "prepared"),
                                              "--mesh", "8", "--steps", "16", "--disk-reserve-gib", "0",
                                              "--source-dir", str(self.root), *extra])

    ## @brief Verify dry run defaults and exact fiducial gate.
    # @return None; unittest assertions report a violated invariant.
    def test_dry_run_defaults_and_exact_fiducial_gate(self):
        with self.assertRaisesRegex(ValueError, "fiducial"):
            benchmark.prepare(self.arguments())
        self.assertFalse((self.root / "prepared").exists())
        with patch.object(benchmark, "revision_record", return_value={"revision": "test"}), \
                patch.object(benchmark.subprocess, "run") as launch:
            result = benchmark.prepare(self.arguments("--allow-nonfiducial"))
            launch.assert_not_called()
        self.assertFalse(result["executed"])
        self.assertEqual(result["state"], "prepared")
        self.assertFalse(result["header_matches_fiducial"])
        self.assertEqual(result["config"]["particle_count"], 64)
        self.assertEqual(result["config"]["np"], 8)
        self.assertEqual(result["config"]["output_redshifts"], "1,0.5,0")
        self.assertEqual(result["ic"]["sha256"], analysis.sha256(self.ic))
        saved = json.loads((self.root / "prepared/benchmark.json").read_text())
        self.assertEqual(saved["command"], result["command"])
        with self.assertRaises(FileExistsError):
            benchmark.prepare(self.arguments("--allow-nonfiducial"))

    ## @brief Verify execute prepared verifies frozen files and rejects false completion.
    # @return None; unittest assertions report a violated invariant.
    def test_execute_prepared_verifies_frozen_files_and_rejects_false_completion(self):
        with patch.object(benchmark, "revision_record", return_value={"revision": "test"}):
            benchmark.prepare(self.arguments("--allow-nonfiducial"))
        manifest_path = self.root / "prepared/benchmark.json"
        with patch.object(benchmark.subprocess, "run") as launch:
            result = benchmark.execute_prepared(manifest_path)
            launch.assert_not_called()
        self.assertFalse(result["executed"])
        with patch.object(benchmark.subprocess, "run") as launch:
            launch.return_value.returncode = 0
            with self.assertRaises(FileNotFoundError):
                benchmark.execute_prepared(manifest_path, run=True)
        saved = json.loads(manifest_path.read_text())
        self.assertEqual(saved["state"], "interrupted_or_failed")
        with self.assertRaisesRegex(ValueError, "Only a prepared"):
            benchmark.execute_prepared(manifest_path, run=True)

    ## @brief Verify execute prepared rejects changed config.
    # @return None; unittest assertions report a violated invariant.
    def test_execute_prepared_rejects_changed_config(self):
        with patch.object(benchmark, "revision_record", return_value={"revision": "test"}):
            benchmark.prepare(self.arguments("--allow-nonfiducial"))
        (self.root / "prepared/input.par").write_text("np=4\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            benchmark.execute_prepared(self.root / "prepared/benchmark.json")

    ## @brief Verify simulation budget and schedule fail before output.
    # @return None; unittest assertions report a violated invariant.
    def test_simulation_budget_and_schedule_fail_before_output(self):
        with self.assertRaises(MemoryError):
            benchmark.prepare(self.arguments("--allow-nonfiducial", "--memory-limit-gib", ".001"))
        with self.assertRaises(ValueError):
            benchmark.prepare(self.arguments("--allow-nonfiducial", "--redshifts", "0", "0"))
        self.assertFalse((self.root / "prepared").exists())
        header = {**benchmark.Fiducial, "flags": 1, "file_count": 512**3}
        self.assertEqual(benchmark.validate_fiducial(header), [])
        header["hubble"] = .7
        with self.assertRaisesRegex(ValueError, "hubble"):
            benchmark.validate_fiducial(header)

    ## @brief Accept representational metadata roundoff while rejecting physical drift and count/mass changes.
    # @return None; unittest assertions report a violated invariant.
    def test_output_metadata_roundoff_is_distinct_from_physical_drift(self):
        import quijote_io
        header = quijote_io.read_header(self.ic)
        canonical(self.root / "particles_initial_rank0.bin", self.positions)
        canonical(self.root / "particles_final_rank0.bin", self.positions, a=1.)
        table = self.root / "snapshots.csv"
        table.write_text("name,step,a,z,ranks,format\ninitial,0,0.0078125,127,1,binary\nfinal,2,1,0,1,binary\n")
        rounded = dict(header)
        rounded["omega_lambda"] = np.nextafter(header["omega_lambda"], np.inf)
        rounded["omega_m"] = np.nextafter(header["omega_m"], np.inf)
        rounded["box_mpc_h"] = np.nextafter(header["box_mpc_h"], np.inf)
        rounded["hubble"] = np.nextafter(header["hubble"], np.inf)
        rounded["a"] = header["a"] * (1 + 5e-11)
        self.assertEqual(len(benchmark.completed_output_record(self.root, [0.], rounded)["epochs"]), 2)
        for key in ("omega_m", "omega_lambda", "box_mpc_h", "hubble"):
            with self.subTest(key=key):
                changed = {**header, key: header[key] * (1 + 1e-6)}
                with self.assertRaisesRegex(ValueError, "changed " + key):
                    benchmark.completed_output_record(self.root, [0.], changed)
        for key, value in (("total_count", header["total_count"] + 1),
                           ("particle_mass_msun_h", np.nextafter(header["particle_mass_msun_h"], np.inf))):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "changed " + key):
                    benchmark.completed_output_record(self.root, [0.], {**header, key: value})

    ## @brief Verify output contract rejects missing or wrong epochs.
    # @return None; unittest assertions report a violated invariant.
    def test_output_contract_rejects_missing_or_wrong_epochs(self):
        import quijote_io
        header = quijote_io.read_header(self.ic)
        canonical(self.root / "particles_initial_rank0.bin", self.positions)
        canonical(self.root / "particles_final_rank0.bin", self.positions, a=1.)
        table = self.root / "snapshots.csv"
        table.write_text("name,step,a,z,ranks,format\ninitial,0,0.0078125,127,1,binary\nfinal,2,1,0,1,binary\n")
        output = benchmark.completed_output_record(self.root, [0.], header)
        self.assertEqual(len(output["epochs"]), 2)
        with self.assertRaisesRegex(ValueError, "requested epoch"):
            benchmark.completed_output_record(self.root, [1., 0.], header)
        table.write_text("name,step,a,z,ranks,format\ninitial,0,0.0078125,127,1,binary\nfinal,2,0.5,1,1,binary\n")
        with self.assertRaisesRegex(ValueError, "epochs disagree"):
            benchmark.completed_output_record(self.root, [0.], header)


if __name__ == "__main__":
    unittest.main()
