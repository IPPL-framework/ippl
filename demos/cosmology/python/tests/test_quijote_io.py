#!/usr/bin/env python3
## @file test_quijote_io.py
# @brief Independent Gadget/HDF5 fixtures for units, split-file integrity and canonical import.
# @ingroup cosmology_python
# @details Tests encode raw Gadget records and official HDF5 datasets directly. The momentum oracle evaluates
# p=a*v_pec/100 independently of the reader's raw-u conversion; exact endpoint
# wrapping and shuffled one-based IDs must preserve the periodic phase space.
"""Tiny independent fixtures; HDF5 tests use optional h5py and hdf5plugin."""
from __future__ import annotations

import json
import importlib
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

## @cond RUNTIME_SETTINGS
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
## @endcond
import quijote_io as io

## @var Hdf5
# @brief Optional serial HDF5 fixture writer; format-1 tests run when h5py is unavailable.
Hdf5 = None
try:
    Hdf5 = importlib.import_module("h5py")
except ImportError:
    pass


## @brief Encode one Gadget file independently, with configurable corruption and precision.
# @param path Fixture destination.
# @param ids One-based labels, allowed to be malformed in negative tests.
# @param total Global particle count recorded in every split header.
# @param files Total split file count, independently checked by the reader.
# @param endian Fixture '<' or '>' byte order.
# @param floatSize Encoded position/velocity element width, 4 or 8 bytes.
# @param idSize Encoded unsigned ID width, 4 or 8 bytes.
# @param a Header scale factor; velocities encode fixed physical peculiar values.
# @param changes Optional raw header replacements as (offset,format,values).
# @param positions Optional raw comoving positions in kpc/h; defaults to exact label-derived coordinates.
# @param velocities Optional raw Gadget u=v_pec/sqrt(a), in km/s; defaults to a known physical-velocity field.
# @param trailing Add an unsupported extra Gadget record to exercise strict record rejection.
# @return Path to the written tiny synthetic Gadget file.
def write_fixture(path, ids=(1, 2, 3, 4), *, total=None, files=1, endian="<",
                  floatSize=4, idSize=4, a=0.25, changes=(), positions=None,
                  velocities=None, trailing=False):
    ids = np.asarray(ids, dtype=np.uint64)
    total = len(ids) if total is None else total
    header = bytearray(256)
    struct.pack_into(endian + "6I", header, 0, 0, len(ids), 0, 0, 0, 0)
    struct.pack_into(endian + "6d", header, 24, 0, 12.5, 0, 0, 0, 0)
    struct.pack_into(endian + "2d", header, 72, a, 1 / a - 1)
    struct.pack_into(endian + "6I", header, 96, 0, total & 0xFFFFFFFF, 0, 0, 0, 0)
    struct.pack_into(endian + "i", header, 124, files)
    struct.pack_into(endian + "4d", header, 128, 1e6, 0.3175, 0.6825, 0.6711)
    struct.pack_into(endian + "6I", header, 168, 0, total >> 32, 0, 0, 0, 0)
    for offset, fmt, values in changes:
        struct.pack_into(endian + fmt, header, offset, *values)
    # Inputs remain representable at both widths so precision cases compare exactly.
    if positions is None:
        positions = np.column_stack((ids * 1000, ids * 2000, ids * 3000))
    if velocities is None:
        physicalVelocity = np.column_stack((ids * 4, -ids.astype(float) * 8, ids * 12))
        velocities = physicalVelocity / math.sqrt(a)
    payloads = [header, np.asarray(positions, dtype=endian + "f" + str(floatSize)).tobytes(),
                np.asarray(velocities, dtype=endian + "f" + str(floatSize)).tobytes(),
                np.asarray(ids, dtype=endian + "u" + str(idSize)).tobytes()]
    if trailing:
        payloads.append(b"optional unknown block")
    with Path(path).open("wb") as stream:
        for payload in payloads:
            marker = struct.pack(endian + "I", len(payload))
            stream.write(marker)
            stream.write(payload)
            stream.write(marker)
    return Path(path)


## @brief Independently encode a tiny official-layout HDF5 IC shard with raw Gadget velocities.
# @param path Destination file.
# @param ids One-based particle IDs, possibly invalid for negative tests.
# @param total Global count; defaults to this fixture's count.
# @param files Header shard count.
# @param endian Explicit array byte order, '<' or '>'.
# @param floatSize Particle position/velocity width, 4 or 8 bytes.
# @param idSize Unsigned ID width, 4 or 8 bytes.
# @param a Scale factor; velocities represent known physical values divided by sqrt(a).
# @param changes Dictionary replacing official HDF5 header attributes.
# @param positions Optional raw kpc/h coordinates.
# @param velocities Optional raw u=v_pec/sqrt(a) values.
# @param compression HDF5 dataset creation keywords for independent filter fixtures.
# @param compressionInfo Include the official JSON provenance with zero truncated bits.
# @return Written fixture path; never calls the production source-header decoder.
def write_hdf5_fixture(path, ids=(1, 2, 3, 4), *, total=None, files=1, endian="<",
                       floatSize=4, idSize=4, a=0.25, changes=None, positions=None,
                       velocities=None, compression=None, compressionInfo=True):
    ids = np.asarray(ids, dtype=np.uint64)
    total = len(ids) if total is None else total
    attrs = dict(NumPart_ThisFile=np.array([0, len(ids), 0, 0, 0, 0], dtype=endian + "i4"),
                 NumPart_Total=np.array([0, total & 0xFFFFFFFF, 0, 0, 0, 0], dtype=endian + "u4"),
                 NumPart_Total_HighWord=np.array([0, total >> 32, 0, 0, 0, 0], dtype=endian + "u4"),
                 MassTable=np.array([0, 12.5, 0, 0, 0, 0], dtype=endian + "f8"),
                 Time=np.float64(a), Redshift=np.float64(1 / a - 1), NumFilesPerSnapshot=np.int32(files),
                 BoxSize=np.float64(1e6), Omega0=np.float64(0.3175),
                 OmegaLambda=np.float64(0.6825), HubbleParam=np.float64(0.6711))
    attrs.update(changes or {})
    if positions is None:
        positions = np.column_stack((ids * 1000, ids * 2000, ids * 3000))
    if velocities is None:
        physicalVelocity = np.column_stack((ids * 4, -ids.astype(float) * 8, ids * 12))
        velocities = physicalVelocity / math.sqrt(a)
    arrays = dict(Coordinates=np.asarray(positions, dtype=endian + "f" + str(floatSize)).reshape(-1, 3),
                  Velocities=np.asarray(velocities, dtype=endian + "f" + str(floatSize)).reshape(-1, 3),
                  ParticleIDs=np.asarray(ids, dtype=endian + "u" + str(idSize)))
    with Hdf5.File(path, "w") as stream:
        header = stream.create_group("Header")
        for name, value in attrs.items():
            header.attrs[name] = value
        particles = stream.create_group("PartType1")
        for name, values in arrays.items():
            particles.create_dataset(name, data=values, **(compression or {}))
        if compressionInfo:
            info = {name: dict(truncbits=0, hdf5=dict(dtype=values.dtype.str))
                    for name, values in arrays.items()}
            info["sort"] = False
            stream.create_group("CompressionInfo").attrs["json"] = json.dumps(info)
    return Path(path)


## @brief Test published-state preservation and fail-closed import contracts.
class QuijoteIOTests(unittest.TestCase):
    ## @brief Give every test an isolated temporary source and output tree.
    # @return None; retains only test-owned temporary resources.
    def setUp(self):
        ## @var temp
        # @brief TemporaryDirectory owner retaining the isolated fixture tree until tearDown.
        self.temp = tempfile.TemporaryDirectory()
        ## @var root
        # @brief Absolute root for this test's tiny input/output fixture files.
        self.root = Path(self.temp.name)

    ## @brief Remove only the tiny files created by this test.
    # @return None; releases the owned temporary fixture directory.
    def tearDown(self):
        self.temp.cleanup()

    ## @brief Compare independently encoded 32/64-bit and little/big-endian inputs.
    # @return None; unittest assertions enforce the declared import/serialization contract.
    def test_all_input_precisions_and_endianness(self):
        for endian in ("<", ">"):
            for floatSize in (4, 8):
                for idSize in (4, 8):
                    with self.subTest(endian=endian, floatSize=floatSize, idSize=idSize):
                        path = write_fixture(self.root / "ics", ids=(4, 1, 3, 2), endian=endian,
                                             floatSize=floatSize, idSize=idSize)
                        metadata = io.inspect_gadget_files(path)
                        self.assertEqual(metadata["total_count"], 4)
                        self.assertEqual(metadata["particle_mass_msun_h"], 12.5e10)
                        chunks = list(io.iter_gadget_chunks(metadata, chunk_size=2))
                        ids = np.concatenate([chunk[0] for chunk in chunks])
                        xyz = np.concatenate([chunk[1] for chunk in chunks])
                        momentum = np.concatenate([chunk[2] for chunk in chunks])
                        np.testing.assert_array_equal(ids, [3, 0, 2, 1])
                        labels = ids + 1
                        np.testing.assert_array_equal(xyz, np.column_stack((labels, 2 * labels, 3 * labels)))
                        physicalVelocity = np.column_stack((labels * 4, -labels.astype(float) * 8, labels * 12))
                        np.testing.assert_allclose(momentum, 0.25 * physicalVelocity / 100, rtol=0, atol=0)

    ## @brief Reorder shuffled split ICs on disk and verify the independent wire layout and hashes.
    # @return None; unittest assertions enforce the declared import/serialization contract.
    def test_split_conversion_round_trip_and_manifest(self):
        write_fixture(self.root / "ics.0", (6, 2, 4), total=6, files=2, floatSize=8, idSize=8)
        write_fixture(self.root / "ics.1", (5, 3, 1), total=6, files=2, floatSize=8, idSize=8)
        output = self.root / "canonical.bin"
        manifest = io.convert(self.root / "ics", output, chunk_size=2,
                              expected={"total_count": 6, "omega_m": 0.3175, "box_mpc_h": 1000})
        data = output.read_bytes()
        header = struct.unpack("<8sQQ6dQ48x", data[:128])
        self.assertEqual(header[:3], (b"IPPLPS01", 6, 6))
        self.assertEqual(header[-1], 1)
        self.assertEqual(header[-2], 12.5e10)
        self.assertEqual(len(data), 128 + 6 * 56)
        for index in range(6):
            record = struct.unpack_from("<Q6d", data, 128 + index * 56)
            label = index + 1
            self.assertEqual(record[:4], (index, label, 2 * label, 3 * label))
            np.testing.assert_allclose(record[4:], [0.01 * label, -0.02 * label, 0.03 * label], rtol=0, atol=3e-17)
        self.assertEqual(manifest["output"]["sha256"], io.sha256(output))
        self.assertEqual(json.loads(output.with_suffix(".bin.json").read_text()), manifest)
        records = np.concatenate(list(io.iter_records(output, chunk_size=1)))
        np.testing.assert_array_equal(records["id"], np.arange(6))
        self.assertEqual(io.read_header(output)["file_count"], 6)
        self.assertFalse(list(self.root.glob(".quijote-convert-*")))

    ## @brief Unit conversion at nontrivial a catches missing sqrt(a), a or factor 100.
    # @return None; unittest assertions enforce the declared import/serialization contract.
    def test_velocity_convention_at_catalogue_epoch(self):
        a = 1 / 128
        velocity = np.array([[120, -55, 9], [-11, 13, 300]], dtype=float)
        source = write_fixture(self.root / "ics", (2, 1), a=a, floatSize=8,
                               velocities=velocity / math.sqrt(a))
        _, _, momentum = next(io.iter_gadget_chunks(source))
        np.testing.assert_allclose(momentum, a * velocity / 100, rtol=2e-16, atol=1e-17)
        self.assertGreater(float(np.max(np.abs(momentum - velocity / 100))), 1)

    ## @brief Exact upper endpoints wrap to zero without perturbing any other coordinate.
    # @return None; unittest assertions enforce the declared import/serialization contract.
    def test_exact_upper_endpoint_only_is_wrapped(self):
        source = write_fixture(self.root / "ics", (1,), positions=[[0, 1e6, 1e6 - 1]], floatSize=8)
        _, xyz, _ = next(io.iter_gadget_chunks(source))
        np.testing.assert_array_equal(xyz, [[0, 0, 999.999]])

    ## @brief Reject missing split files even when one shard itself parses correctly.
    # @return None; unittest assertions enforce the declared import/serialization contract.
    def test_missing_split_file(self):
        source = write_fixture(self.root / "ics.0", (1, 2), total=4, files=2)
        with self.assertRaisesRegex(ValueError, "coverage"):
            io.inspect_gadget_files(source)

    ## @brief Reject mismatching split headers before accepting any particle data.
    # @return None; unittest assertions enforce the declared import/serialization contract.
    def test_split_header_disagreement(self):
        a = write_fixture(self.root / "ics.0", (1, 2), total=4, files=2)
        b = write_fixture(self.root / "ics.1", (3, 4), total=4, files=2, a=0.5)
        with self.assertRaisesRegex(ValueError, "headers disagree"):
            io.inspect_gadget_files([a, b])

    ## @brief Reject repeated file paths, noncontiguous suffixes and mixed prefixes.
    # @return None; unittest assertions enforce the declared import/serialization contract.
    def test_split_filename_coverage(self):
        a = write_fixture(self.root / "ics.0", (1, 2), total=4, files=2)
        for name in ("ics.2", "other.1", "unsuffixed"):
            b = write_fixture(self.root / name, (3, 4), total=4, files=2)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "numbered"):
                io.inspect_gadget_files([a, b])
        with self.assertRaisesRegex(ValueError, "distinct"):
            io.inspect_gadget_files([a, a])

    ## @brief Fail on wrong species, variable/nonpositive mass, background or epoch.
    # @return None; unittest assertions enforce the declared import/serialization contract.
    def test_invalid_scientific_headers(self):
        changes = [((0, "I", (1,)), "type-1"),
                   ((96, "I", (1,)), "type-1"),
                   ((32, "d", (0,)), "positive"),
                   ((32, "d", (-1,)), "positive"),
                   ((32, "d", (float("nan"),)), "Nonfinite"),
                   ((144, "d", (0.5,)), "flat"),
                   ((72, "d", (0,)), "Scale factor"),
                   ((80, "d", (49,)), "inconsistent"),
                   ((128, "d", (0,)), "positive"),
                   ((152, "d", (float("inf"),)), "Nonfinite")]
        for change, message in changes:
            with self.subTest(change=change):
                path = write_fixture(self.root / "ics", changes=[change])
                with self.assertRaisesRegex(ValueError, message):
                    io.inspect_gadget_files(path)

    ## @brief Discrepant physical mass is reported, not hidden by a new acceptance threshold.
    # @return None; unittest assertions enforce the declared import/serialization contract.
    def test_implied_mass_ratio_is_diagnostic(self):
        source = write_fixture(self.root / "ics")
        metadata = io.inspect_gadget_files(source)
        expectedMass = 2.77536627e11 * 0.3175 * 1000**3 / 4
        self.assertAlmostEqual(metadata["implied_mass_ratio"], 12.5e10 / expectedMass)

    ## @brief Detect both same-chunk and cross-chunk duplicated IDs and clean failed output.
    # @return None; unittest assertions enforce the declared import/serialization contract.
    def test_duplicate_ids_and_failure_cleanup(self):
        for chunk in (1, 4):
            with self.subTest(chunk=chunk):
                source = write_fixture(self.root / "ics", (1, 2, 1, 4))
                output = self.root / f"failed{chunk}.bin"
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    io.convert(source, output, chunk_size=chunk)
                self.assertFalse(output.exists())
                self.assertFalse(output.with_suffix(".bin.json").exists())
                self.assertFalse(list(self.root.glob(".quijote-convert-*")))

    ## @brief One-based label contract refuses both zero and IDs beyond global N.
    # @return None; unittest assertions enforce the declared import/serialization contract.
    def test_out_of_range_ids(self):
        for ids in ((0, 2, 3, 4), (1, 2, 3, 5)):
            with self.subTest(ids=ids):
                source = write_fixture(self.root / "ics", ids)
                with self.assertRaisesRegex(ValueError, "one-based"):
                    io.convert(source, self.root / "bad.bin")

    ## @brief Catch nonfinite source velocity and positions outside the declared periodic box.
    # @return None; unittest assertions enforce the declared import/serialization contract.
    def test_nonfinite_or_outside_phase_space(self):
        for values in (dict(velocities=[[float("nan"), 0, 0]]),
                       dict(positions=[[float("inf"), 0, 0]]),
                       dict(positions=[[-1, 0, 0]]), dict(positions=[[1000001, 0, 0]])):
            with self.subTest(values=values):
                source = write_fixture(self.root / "ics", (1,), **values)
                with self.assertRaises(ValueError):
                    io.convert(source, self.root / "bad.bin")

    ## @brief Require caller identity assertions to match and reject unknown expected keys.
    # @return None; unittest assertions enforce the declared import/serialization contract.
    def test_expected_metadata(self):
        source = write_fixture(self.root / "ics")
        for expected in ({"total_count": 512**3}, {"a": 1 / 128}, {"hubble": 0.7},
                         {"box_mpc_h": 1}, {"omgea_m": 0.3175}, {"omega_m": float("nan")}):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                io.inspect_gadget_files(source, expected=expected)

    ## @brief Reject broken record markers, truncation, missing records and optional trailing blocks.
    # @return None; unittest assertions enforce the declared import/serialization contract.
    def test_malformed_records(self):
        source = write_fixture(self.root / "ics")
        original = source.read_bytes()
        variants = [original[:100], original[:-1], original[:264],
                    original[:260] + struct.pack("<I", 255) + original[264:],
                    original[:264] + struct.pack("<I", 24) + original[268:]]
        for data in variants:
            source.write_bytes(data)
            with self.assertRaises(ValueError):
                io.inspect_gadget_files(source)
        source = write_fixture(source, trailing=True)
        with self.assertRaisesRegex(ValueError, "trailing"):
            io.inspect_gadget_files(source)

    ## @brief Fail explicitly for unavailable prefixes and unsupported HDF5 input.
    # @return None; unittest assertions enforce the declared import/serialization contract.
    def test_unavailable_and_hdf5_inputs(self):
        with self.assertRaises(FileNotFoundError):
            io.inspect_gadget_files(self.root / "missing")
        path = self.root / "ics.hdf5"
        path.write_bytes(b"\x89HDF\r\n\x1a\n" + bytes(300))
        with self.assertRaisesRegex(ValueError, "not HDF5"):
            io.inspect_gadget_files(path)

    ## @brief Conversion refuses either pre-existing artifact without altering its bytes.
    # @return None; unittest assertions enforce the declared import/serialization contract.
    def test_existing_output_or_manifest_is_preserved(self):
        source = write_fixture(self.root / "ics")
        for name in ("saved.bin", "saved.bin.json"):
            path = self.root / name
            path.write_bytes(b"user data")
            with self.assertRaises(FileExistsError):
                io.convert(source, self.root / "saved.bin")
            self.assertEqual(path.read_bytes(), b"user data")
            path.unlink()

    ## @brief Source mutation after inspection invalidates the frozen manifest and output.
    # @return None; unittest assertions enforce the declared import/serialization contract.
    def test_changed_source_is_rejected(self):
        source = write_fixture(self.root / "ics")
        original = io.iter_gadget_chunks
        ## @brief Mutate one source byte after yielding valid records to exercise the post-conversion hash guard.
        # @param paths Decoded Gadget metadata forwarded unchanged to the genuine chunk reader.
        # @param chunk_size Positive bounded particle count forwarded to the genuine reader.
        # @return Iterator of valid records followed by one intentional source-file mutation.
        def changed(paths, chunk_size):
            yield from original(paths, chunk_size=chunk_size)
            with source.open("r+b") as stream:
                stream.seek(4 + 220)
                stream.write(b"x")
        with mock.patch.object(io, "iter_gadget_chunks", changed):
            with self.assertRaisesRegex(ValueError, "changed during"):
                io.convert(source, self.root / "bad.bin")
        self.assertFalse((self.root / "bad.bin").exists())

    ## @brief Validate canonical flags, exact length, reserved bytes and sorted IDs independently.
    # @return None; unittest assertions enforce the declared import/serialization contract.
    def test_canonical_corruption(self):
        source = write_fixture(self.root / "ics")
        output = self.root / "good.bin"
        io.convert(source, output)
        original = output.read_bytes()
        variants = [b"BADMAGIC" + original[8:], original[:79], original + b"x",
                    original[:80] + b"x" + original[81:],
                    original[:72] + struct.pack("<Q", 2) + original[80:]]
        for data in variants:
            output.write_bytes(data)
            with self.assertRaises(ValueError):
                io.read_header(output)
        output.write_bytes(original[:128] + struct.pack("<Q", 2) + original[136:])
        with self.assertRaisesRegex(ValueError, "record indices"):
            list(io.iter_records(output))

    ## @brief Empty unordered output shards have valid headers and yield no records.
    # @return None; unittest assertions enforce the declared import/serialization contract.
    def test_empty_canonical_snapshot_shard(self):
        path = self.root / "empty.bin"
        path.write_bytes(struct.pack("<8sQQ6dQ48x", b"IPPLPS01", 0, 4,
                                     0.5, 1000, 0.3175, 0.6825, 0.6711, 1e10, 0))
        self.assertEqual(io.read_header(path)["file_count"], 0)
        self.assertEqual(list(io.iter_records(path)), [])

    ## @brief Zero/negative/nonintegral chunks fail before streaming allocations.
    # @return None; unittest assertions enforce the declared import/serialization contract.
    def test_invalid_chunk_sizes(self):
        source = write_fixture(self.root / "ics")
        for chunk in (0, -1, True, 1.5):
            with self.subTest(chunk=chunk), self.assertRaisesRegex(ValueError, "chunk_size"):
                io.convert(source, self.root / "bad.bin", chunk_size=chunk)


## @brief Qualify offline HDF5 input, exact source preservation and fail-closed scientific metadata.
@unittest.skipUnless(Hdf5 is not None, "h5py is optional for Gadget format-1 users")
class QuijoteHdf5Tests(unittest.TestCase):
    ## @brief Allocate a tiny isolated fixture tree; no catalogue data is downloaded.
    # @return None; retains temporary resources until tearDown.
    def setUp(self):
        ## @var temp
        # @brief Owner of this test's temporary HDF5 and canonical files.
        self.temp = tempfile.TemporaryDirectory()
        ## @var root
        # @brief Absolute fixture root.
        self.root = Path(self.temp.name)

    ## @brief Remove only the current test's fixture files.
    # @return None.
    def tearDown(self):
        self.temp.cleanup()

    ## @brief Both byte orders and 32/64-bit storage produce the same physical phase space.
    # @return None; exact unit checks catch an extra or omitted sqrt(a) factor.
    def test_precisions_endianness_and_velocity_units(self):
        for endian in ("<", ">"):
            for floatSize in (4, 8):
                for idSize in (4, 8):
                    with self.subTest(endian=endian, floatSize=floatSize, idSize=idSize):
                        path = write_hdf5_fixture(self.root / "ics.hdf5", (4, 1, 3, 2),
                            endian=endian, floatSize=floatSize, idSize=idSize)
                        metadata = io.inspect_hdf5_files(path)
                        self.assertEqual(metadata["input_format"], "gadget-hdf5")
                        chunks = list(io.iter_hdf5_chunks(metadata, chunk_size=2))
                        ids = np.concatenate([part[0] for part in chunks])
                        labels = ids + 1
                        np.testing.assert_array_equal(ids, [3, 0, 2, 1])
                        np.testing.assert_array_equal(np.concatenate([part[1] for part in chunks]),
                            np.column_stack((labels, 2 * labels, 3 * labels)))
                        physical = np.column_stack((labels * 4, -labels.astype(float) * 8, labels * 12))
                        np.testing.assert_array_equal(np.concatenate([part[2] for part in chunks]),
                                                      0.25 * physical / 100)

    ## @brief Eight shuffled HDF5 shards convert bit-identically to independently encoded Gadget ICs.
    # @return None; checks the wire contract, prefix coverage, catalogue assertions and manifest provenance.
    def test_eight_shards_identical_canonical_and_provenance(self):
        for index in range(8):
            write_hdf5_fixture(self.root / f"ics.{index}.hdf5", (16 - index, index + 1),
                               total=16, files=8, a=1 / 128)
        expected = {"a": 1 / 128, "total_count": 16, "omega_m": 0.3175}
        manifest = io.convert(self.root / "ics", self.root / "hdf.bin", chunk_size=1, expected=expected)
        gadget = write_fixture(self.root / "gadget", range(1, 17), a=1 / 128)
        io.convert(gadget, self.root / "gadget.bin", chunk_size=3)
        self.assertEqual((self.root / "hdf.bin").read_bytes(), (self.root / "gadget.bin").read_bytes())
        self.assertEqual(manifest["schema"], io.Hdf5Schema)
        self.assertEqual(manifest["initial_conditions"]["input_format"], "gadget-hdf5")
        self.assertEqual(len(manifest["sources"]), 8)
        for source in manifest["sources"]:
            self.assertEqual(source["sha256"], io.sha256(source["path"]))
            self.assertEqual(source["blocks"]["velocity"]["dataset"], "/PartType1/Velocities")
            self.assertEqual(source["compression_info"]["Velocities"]["truncbits"], 0)
        self.assertEqual(json.loads((self.root / "hdf.bin.json").read_text()), manifest)

    ## @brief Read the real public-release Blosc bitshuffle/Zstandard filter and its dependency failure.
    # @return None; skips only when the optional production filter package is unavailable.
    def test_official_blosc_filter_and_dependency_error(self):
        try:
            plugin = importlib.import_module("hdf5plugin")
        except ImportError:
            self.skipTest("hdf5plugin required to exercise the official Quijote Blosc filter")
        options = plugin.Blosc(cname="zstd", clevel=5, shuffle=plugin.Blosc.BITSHUFFLE)
        path = write_hdf5_fixture(self.root / "ics.hdf5", compression=options, a=1 / 128)
        manifest = io.convert(path, self.root / "blosc.bin", chunk_size=1)
        filters = manifest["sources"][0]["blocks"]["position"]["filters"]
        self.assertEqual([entry["id"] for entry in filters], [32001])
        gadget = write_fixture(self.root / "gadget", a=1 / 128)
        io.convert(gadget, self.root / "plain.bin")
        self.assertEqual((self.root / "blosc.bin").read_bytes(), (self.root / "plain.bin").read_bytes())
        metadataPath = self.root / "metadata.json"
        metadataPath.write_text(json.dumps(io.inspect_hdf5_files(path)))
        program = ("import json,sys; sys.path.insert(0, sys.argv[1]); import quijote_io; "
                   "metadata=json.load(open(sys.argv[2])); "
                   "print(sum(len(chunk[0]) for chunk in quijote_io.iter_hdf5_chunks(metadata, 1)))")
        process = subprocess.run([sys.executable, "-B", "-c", program,
                                  str(Path(io.__file__).parent), str(metadataPath)],
                                 check=True, capture_output=True, text=True)
        self.assertEqual(process.stdout.strip(), "4")
        with mock.patch.object(Hdf5.h5z, "filter_avail", return_value=False), \
                mock.patch.object(io.importlib, "import_module", side_effect=ImportError("missing plugin")):
            with Hdf5.File(path, "r") as stream, self.assertRaisesRegex(ValueError, "requires hdf5plugin"):
                io._hdf5_filters(stream["PartType1/Coordinates"], Hdf5)

    ## @brief Missing h5py has an explicit error and never affects the format-1 path.
    # @return None.
    def test_optional_h5py_dependency(self):
        with mock.patch.object(io.importlib, "import_module", side_effect=ImportError("missing h5py")):
            with self.assertRaisesRegex(ValueError, "requires h5py"):
                io.inspect_hdf5_files(self.root / "not-opened.hdf5")
            source = write_fixture(self.root / "gadget")
            self.assertEqual(io.inspect_source_files(source)["input_format"], "gadget-format1")

    ## @brief Reject invalid species, counts, physical mass, epoch, and non-scalar/vector attributes.
    # @return None; scientific metadata is never silently coerced to accepted counts.
    def test_invalid_scientific_headers_and_attribute_types(self):
        cases = [dict(NumPart_ThisFile=[1, 4, 0, 0, 0, 0]),
                 dict(NumPart_Total=[0, 4, 1, 0, 0, 0]),
                 dict(NumPart_ThisFile=[0, -1, 0, 0, 0, 0]),
                 dict(NumPart_ThisFile=[0., 4., 0., 0., 0., 0.]),
                 dict(NumPart_Total_HighWord=[0, 2**32, 0, 0, 0, 0]),
                 dict(NumFilesPerSnapshot=0), dict(NumFilesPerSnapshot=1.5),
                 dict(MassTable=[0, 0, 0, 0, 0, 0]), dict(MassTable=[0, -1, 0, 0, 0, 0]),
                 dict(MassTable=[0, float("nan"), 0, 0, 0, 0]),
                 dict(OmegaLambda=0.5), dict(Time=0), dict(Redshift=9),
                 dict(BoxSize=0), dict(HubbleParam=float("inf")), dict(Time=[0.25]),
                 dict(MassTable=[12.5])]
        for changes in cases:
            with self.subTest(changes=changes):
                path = write_hdf5_fixture(self.root / "bad.hdf5", changes=changes)
                with self.assertRaises(ValueError):
                    io.inspect_hdf5_files(path)
        path = write_hdf5_fixture(self.root / "missing.hdf5")
        with Hdf5.File(path, "r+") as stream:
            del stream["Header"].attrs["Time"]
        with self.assertRaisesRegex(ValueError, "Missing HDF5 header attribute"):
            io.inspect_hdf5_files(path)
        high = write_hdf5_fixture(self.root / "high.hdf5", total=2**32 + 4)
        with self.assertRaisesRegex(ValueError, "coverage"):
            io.inspect_hdf5_files(high)

    ## @brief Require complete same-prefix HDF5 shards with identical mass, epoch and background.
    # @return None.
    def test_coverage_headers_and_expected_identity(self):
        first = write_hdf5_fixture(self.root / "ics.0.hdf5", (1, 2), total=4, files=2)
        with self.assertRaisesRegex(ValueError, "coverage"):
            io.inspect_hdf5_files(first)
        for name, kwargs in (("ics.2.hdf5", {}), ("other.1.hdf5", {}),
                             ("ics.1.hdf5", {"a": 0.5}),
                             ("ics.1.hdf5", {"changes": {"MassTable": [0, 13, 0, 0, 0, 0]}})):
            with self.subTest(name=name, kwargs=kwargs):
                second = write_hdf5_fixture(self.root / name, (3, 4), total=4, files=2, **kwargs)
                with self.assertRaises(ValueError):
                    io.inspect_hdf5_files([first, second])
        source = write_hdf5_fixture(self.root / "single.hdf5")
        with self.assertRaisesRegex(ValueError, "differs from expected"):
            io.inspect_source_files(source, expected={"a": 1 / 128})
        with self.assertRaisesRegex(ValueError, "distinct"):
            io.inspect_hdf5_files([source, source])

    ## @brief Reject malformed dataset precision, dimensionality, species and variable masses.
    # @return None.
    def test_dataset_shapes_types_and_species(self):
        cases = [("Coordinates", np.zeros((4, 2), dtype="f4")),
                 ("Velocities", np.zeros((3, 3), dtype="f4")),
                 ("Coordinates", np.zeros((4, 3), dtype="i4")),
                 ("Velocities", np.zeros((4, 3), dtype="f2")),
                 ("ParticleIDs", np.arange(1, 5, dtype="i8")),
                 ("ParticleIDs", np.arange(1, 5, dtype="f8")),
                 ("ParticleIDs", np.arange(1, 5, dtype="u2")),
                 ("ParticleIDs", np.arange(1, 5, dtype="u4").reshape(4, 1))]
        for name, values in cases:
            with self.subTest(name=name, dtype=values.dtype, shape=values.shape):
                path = write_hdf5_fixture(self.root / "bad.hdf5")
                with Hdf5.File(path, "r+") as stream:
                    del stream["PartType1"][name]
                    stream["PartType1"].create_dataset(name, data=values)
                with self.assertRaisesRegex(ValueError, "shape or dtype"):
                    io.inspect_hdf5_files(path)
        for name in ("PartType1/Masses", "PartType2/ParticleIDs"):
            path = write_hdf5_fixture(self.root / "extra.hdf5")
            with Hdf5.File(path, "r+") as stream:
                stream.create_dataset(name, data=[1])
            with self.subTest(name=name), self.assertRaises(ValueError):
                io.inspect_hdf5_files(path)
        path = write_hdf5_fixture(self.root / "missing.hdf5")
        with Hdf5.File(path, "r+") as stream:
            del stream["PartType1/ParticleIDs"]
        with self.assertRaises(ValueError):
            io.inspect_hdf5_files(path)

    ## @brief IC provenance must report zero truncated bits for every particle field.
    # @return None; also rejects a lossy HDF5 scale-offset filter and malformed metadata.
    def test_lossy_or_invalid_compression(self):
        for field in ("Coordinates", "Velocities", "ParticleIDs"):
            path = write_hdf5_fixture(self.root / "lossy.hdf5")
            with Hdf5.File(path, "r+") as stream:
                info = json.loads(stream["CompressionInfo"].attrs["json"])
                info[field]["truncbits"] = 1
                stream["CompressionInfo"].attrs["json"] = json.dumps(info)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Lossy"):
                io.inspect_hdf5_files(path)
        for text in ("invalid json", "[]", "{}"):
            path = write_hdf5_fixture(self.root / "malformed.hdf5")
            with Hdf5.File(path, "r+") as stream:
                stream["CompressionInfo"].attrs["json"] = text
            with self.subTest(text=text), self.assertRaises(ValueError):
                io.inspect_hdf5_files(path)
        path = write_hdf5_fixture(self.root / "dtype.hdf5")
        with Hdf5.File(path, "r+") as stream:
            info = json.loads(stream["CompressionInfo"].attrs["json"])
            info["Coordinates"]["hdf5"]["dtype"] = "f8"
            stream["CompressionInfo"].attrs["json"] = json.dumps(info)
        with self.assertRaisesRegex(ValueError, "dtype disagrees"):
            io.inspect_hdf5_files(path)
        path = write_hdf5_fixture(self.root / "filter.hdf5", compression=dict(scaleoffset=1))
        with self.assertRaisesRegex(ValueError, "lossy HDF5 filter"):
            io.inspect_hdf5_files(path)

    ## @brief Missing, duplicated or out-of-range labels cannot produce a canonical artifact.
    # @return None; same-chunk and cross-chunk duplicate checks run for both input formats.
    def test_invalid_ids_and_failed_output_cleanup(self):
        for ids in ((1, 2, 1, 4), (0, 2, 3, 4), (1, 2, 3, 5)):
            for chunk in (1, 4):
                with self.subTest(ids=ids, chunk=chunk):
                    path = write_hdf5_fixture(self.root / "bad.hdf5", ids)
                    with self.assertRaises(ValueError):
                        io.convert(path, self.root / "bad.bin", chunk_size=chunk)
                    self.assertFalse((self.root / "bad.bin").exists())
                    self.assertFalse((self.root / "bad.bin.json").exists())
                    self.assertFalse(list(self.root.glob(".quijote-convert-*")))

    ## @brief Reject nonfinite phase space and out-of-box values; preserve exact periodic wrapping.
    # @return None.
    def test_phase_space_validation_and_periodic_endpoint(self):
        for kwargs in (dict(positions=[[float("nan"), 0, 0]]), dict(positions=[[-1, 0, 0]]),
                       dict(positions=[[1000001, 0, 0]]), dict(velocities=[[0, float("inf"), 0]])):
            path = write_hdf5_fixture(self.root / "bad.hdf5", (1,), **kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                io.convert(path, self.root / "bad.bin")
        path = write_hdf5_fixture(self.root / "wrapped.hdf5", (1,),
                                  positions=[[0, 1e6, 1e6 - 1]], floatSize=8)
        _, positions, _ = next(io.iter_hdf5_chunks(path))
        np.testing.assert_array_equal(positions, [[0, 0, 999.999]])

    ## @brief Signature detection accepts uncompressed files and rejects mixed storage formats.
    # @return None.
    def test_signature_detection_unannotated_input_and_mixed_format(self):
        path = write_hdf5_fixture(self.root / "unsuffixed", compressionInfo=False)
        self.assertEqual(io.inspect_source_files(path)["input_format"], "gadget-hdf5")
        metadata = io.inspect_hdf5_files(path)
        self.assertIsNone(metadata["files"][0]["compression_info"])
        plain = write_fixture(self.root / "gadget")
        with self.assertRaisesRegex(ValueError, "Mixed"):
            io.inspect_source_files([path, plain])
        path = write_hdf5_fixture(self.root / "prefix.hdf5")
        self.assertEqual(io.inspect_source_files(self.root / "prefix")["total_count"], 4)

    ## @brief Forbid linked or externally stored particle bytes omitted from source SHA256 provenance.
    # @return None.
    def test_nonlocal_hdf5_storage_rejected(self):
        for kind in ("soft", "external"):
            path = write_hdf5_fixture(self.root / "link.hdf5")
            with Hdf5.File(path, "r+") as stream:
                del stream["PartType1/Coordinates"]
                if kind == "soft":
                    stream["PartType1/Coordinates"] = Hdf5.SoftLink("/PartType1/Velocities")
                else:
                    stream["PartType1"].create_dataset("Coordinates", shape=(4, 3), dtype="f4",
                        external=[(str(self.root / "external.bin"), 0, 4 * 3 * 4)])
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "nonlocal|External"):
                io.inspect_hdf5_files(path)

    ## @brief Dataset reads remain bounded slices even when particles span multiple compressed chunks.
    # @return None; no production read may use a full-array selector.
    def test_bounded_slices_and_storage_chunk_limit(self):
        path = write_hdf5_fixture(self.root / "gzip.hdf5", range(1, 10), compression=dict(compression="gzip"))
        original = Hdf5.Dataset.__getitem__
        selections = []
        ## @brief Record production particle slice lengths while forwarding the genuine HDF5 read.
        # @param dataset Open source particle dataset.
        # @param selector Explicit bounded first-axis slice.
        # @return Actual decoded particle values.
        def checked(dataset, selector):
            self.assertIsInstance(selector, slice)
            self.assertLessEqual(selector.stop - selector.start, 2)
            selections.append(selector)
            return original(dataset, selector)
        with mock.patch.object(Hdf5.Dataset, "__getitem__", checked):
            io.convert(path, self.root / "bounded.bin", chunk_size=2)
        self.assertEqual(len(selections), 15)
        with mock.patch.object(io, "MaxHdf5ChunkBytes", 16), self.assertRaisesRegex(ValueError, "memory bound"):
            io.inspect_hdf5_files(path)

    ## @brief A source changed during conversion is detected before publishing output or manifest.
    # @return None.
    def test_source_mutation_rejected(self):
        path = write_hdf5_fixture(self.root / "source.hdf5")
        original = io.iter_hdf5_chunks
        ## @brief Append a byte after the validated particle read to simulate a concurrent source mutation.
        # @param metadata Inspected source metadata.
        # @param chunk_size Bounded source particle chunk.
        # @return Genuine converted arrays followed by an intentional file mutation.
        def changed(metadata, chunk_size):
            yield from original(metadata, chunk_size=chunk_size)
            with path.open("ab") as stream:
                stream.write(b"x")
        with mock.patch.object(io, "iter_hdf5_chunks", changed), self.assertRaisesRegex(ValueError, "changed during"):
            io.convert(path, self.root / "bad.bin")
        self.assertFalse((self.root / "bad.bin").exists())

    ## @brief Empty HDF5 shards are valid when their declared complete set still contains every ID.
    # @return None.
    def test_empty_shard(self):
        write_hdf5_fixture(self.root / "ics.0.hdf5", (), total=2, files=2)
        write_hdf5_fixture(self.root / "ics.1.hdf5", (2, 1), total=2, files=2)
        manifest = io.convert(self.root / "ics", self.root / "complete.bin", chunk_size=1)
        self.assertEqual(manifest["output"]["header"]["file_count"], 2)


## @cond CLI_DISPATCH
if __name__ == "__main__":
    unittest.main()
## @endcond
