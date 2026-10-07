#!/usr/bin/env python3
## @file quijote_io.py
# @brief Bounded serial Gadget format-1/HDF5 import preserving published Quijote phase space.
# @ingroup cosmology_python
# @details For Gadget raw velocity u=v_pec/sqrt(a), canonical momentum is
# p=a^(3/2)u/100 in Mpc/h; x[Mpc/h]=x_Gadget[kpc/h]/1000. No growth
# rescaling, IC generation or change of the input Lagrangian order occurs.
# The IPPLPS01 stream stores equal-weight collisionless particles; physical
# mass is retained in the header, while the solver normalizes the density by N/L^3.
"""Read split Gadget format-1 or HDF5 ICs into an ID-ordered binary stream.

Only the documented Quijote units and one-based contiguous type-1 IDs are
accepted. HDF5 input requires h5py; the official Blosc-compressed release also
requires hdf5plugin. This offline serial conversion does not use parallel
HDF5 simulation I/O. Variable masses, extra species, lossy compression and
optional trailing Gadget blocks are rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile

import numpy as np

## @var Schema
# @brief Version identifier for converter manifests and their input/output provenance.
Schema = "ippl-quijote-format1-conversion-v1"
## @var Hdf5Schema
# @brief HDF5 manifest version; the canonical binary contract remains IPPLPS01.
Hdf5Schema = "ippl-quijote-hdf5-conversion-v1"
## @var Hdf5Magic
# @brief Standard HDF5 file signature; format detection does not depend on a filename suffix.
Hdf5Magic = b"\x89HDF\r\n\x1a\n"
## @var MaxHdf5ChunkBytes
# @brief Maximum decoded storage-chunk size (64 MiB), bounding filter working memory.
MaxHdf5ChunkBytes = 64 * 1024 * 1024
## @var CanonicalSchema
# @brief Version identifier reported by the canonical binary header reader.
CanonicalSchema = "ippl-phase-space-v1"
## @var HeaderFormat
# @brief Little-endian magic, two counts, six physical scalars, flags and 48 zero reserved bytes.
HeaderFormat = "<8sQQ6dQ48x"
## @var HeaderStruct
# @brief Compiled 128-byte header encoder/decoder; native alignment never enters the wire format.
HeaderStruct = struct.Struct(HeaderFormat)
## @var HeaderSize
# @brief Exact byte offset of the first canonical particle record, 128.
HeaderSize = HeaderStruct.size
## @var HeaderBytes
# @brief Alias for HeaderSize used by downstream mmap readers.
HeaderBytes = HeaderSize
## @var Magic
# @brief Eight-byte IPPLPS01 signature selecting the version-one phase-space contract.
Magic = b"IPPLPS01"
## @var RecordDtype
# @brief One uint64 ID followed by position and canonical momentum vectors, all little-endian.
RecordDtype = np.dtype([("id", "<u8"), ("position", "<f8", (3,)),
                        ("momentum", "<f8", (3,))])
## @var RecordSize
# @brief Fixed 56-byte particle record size without native padding.
RecordSize = RecordDtype.itemsize
## @var DefaultChunkSize
# @brief Maximum default particle count per materialized source chunk; reorder arrays are disk backed.
DefaultChunkSize = 262144
## @var CriticalDensity
# @brief Present critical density in Msun h^2/Mpc^3 used only to report the implied mass ratio.
CriticalDensity = 2.77536627e11  # Msun h^2 / Mpc^3; report only, not a gate.


## @brief Return SHA256 of exact file bytes with bounded host memory.
# @param path Existing input or output file.
# @return Lowercase hexadecimal SHA256.
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


## @brief Reject invalid chunk sizes before allocating any arrays.
# @param chunk_size Positive integer number of records per bounded host chunk.
# @return None; invalid sizes raise ValueError.
def _validate_chunk_size(chunk_size):
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")


## @brief Validate the supported flat matter-plus-Lambda background and length units.
# @param metadata Mapping with finite a,L,Omega_m,Omega_Lambda,h and physical mass.
# @return None; unsupported physical parameters raise ValueError.
def _validate_background(metadata):
    names = ("a", "box_mpc_h", "omega_m", "omega_lambda", "hubble", "particle_mass_msun_h")
    if any(not math.isfinite(metadata[name]) for name in names):
        raise ValueError("Nonfinite cosmology or physical mass in header")
    if not 0 < metadata["a"] <= 1:
        raise ValueError("Scale factor must satisfy 0 < a <= 1")
    if metadata["box_mpc_h"] <= 0 or metadata["hubble"] <= 0 or metadata["particle_mass_msun_h"] <= 0:
        raise ValueError("Box, Hubble parameter and equal particle mass must be positive")
    if not 0 < metadata["omega_m"] <= 1 or metadata["omega_lambda"] < 0:
        raise ValueError("Unsupported matter/Lambda background")
    if not math.isclose(metadata["omega_m"] + metadata["omega_lambda"], 1.0,
                        rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Only a flat matter-plus-Lambda background is supported")


## @brief Check caller-declared catalogue values without silently accepting misspelled keys.
# @param metadata Actual decoded source metadata.
# @param expected Optional subset of numeric metadata, matched at relative 1e-10.
# @return None; malformed or mismatching declarations raise ValueError.
def _validate_expected(metadata, expected):
    if expected is None:
        return
    allowed = {"total_count", "a", "redshift", "box_mpc_h", "omega_m", "omega_lambda",
               "hubble", "particle_mass_msun_h", "particle_mass_code"}
    for name, value in expected.items():
        if name not in allowed:
            raise ValueError(f"Unknown expected metadata key: {name}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Expected {name} must be a finite numeric value")
        if name == "total_count":
            matches = value == metadata[name]
        else:
            matches = math.isclose(value, metadata[name], rel_tol=1e-10, abs_tol=1e-12)
        if not matches:
            raise ValueError(f"Header {name}={metadata[name]} differs from expected {value}")


## @brief Read a complete canonical header and validate its exact file-length contract.
# @param path IPPLPS01 file, either an ordered IC or an unordered snapshot shard.
# @return JSON-serializable header with file_count,total_count,a,box_mpc_h,
# omega_m,omega_lambda,hubble,particle_mass_msun_h,flags and record/header sizes.
def read_header(path):
    path = Path(path)
    with path.open("rb") as stream:
        data = stream.read(HeaderSize)
    if len(data) != HeaderSize:
        raise ValueError("Truncated canonical phase-space header")
    values = HeaderStruct.unpack(data)
    if values[0] != Magic or any(data[80:]):
        raise ValueError("Unknown canonical magic or nonzero reserved header bytes")
    names = ("file_count", "total_count", "a", "box_mpc_h", "omega_m", "omega_lambda",
             "hubble", "particle_mass_msun_h", "flags")
    metadata = dict(zip(names, values[1:]))
    _validate_background(metadata)
    if metadata["total_count"] < 1 or metadata["file_count"] > metadata["total_count"]:
        raise ValueError("Invalid canonical particle counts")
    if metadata["flags"] not in (0, 1):
        raise ValueError("Unsupported canonical flags")
    if metadata["flags"] == 1 and metadata["file_count"] != metadata["total_count"]:
        raise ValueError("Ordered complete canonical input must contain all particles")
    if path.stat().st_size != HeaderSize + metadata["file_count"] * RecordSize:
        raise ValueError("Canonical file length disagrees with its particle count")
    metadata.update(schema=CanonicalSchema, header_size=HeaderSize, record_size=RecordSize)
    return metadata


## @brief Stream canonical records and validate IDs, finite phase space and half-open positions.
# @param path Canonical binary file. Unordered shards require a global ID audit by the caller.
# @param chunk_size Maximum number of owned records materialized on the host.
# @return Iterator of independent structured RecordDtype chunks.
def iter_records(path, chunk_size=DefaultChunkSize):
    _validate_chunk_size(chunk_size)
    metadata = read_header(path)
    with Path(path).open("rb") as stream:
        stream.seek(HeaderSize)
        for first in range(0, metadata["file_count"], chunk_size):
            count = min(chunk_size, metadata["file_count"] - first)
            records = np.fromfile(stream, dtype=RecordDtype, count=count)
            if len(records) != count:
                raise ValueError("Canonical file changed or was truncated during read")
            ids = records["id"]
            if np.any(ids >= metadata["total_count"]):
                raise ValueError("Canonical particle ID is out of range")
            if metadata["flags"] == 1 and not np.array_equal(ids, np.arange(first, first + count, dtype=np.uint64)):
                raise ValueError("Ordered canonical IDs do not match record indices")
            if not np.isfinite(records["position"]).all() or not np.isfinite(records["momentum"]).all():
                raise ValueError("Nonfinite canonical phase space")
            if np.any(records["position"] < 0) or np.any(records["position"] >= metadata["box_mpc_h"]):
                raise ValueError("Canonical positions are outside [0,L)")
            yield records


## @brief Resolve explicit files or discover numbered Gadget/HDF5 shards deterministically.
# @param inputPaths File/prefix or sequence of actual file paths.
# @return Deterministically ordered resolved paths; numbered coverage is checked separately.
def _resolve_paths(inputPaths):
    if isinstance(inputPaths, (str, Path)):
        candidate = Path(inputPaths)
        if candidate.is_file():
            paths = [candidate]
        else:
            paths = [p for p in candidate.parent.glob(candidate.name + ".*")
                     if p.is_file() and re.fullmatch(r"(?:[0-9]+(?:\.hdf5)?|hdf5)",
                                                     p.name[len(candidate.name) + 1:])]
            paths.sort(key=lambda p: -1 if p.name == candidate.name + ".hdf5"
                       else int(p.name[len(candidate.name) + 1:].split(".")[0]))
    else:
        paths = [Path(p) for p in inputPaths]
    if not paths:
        raise FileNotFoundError("No Gadget format-1 or HDF5 files found")
    resolved = [p.resolve(strict=True) for p in paths]
    if len(set(resolved)) != len(resolved) or any(not p.is_file() for p in resolved):
        raise ValueError("Gadget input paths must be distinct regular files")
    return resolved


## @brief Parse the 256-byte Gadget header, restricting it to fixed-mass collisionless type 1.
# @param data Exactly 256 payload bytes, independent of Fortran record markers.
# @param endian Explicit '<' or '>' byte order from the leading header marker.
# @return Source metadata with local/global counts, epoch, cosmology and physical mass.
def _gadget_header(data, endian):
    counts = struct.unpack_from(endian + "6I", data, 0)
    masses = struct.unpack_from(endian + "6d", data, 24)
    a, redshift = struct.unpack_from(endian + "2d", data, 72)
    low = struct.unpack_from(endian + "6I", data, 96)
    files = struct.unpack_from(endian + "i", data, 124)[0]
    box, om, ol, hubble = struct.unpack_from(endian + "4d", data, 128)
    high = struct.unpack_from(endian + "6I", data, 168)
    totals = tuple(lo + (hi << 32) for lo, hi in zip(low, high))
    return _source_header(counts, masses, a, redshift, totals, files, box, om, ol, hubble)


## @brief Validate common fixed-mass, type-1 scientific metadata independently of storage format.
# @param counts Six exact nonnegative file-local particle counts.
# @param masses Six header masses in 1e10 Msun/h.
# @param a Scale factor recorded by the source.
# @param redshift Source redshift, consistent with a=1/(1+z).
# @param totals Six exact global particle counts, with high words already combined.
# @param files Positive total shard count.
# @param box Periodic side in kpc/h.
# @param om Present matter fraction.
# @param ol Present cosmological-constant fraction.
# @param hubble Dimensionless H0/(100 km/s/Mpc).
# @return Common JSON-compatible metadata; physical mass is retained without density rescaling.
def _source_header(counts, masses, a, redshift, totals, files, box, om, ol, hubble):
    if any(counts[i] or totals[i] for i in (0, 2, 3, 4, 5)):
        raise ValueError("Only one collisionless type-1 particle species is supported")
    if totals[1] < 1 or counts[1] > totals[1] or files < 1:
        raise ValueError("Invalid Gadget global/local particle counts or file count")
    metadata = dict(local_count=counts[1], total_count=totals[1], num_files=files,
                    a=a, redshift=redshift, box_mpc_h=box / 1000.0,
                    omega_m=om, omega_lambda=ol, hubble=hubble,
                    particle_mass_code=masses[1], particle_mass_msun_h=masses[1] * 1e10)
    _validate_background(metadata)
    if not math.isfinite(redshift) or redshift < 0 or not math.isclose(a * (1 + redshift), 1.0, rel_tol=1e-10):
        raise ValueError("Gadget time and redshift are inconsistent")
    return metadata


## @brief Scan an unlabelled Fortran record without materializing its particle payload.
# @param stream Seekable Gadget file positioned before a record marker.
# @param endian Explicit input byte order.
# @param fileSize Exact current file size in bytes.
# @return Payload offset and byte length, after validating both markers and file bounds.
def _record(stream, endian, fileSize):
    marker = stream.read(4)
    if len(marker) != 4:
        raise ValueError("Truncated Gadget record marker")
    size = struct.unpack(endian + "I", marker)[0]
    offset = stream.tell()
    if offset + size + 4 > fileSize:
        raise ValueError("Gadget record extends beyond file end")
    stream.seek(size, os.SEEK_CUR)
    trailer = stream.read(4)
    if trailer != marker:
        raise ValueError("Mismatched Gadget record markers")
    return dict(offset=offset, bytes=size)


## @brief Inspect every split Gadget header/record and freeze source hashes.
# @param inputPaths Explicit path sequence or a numbered-file prefix such as 'ics'.
# @param expected Optional numeric catalogue identity fields; see _validate_expected.
# @return JSON-compatible shared metadata and a files list including offsets/dtypes/SHA256.
# This header pass does not inspect particle values; iter_gadget_chunks performs that check.
def inspect_gadget_files(inputPaths, *, expected=None):
    paths = _resolve_paths(inputPaths)
    files = []
    common = None
    for path in paths:
        fileSize = path.stat().st_size
        with path.open("rb") as stream:
            marker = stream.read(4)
            if marker == struct.pack("<I", 256):
                endian = "<"
            elif marker == struct.pack(">I", 256):
                endian = ">"
            else:
                raise ValueError("Expected Gadget format-1 256-byte header, not HDF5/format-2")
            stream.seek(0)
            headerRecord = _record(stream, endian, fileSize)
            stream.seek(headerRecord["offset"])
            metadata = _gadget_header(stream.read(256), endian)
            stream.seek(264)
            blocks = {}
            for name, width, kind in (("position", 3, "f"), ("velocity", 3, "f"), ("id", 1, "u")):
                block = _record(stream, endian, fileSize)
                count = metadata["local_count"] * width
                precision = block["bytes"] // count if count else 4
                if precision not in (4, 8) or block["bytes"] != count * precision:
                    raise ValueError(f"Unexpected Gadget {name} block length or precision")
                block["dtype"] = endian + kind + str(precision)
                blocks[name] = block
            if stream.tell() != fileSize:
                raise ValueError("Unexpected trailing Gadget blocks; only fixed-mass type-1 phase space is supported")
        shared = {key: value for key, value in metadata.items() if key != "local_count"}
        if common is None:
            common = shared
        elif common != shared:
            raise ValueError("Split Gadget headers disagree on epoch/cosmology/mass/global counts")
        files.append(dict(path=str(path), bytes=fileSize, sha256=sha256(path), input_format="gadget-format1",
                          local_count=metadata["local_count"], endian=endian, blocks=blocks))
    return _finish_source_metadata(common, files, "gadget-format1", expected)


## @brief Validate complete shard coverage and report the equal mass implied by Omega_m.
# @param common Shared scientific header fields, excluding local count.
# @param files Inspected source-file provenance entries.
# @param inputFormat Explicit 'gadget-format1' or 'gadget-hdf5' storage convention.
# @param expected Optional caller declarations checked against the decoded metadata.
# @return Validated metadata with source hashes and a diagnostic physical-mass ratio.
def _finish_source_metadata(common, files, inputFormat, expected):
    if len(files) != common["num_files"] or sum(f["local_count"] for f in files) != common["total_count"]:
        raise ValueError("Incomplete Gadget file coverage or summed particle counts")
    if len(files) > 1:
        suffix = r"(.+)\.([0-9]+)\.hdf5" if inputFormat == "gadget-hdf5" else r"(.+)\.([0-9]+)"
        suffixes = [re.fullmatch(suffix, Path(f["path"]).name) for f in files]
        parents = {str(Path(f["path"]).parent) for f in files}
        if (any(match is None for match in suffixes) or len(parents) != 1
                or len({match.group(1) for match in suffixes}) != 1
                or sorted(int(match.group(2)) for match in suffixes) != list(range(len(files)))):
            raise ValueError("Split Gadget files must cover one prefix numbered 0..num_files-1")
    impliedMass = CriticalDensity * common["omega_m"] * common["box_mpc_h"]**3 / common["total_count"]
    common.update(input_format=inputFormat, implied_mass_ratio=common["particle_mass_msun_h"] / impliedMass,
                  implied_particle_mass_msun_h=impliedMass, files=files)
    _validate_expected(common, expected)
    return common


## @brief Load the optional serial HDF5 reader only for HDF5 inputs.
# @return h5py module, or a precise dependency error that leaves format-1 workflows available.
def _hdf5_runtime():
    try:
        return importlib.import_module("h5py")
    except ImportError as error:
        raise ValueError("HDF5 input requires h5py; install h5py and hdf5plugin for public Quijote ICs") from error


## @brief Read a required numeric HDF5 attribute without silently truncating counts or vectors.
# @param attrs HDF5 header attribute manager.
# @param name Required official Gadget HDF5 attribute name.
# @param shape Exact scalar () or six-species (6,) shape.
# @param integer Require a nonnegative integer dtype for counts, including high words.
# @return Python scalar or list; no HDF5 object escapes its file lifetime.
def _hdf5_attribute(attrs, name, shape=(), *, integer=False):
    if name not in attrs:
        raise ValueError(f"Missing HDF5 header attribute {name}")
    values = np.asarray(attrs[name])
    if values.shape != shape or values.dtype.kind not in ("iu" if integer else "iuf"):
        raise ValueError(f"Invalid HDF5 header attribute shape/type: {name}")
    if integer and np.any(values < 0):
        raise ValueError(f"Negative HDF5 count: {name}")
    return values.tolist()


## @brief Require local hard-linked HDF5 objects so each source hash covers all particle storage.
# @param group Open parent group.
# @param name Immediate child name, not an unresolved external or soft link.
# @param objectType Required h5py Group or Dataset class.
# @param h5py Loaded serial HDF5 module.
# @return Validated object owned by the caller's open file.
def _hdf5_object(group, name, objectType, h5py):
    if not isinstance(group.get(name, getlink=True), h5py.HardLink):
        raise ValueError(f"Missing or nonlocal HDF5 object: {group.name}/{name}")
    obj = group[name]
    if not isinstance(obj, objectType):
        raise ValueError(f"Wrong HDF5 object type: {obj.name}")
    return obj


## @brief Validate only supported lossless filters and bound each decoded HDF5 storage chunk.
# @param dataset Numeric particle dataset already checked for local storage and shape.
# @param h5py Loaded serial HDF5 module.
# @return JSON-compatible filter IDs, flags and client data for provenance.
def _hdf5_filters(dataset, h5py):
    if dataset.is_virtual or dataset.external:
        raise ValueError("External or virtual HDF5 particle storage is unsupported")
    if dataset.chunks and math.prod(dataset.chunks) * dataset.dtype.itemsize > MaxHdf5ChunkBytes:
        raise ValueError("Decoded HDF5 storage chunk exceeds the 64 MiB memory bound")
    filters = []
    creation = dataset.id.get_create_plist()
    for index in range(creation.get_nfilters()):
        filterId, flags, values, name = creation.get_filter(index)
        # Deflate, shuffle, Fletcher32, LZF and the official Blosc filter are lossless.
        if filterId not in (1, 2, 3, 32000, 32001):
            raise ValueError(f"Unsupported or potentially lossy HDF5 filter {filterId}")
        if filterId == 32001 and not h5py.h5z.filter_avail(filterId):
            try:
                importlib.import_module("hdf5plugin")
            except ImportError as error:
                raise ValueError("Blosc-compressed Quijote HDF5 input requires hdf5plugin") from error
        if (not h5py.h5z.filter_avail(filterId)
                or not h5py.h5z.get_filter_info(filterId) & h5py.h5z.FILTER_CONFIG_DECODE_ENABLED):
            raise ValueError(f"No HDF5 decoder is available for filter {filterId}; install hdf5plugin")
        filters.append(dict(id=filterId, flags=flags, values=list(values),
                            name=name.decode("utf-8", errors="replace")))
    return filters


## @brief Validate Quijote's compression provenance without accepting truncated IC coordinates.
# @param stream Open source HDF5 file.
# @param datasets Mapping of official dataset names to already validated datasets.
# @param h5py Loaded serial HDF5 module.
# @return Parsed CompressionInfo JSON or None for an unannotated source.
# The official IC conversion uses truncbits=0 for x,u,IDs; no correction can restore discarded bits.
def _hdf5_compression_info(stream, datasets, h5py):
    if "CompressionInfo" not in stream:
        return None
    group = _hdf5_object(stream, "CompressionInfo", h5py.Group, h5py)
    try:
        info = json.loads(group.attrs["json"])
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        raise ValueError("Invalid HDF5 CompressionInfo JSON") from error
    if not isinstance(info, dict):
        raise ValueError("Invalid HDF5 CompressionInfo mapping")
    for name, dataset in datasets.items():
        options = info.get(name)
        if not isinstance(options, dict) or type(options.get("truncbits")) is not int or options["truncbits"] != 0:
            raise ValueError(f"Lossy or incomplete HDF5 IC compression metadata for {name}")
        try:
            declared = np.dtype(options["hdf5"]["dtype"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid HDF5 compression dtype for {name}") from error
        if declared.kind != dataset.dtype.kind or declared.itemsize != dataset.dtype.itemsize:
            raise ValueError(f"HDF5 compression dtype disagrees with dataset {name}")
    return info


## @brief Inspect exact Gadget HDF5 headers, lossless compression, dataset shapes and source hashes.
# @param inputPaths Explicit HDF5 paths or a prefix such as 'ics' discovering ics.0.hdf5 shards.
# @param expected Optional numeric catalogue declarations; no IC generation or amplitude fitting occurs.
# @return Metadata compatible with the format-1 reader plus HDF5 dataset/filter/compression provenance.
# @details Coordinates are kpc/h, Velocities are raw u=v_pec/sqrt(a), MassTable is 1e10 Msun/h.
# Conventions follow https://github.com/lgarrison/quijote-compression/blob/main/compress_gadget.py.
def inspect_hdf5_files(inputPaths, *, expected=None):
    h5py = _hdf5_runtime()
    files = []
    common = None
    for path in _resolve_paths(inputPaths):
        fileSize = path.stat().st_size
        with h5py.File(path, "r", rdcc_nbytes=8 * 1024 * 1024) as stream:
            header = _hdf5_object(stream, "Header", h5py.Group, h5py).attrs
            counts = _hdf5_attribute(header, "NumPart_ThisFile", (6,), integer=True)
            low = _hdf5_attribute(header, "NumPart_Total", (6,), integer=True)
            high = _hdf5_attribute(header, "NumPart_Total_HighWord", (6,), integer=True)
            if any(word > 0xFFFFFFFF for word in low + high):
                raise ValueError("HDF5 global-count words exceed uint32")
            totals = [lo + (hi << 32) for lo, hi in zip(low, high)]
            metadata = _source_header(counts, _hdf5_attribute(header, "MassTable", (6,)),
                _hdf5_attribute(header, "Time"), _hdf5_attribute(header, "Redshift"), totals,
                _hdf5_attribute(header, "NumFilesPerSnapshot", integer=True),
                _hdf5_attribute(header, "BoxSize"), _hdf5_attribute(header, "Omega0"),
                _hdf5_attribute(header, "OmegaLambda"), _hdf5_attribute(header, "HubbleParam"))
            for name in stream:
                if name.startswith("PartType") and name != "PartType1":
                    group = _hdf5_object(stream, name, h5py.Group, h5py)
                    if len(group):
                        raise ValueError("Only one collisionless type-1 particle species is supported")
            particles = _hdf5_object(stream, "PartType1", h5py.Group, h5py)
            if set(particles) != {"Coordinates", "Velocities", "ParticleIDs"}:
                raise ValueError("HDF5 type-1 group must contain only Coordinates, Velocities and ParticleIDs")
            datasets = {}
            blocks = {}
            for name, field, width, kind in (("Coordinates", "position", 3, "f"),
                                             ("Velocities", "velocity", 3, "f"),
                                             ("ParticleIDs", "id", 1, "u")):
                dataset = _hdf5_object(particles, name, h5py.Dataset, h5py)
                shape = (metadata["local_count"], 3) if width == 3 else (metadata["local_count"],)
                if dataset.shape != shape or dataset.dtype.kind != kind or dataset.dtype.itemsize not in (4, 8):
                    raise ValueError(f"Invalid HDF5 {name} shape or dtype")
                blocks[field] = dict(dataset=dataset.name, shape=list(shape), dtype=dataset.dtype.str,
                                     chunks=list(dataset.chunks) if dataset.chunks else None,
                                     filters=_hdf5_filters(dataset, h5py))
                datasets[name] = dataset
            compression = _hdf5_compression_info(stream, datasets, h5py)
        shared = {key: value for key, value in metadata.items() if key != "local_count"}
        if common is None:
            common = shared
        elif common != shared:
            raise ValueError("Split Gadget HDF5 headers disagree on epoch/cosmology/mass/global counts")
        files.append(dict(path=str(path), bytes=fileSize, sha256=sha256(path), input_format="gadget-hdf5",
                          local_count=metadata["local_count"], blocks=blocks, compression_info=compression))
    return _finish_source_metadata(common, files, "gadget-hdf5", expected)


## @brief Select an explicit reader from on-disk signatures and reject mixed input formats.
# @param inputPaths Explicit source files or a numbered Gadget/HDF5 prefix.
# @param expected Optional source identity checks forwarded to the selected reader.
# @return Fully validated, hashed metadata tagged with input_format.
def inspect_source_files(inputPaths, *, expected=None):
    paths = _resolve_paths(inputPaths)
    signatures = []
    for path in paths:
        with path.open("rb") as stream:
            signatures.append(stream.read(8) == Hdf5Magic)
    if any(signatures) and not all(signatures):
        raise ValueError("Mixed HDF5 and Gadget format-1 source files")
    reader = inspect_hdf5_files if all(signatures) else inspect_gadget_files
    return reader(paths, expected=expected)


## @brief Stream bounded source chunks, converting exact Gadget units and one-based IDs.
# @param inputPaths Input paths/prefix or metadata returned by inspect_gadget_files.
# @param chunk_size Maximum rows held per source chunk.
# @return Iterator of (zero-based uint64 IDs, xyz Mpc/h, canonical momenta Mpc/h).
# Position L wraps exactly to zero; outside [0,L] is rejected, never perturbed.
def iter_gadget_chunks(inputPaths, chunk_size=DefaultChunkSize):
    _validate_chunk_size(chunk_size)
    metadata = inputPaths if isinstance(inputPaths, dict) else inspect_gadget_files(inputPaths)
    for source in metadata["files"]:
        with Path(source["path"]).open("rb") as stream:
            if os.fstat(stream.fileno()).st_size != source["bytes"]:
                raise ValueError("Gadget file length changed after inspection")
            for first in range(0, source["local_count"], chunk_size):
                count = min(chunk_size, source["local_count"] - first)
                arrays = {}
                for name, width in (("position", 3), ("velocity", 3), ("id", 1)):
                    block = source["blocks"][name]
                    dtype = np.dtype(block["dtype"])
                    stream.seek(block["offset"] + first * width * dtype.itemsize)
                    array = np.fromfile(stream, dtype=dtype, count=count * width)
                    if len(array) != count * width:
                        raise ValueError("Gadget file truncated during particle read")
                    arrays[name] = array.reshape(count, width) if width == 3 else array
                yield _canonical_chunk(arrays, metadata)


## @brief Apply identical physical conversions to raw Gadget format-1 and HDF5 source chunks.
# @param arrays Source id,position,velocity arrays, with validated unsigned and floating dtypes.
# @param metadata Shared source epoch, global count and periodic side.
# @return Zero-based IDs, x in Mpc/h and p=a^(3/2)u/100 in Mpc/h; only x=L wraps to zero.
def _canonical_chunk(arrays, metadata):
    ids = arrays["id"].astype(np.uint64)
    if np.any(ids == 0) or np.any(ids > metadata["total_count"]):
        raise ValueError("Quijote particle IDs must lie in the one-based interval [1,N]")
    positions = arrays["position"].astype(np.float64) / 1000.0
    momenta = arrays["velocity"].astype(np.float64) * (metadata["a"]**1.5 / 100.0)
    if not np.isfinite(positions).all() or not np.isfinite(momenta).all():
        raise ValueError("Nonfinite Gadget positions or velocities")
    box = metadata["box_mpc_h"]
    if np.any(positions < 0) or np.any(positions > box):
        raise ValueError("Gadget positions are outside [0,L]; check units and box header")
    positions[positions == box] = 0.0
    return ids - np.uint64(1), positions, momenta


## @brief Slice HDF5 datasets serially without materializing complete particle arrays.
# @param inputPaths Explicit source paths/prefix or metadata returned by inspect_hdf5_files.
# @param chunk_size Maximum particles in each returned host chunk.
# @return Iterator of zero-based IDs, comoving coordinates and canonical momentum.
# HDF5 storage chunks are independently limited by MaxHdf5ChunkBytes during inspection.
def iter_hdf5_chunks(inputPaths, chunk_size=DefaultChunkSize):
    _validate_chunk_size(chunk_size)
    metadata = inputPaths if isinstance(inputPaths, dict) else inspect_hdf5_files(inputPaths)
    h5py = _hdf5_runtime()
    for source in metadata["files"]:
        if Path(source["path"]).stat().st_size != source["bytes"]:
            raise ValueError("HDF5 file length changed after inspection")
        with h5py.File(source["path"], "r", rdcc_nbytes=8 * 1024 * 1024) as stream:
            datasets = {}
            for name, block in source["blocks"].items():
                dataset = stream[block["dataset"]]
                if dataset.shape != tuple(block["shape"]) or dataset.dtype != np.dtype(block["dtype"]):
                    raise ValueError("HDF5 dataset shape or dtype changed after inspection")
                # A caller can deserialize inspected metadata in a fresh process; register its decoder here too.
                _hdf5_filters(dataset, h5py)
                datasets[name] = dataset
            for first in range(0, source["local_count"], chunk_size):
                stop = min(first + chunk_size, source["local_count"])
                arrays = {name: dataset[first:stop] for name, dataset in datasets.items()}
                yield _canonical_chunk(arrays, metadata)


## @brief Stream either supported source format through its explicitly tagged metadata.
# @param inputPaths Source files/prefix or metadata from inspect_source_files.
# @param chunk_size Maximum particles in each source chunk.
# @return Iterator with the same canonical arrays for both supported file formats.
def iter_source_chunks(inputPaths, chunk_size=DefaultChunkSize):
    _validate_chunk_size(chunk_size)
    metadata = inputPaths if isinstance(inputPaths, dict) else inspect_source_files(inputPaths)
    if metadata.get("input_format", "gadget-format1") == "gadget-format1":
        yield from iter_gadget_chunks(metadata, chunk_size=chunk_size)
    elif metadata["input_format"] == "gadget-hdf5":
        yield from iter_hdf5_chunks(metadata, chunk_size=chunk_size)
    else:
        raise ValueError("Unsupported source input_format")


## @brief Convert a complete source set using disk-backed reordering and exact duplicate checks.
# @param inputPaths Gadget format-1/HDF5 files or prefix; bytes are hashed before and after conversion.
# @param output Fresh canonical binary destination; sidecar is output+'.json'.
# @param chunk_size Bounded particle chunk; the O(N) reorder/seen maps are disk backed.
# @param expected Optional source identity assertions, including fiducial cosmology/count/epoch.
# @return Manifest with source hashes, unit transformations, canonical header and output hash.
# Exact same-chunk and inter-chunk duplicate detection plus N observed IDs proves completeness.
def convert(inputPaths, output, *, chunk_size=DefaultChunkSize, expected=None):
    _validate_chunk_size(chunk_size)
    output = Path(output).resolve()
    sidecar = output.with_suffix(output.suffix + ".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("Refusing to overwrite canonical phase space or its manifest")
    metadata = inspect_source_files(inputPaths, expected=expected)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = metadata["total_count"]
    header = HeaderStruct.pack(Magic, count, count, metadata["a"], metadata["box_mpc_h"],
                               metadata["omega_m"], metadata["omega_lambda"], metadata["hubble"],
                               metadata["particle_mass_msun_h"], 1)
    published = []
    with tempfile.TemporaryDirectory(prefix=".quijote-convert-", dir=output.parent) as tempDir:
        tempDir = Path(tempDir)
        temporary = tempDir / "phase-space.bin"
        with temporary.open("xb") as stream:
            stream.write(header)
            stream.truncate(HeaderSize + count * RecordSize)
        seenPath = tempDir / "seen.bin"
        with seenPath.open("xb") as stream:
            stream.truncate(count)
        records = np.memmap(temporary, mode="r+", dtype=RecordDtype, offset=HeaderSize, shape=(count,))
        seen = np.memmap(seenPath, mode="r+", dtype=np.uint8, shape=(count,))
        try:
            observed = 0
            for ids, positions, momenta in iter_source_chunks(metadata, chunk_size=chunk_size):
                if np.unique(ids).size != len(ids) or np.any(seen[ids]):
                    raise ValueError("Duplicate Gadget particle IDs; canonical import requires each ID exactly once")
                records["id"][ids] = ids
                records["position"][ids] = positions
                records["momentum"][ids] = momenta
                seen[ids] = 1
                observed += len(ids)
            if observed != count:
                raise ValueError("Missing Gadget particle IDs")
            for first in range(0, count, chunk_size):
                if not np.all(seen[first:first + chunk_size]):
                    raise ValueError("Missing Gadget particle IDs")
            records.flush()
        finally:
            records._mmap.close()
            seen._mmap.close()
        for source in metadata["files"]:
            if Path(source["path"]).stat().st_size != source["bytes"] or sha256(source["path"]) != source["sha256"]:
                raise ValueError("Gadget source changed during conversion")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        outputHeader = read_header(temporary)
        manifest = dict(schema=Hdf5Schema if metadata["input_format"] == "gadget-hdf5" else Schema,
                        initial_conditions={key: value for key, value in metadata.items() if key != "files"},
                        sources=metadata["files"],
                        output=dict(path=str(output), sha256=sha256(temporary), bytes=temporary.stat().st_size,
                                    header=outputHeader),
                        units=dict(position="Mpc/h; x=x_Gadget[kpc/h]/1000",
                                   momentum="Mpc/h; p=a^(3/2)*u_Gadget/100; u_Gadget=v_pec/sqrt(a)",
                                   physical_mass="Msun/h; Gadget header mass multiplied by 1e10",
                                   ids="source one-based IDs mapped exactly to zero-based IDs",
                                   ic_generation="none; positions and velocities imported without growth rescaling",
                                   particle_weights="equal; physical mass retained in header"),
                        validation=dict(unique_complete_ids=True, source_hashes_unchanged=True,
                                        implied_mass_ratio_is_diagnostic=True, chunk_size=chunk_size))
        tempManifest = tempDir / "manifest.json"
        with tempManifest.open("x") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Exclusive links publish complete files without replacing earlier evidence.
            os.link(temporary, output)
            published.append(output)
            os.link(tempManifest, sidecar)
            published.append(sidecar)
        except BaseException:
            for path in reversed(published):
                path.unlink()
            raise
    return manifest


## @brief Convert supplied format-1/HDF5 files offline; no download or simulation is launched.
# @return Zero on successful conversion; argparse reports invalid input as an error.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Gadget format-1/HDF5 files, or one numbered prefix")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=DefaultChunkSize)
    parser.add_argument("--expect-count", type=int)
    parser.add_argument("--expect-box", type=float, help="Expected comoving box in Mpc/h")
    parser.add_argument("--expect-a", type=float)
    parser.add_argument("--expect-omega-m", type=float)
    parser.add_argument("--expect-hubble", type=float)
    args = parser.parse_args()
    expected = {key: value for key, value in {
        "total_count": args.expect_count, "box_mpc_h": args.expect_box, "a": args.expect_a,
        "omega_m": args.expect_omega_m, "hubble": args.expect_hubble}.items() if value is not None}
    try:
        manifest = convert(args.inputs[0] if len(args.inputs) == 1 else args.inputs,
                           args.output, chunk_size=args.chunk_size, expected=expected)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(manifest["output"], indent=2))
    return 0


## @cond CLI_DISPATCH
if __name__ == "__main__":
    raise SystemExit(main())
## @endcond
