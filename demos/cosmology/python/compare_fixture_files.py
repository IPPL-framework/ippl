#!/usr/bin/env python3
## @file compare_fixture_files.py
# @brief Read-only cross-host comparison of common-particle cosmology fixture CSVs.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
"""Read-only cross-host comparison of common-particle cosmology fixture CSVs.

Example::

    python -B compare_fixture_files.py original.csv remote.csv \
        --box-size 168.75 --output fixture-parity.json

No physical tolerance or qualification is applied. Exit zero means valid
inputs were compared and the report was written, not that the files agree.
Output is created exclusively; existing reports and input CSVs are untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct

import numpy as np
import pandas as pd


## @var Columns
# @brief Named Columns protocol/schema value; the source initializer records its exact contents.
Columns = ["id", "x", "y", "z", "px", "py", "pz", "mass"]
## @var ValueColumns
# @brief Named ValueColumns protocol/schema value; the source initializer records its exact contents.
ValueColumns = Columns[1:]
## @var DigestDomain
# @brief Named DigestDomain protocol/schema value; the source initializer records its exact contents.
DigestDomain = b"cosmology-fixture-sorted-u64-f64-v1\0"


## @brief Compute the streaming SHA256 digest of the file bytes.
# @see cosmology_tools
#
# @param path Input/output filesystem path; freshness and hash requirements are defined by this routine.
# @return Lowercase hexadecimal SHA256 of the exact retained bytes.
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


## @brief Evaluate the particle grid helper in the documented module workflow.
# @see cosmology_tools
#
# @param count Expected number of particles/elements, or byte-count context explicitly declared by the routine.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def particle_grid(count):
    if count < 1:
        raise ValueError("Fixture must not be empty")
    grid = round(count**(1 / 3))
    if grid**3 != count:
        raise ValueError("Fixture particle count must be a perfect cube")
    return grid


## @brief Digest sorted exact values with fixed little-endian, row-major encoding.
# @see cosmology_tools
#
# @param frame Particle DataFrame with the exact columns/ID/finite-state contract checked by this routine.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def canonical_digest(frame):
    """Digest sorted exact values with fixed little-endian, row-major encoding."""
    digest = hashlib.sha256(DigestDomain)
    digest.update(struct.pack("<Q", len(frame)))
    digest.update(frame.id.to_numpy(dtype="<u8").tobytes())
    digest.update(frame[ValueColumns].to_numpy(dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


## @brief Read and verify fixture.
# @see cosmology_tools
#
# @param path Input/output filesystem path; freshness and hash requirements are defined by this routine.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def read_fixture(path):
    path = Path(path).resolve()
    before = sha256(path)
    # Parse IDs as strings first: never accept float-rounded, negative, or
    # silently wrapped identifiers. All other values use round-trip float64.
    frame = pd.read_csv(path, dtype={"id": "string", **{name: np.float64 for name in ValueColumns}},
                        float_precision="round_trip")
    if list(frame.columns) != Columns:
        raise ValueError(f"{path}: require exact columns {','.join(Columns)}")
    grid = particle_grid(len(frame))
    if not frame.id.str.fullmatch(r"[0-9]+").fillna(False).all():
        raise ValueError(f"{path}: IDs must be unsigned decimal integers")
    ids = [int(value) for value in frame.id]
    if any(value > np.iinfo(np.uint64).max for value in ids):
        raise ValueError(f"{path}: ID exceeds uint64 range")
    frame["id"] = np.asarray(ids, dtype=np.uint64)
    frame = frame.sort_values("id").reset_index(drop=True)
    if not np.array_equal(frame.id.to_numpy(), np.arange(len(frame), dtype=np.uint64)):
        raise ValueError(f"{path}: IDs must contain every integer in [0,NP^3) exactly once")
    if not np.isfinite(frame[ValueColumns].to_numpy()).all():
        raise ValueError(f"{path}: position, momentum and mass must be finite")
    if not (frame.mass == 1.).all():
        raise ValueError(f"{path}: common-particle comparison requires unit mass")
    if sha256(path) != before:
        raise ValueError(f"{path}: CSV changed during read")
    metadata = {"path": str(path), "csv_sha256": before, "csv_bytes": path.stat().st_size,
                "particles": len(frame), "particle_grid": grid,
                "canonical_value_sha256": canonical_digest(frame)}
    return frame, metadata


## @brief sqrt(mean(sum_d(value_d**2))), with scaling to avoid intermediate overflow.
# @see cosmology_tools
#
# @param values Recorded diagnostic values in the metric/schema defined by the caller.
# @return Vector RMS in the input array's units.
def vector_rms(values):
    """sqrt(mean(sum_d(value_d**2))), with scaling to avoid intermediate overflow."""
    values = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(abs(values)))
    if maximum == 0.:
        return 0.
    result = maximum * float(np.sqrt(np.mean(np.sum((values / maximum)**2, axis=1))))
    if not math.isfinite(result):
        raise ValueError("Vector RMS exceeds finite float64 range")
    return result


## @brief Evaluate the residual metrics helper in the documented module workflow.
# @see cosmology_tools
#
# @param delta Density or difference array in the metric's explicit convention; see the caller for normalization.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def residual_metrics(delta):
    maximum = float(np.max(abs(delta)))
    largest_norm = (maximum * float(np.sqrt(np.max(np.sum((delta / maximum)**2, axis=1))))
                    if maximum else 0.)
    if not math.isfinite(largest_norm):
        raise ValueError("Vector residual exceeds finite float64 range")
    return {"vector_rms": vector_rms(delta), "maximum_particle_norm": largest_norm,
            "maximum_absolute_component": maximum,
            "nonzero_component_count": int(np.count_nonzero(delta)),
            "nonzero_particle_count": int(np.count_nonzero(np.any(delta != 0., axis=1)))}


## @brief Compare fixtures.
# @see cosmology_tools
#
# @param original_path Original/local CSV or artifact path used as the declared comparison reference.
# @param comparison_path Second-host/code CSV or artifact path whose identity and contents are checked.
# @param box_size Positive periodic comoving box side in Mpc/h.
# @param cell_grid Positive grid side used only to report position residuals in cell units.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def compare_fixtures(original_path, comparison_path, box_size, cell_grid=None):
    if not math.isfinite(box_size) or box_size <= 0:
        raise ValueError("box_size must be finite and positive")
    left, left_info = read_fixture(original_path)
    right, right_info = read_fixture(comparison_path)
    if len(left) != len(right):
        raise ValueError("Fixtures must have the same particle count and IDs")
    if cell_grid is None:
        cell_grid = left_info["particle_grid"]
        cell_basis = "particle lattice spacing (default); not a simulation mesh assumption"
    else:
        cell_basis = "explicit --cell-grid"
    if isinstance(cell_grid, bool) or not isinstance(cell_grid, (int, np.integer)) or cell_grid < 1:
        raise ValueError("cell_grid must be a positive integer")
    cell_size = box_size / cell_grid
    if not math.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("Cell spacing is not representable as a positive finite value")
    left_values = left[ValueColumns].to_numpy(dtype="<f8")
    right_values = right[ValueColumns].to_numpy(dtype="<f8")
    left_bits, right_bits = left_values.view("<u8"), right_values.view("<u8")
    # Wrap before subtraction so valid multiple-box coordinates do not need
    # enormous differences. Positions always compare through minimum images.
    left_x = np.remainder(left_values[:, :3], box_size)
    right_x = np.remainder(right_values[:, :3], box_size)
    with np.errstate(over="raise", invalid="raise"):
        dx = right_x - left_x
        dx -= box_size * np.rint(dx / box_size)
        dp = right_values[:, 3:6] - left_values[:, 3:6]
    position, momentum = residual_metrics(dx), residual_metrics(dp)
    position["vector_rms_cells"] = position["vector_rms"] / cell_size
    position["maximum_particle_norm_cells"] = position["maximum_particle_norm"] / cell_size
    reference_rms = vector_rms(left_values[:, 3:6])
    momentum.update(original_vector_rms=reference_rms, comparison_vector_rms=vector_rms(right_values[:, 3:6]),
                    relative_vector_rms=momentum["vector_rms"] / reference_rms if reference_rms else None,
                    normalization_status="defined" if reference_rms else "undefined_zero_original_momentum")
    report = {"schema": "cosmology-fixture-comparison-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_complete": True, "physical_qualification": "not assessed; no tolerance or physical pass/fail applied",
        "original": left_info, "comparison": right_info,
        "geometry": {"box_size_mpc_over_h": box_size, "cell_grid": int(cell_grid),
                     "cell_size_mpc_over_h": cell_size, "cell_normalization": cell_basis},
        "agreement": {"csv_bytes_identical": left_info["csv_sha256"] == right_info["csv_sha256"],
                      "same_resolved_path": left_info["path"] == right_info["path"],
                      "sorted_numeric_values_equal": bool(np.array_equal(left_values, right_values)),
                      "sorted_numeric_bits_identical": bool(np.array_equal(left_bits, right_bits)),
                      "different_numeric_components": int(np.count_nonzero(left_values != right_values)),
                      "different_bitwise_components": int(np.count_nonzero(left_bits != right_bits))},
        "position_difference": position, "momentum_difference": momentum,
        "conventions": {"particle_matching": "complete unique uint64 IDs, sorted before comparison; unit masses",
            "canonical_value_digest": "SHA256(domain, particle count u64 LE, sorted IDs u64 LE, x,y,z,px,py,pz,mass float64 LE row-major)",
            "canonical_digest_domain": DigestDomain.decode("ascii"),
            "difference_direction": "comparison minus original",
            "positions": "Mpc/h; minimum-image periodic difference; half-box ties use numpy.rint",
            "momenta": "Mpc/h; canonical p=a^2 dx/d(H0*t); unwrapped direct difference",
            "vector_rms": "sqrt(mean_particles(sum_components(value**2)))",
            "momentum_relative": "RMS(dp) / RMS(p_original); null if original RMS is zero",
            "bitwise_values": "Sorted parsed float64 values, including mass and signed-zero bits; independent of CSV formatting"},
        "tool": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__),
                 "numpy_version": np.__version__, "pandas_version": pd.__version__}}
    # No custom tolerance, fitted threshold, or implicit physical success flag.
    json.dumps(report, allow_nan=False)
    return report


## @brief Write report.
# @see cosmology_tools
#
# @param report Structured campaign/audit report; recorded failures are not retroactively changed.
# @param output Output path; use a fresh destination where the workflow rejects existing evidence.
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def write_report(report, output):
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with Path(output).open("x") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


## @brief Parse the documented command-line interface and execute its selected workflow, retaining nonzero failures.
# @see cosmology_tools
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path, help="Original/local CSV; denominator for relative momentum")
    parser.add_argument("comparison", type=Path, help="Other/remote CSV to compare")
    parser.add_argument("--box-size", type=float, required=True, help="Periodic box side in Mpc/h")
    parser.add_argument("--cell-grid", type=int, help="Grid side for dx/cell; default cubic particle lattice side")
    parser.add_argument("--output", type=Path, required=True, help="New JSON path; parent directory must exist")
    args = parser.parse_args()
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        report = compare_fixtures(args.original, args.comparison, args.box_size, args.cell_grid)
        write_report(report, args.output)
    except (OSError, ValueError, OverflowError, FloatingPointError) as error:
        print(f"Fixture comparison failed: {error}")
        return 1
    print(f"Comparison report: {args.output.resolve()}")
    print("CSV identical:", report["agreement"]["csv_bytes_identical"],
          "; sorted numeric bits identical:", report["agreement"]["sorted_numeric_bits_identical"])
    print("No physical acceptance criterion was applied.")
    return 0


## @cond CLI_DISPATCH
if __name__ == "__main__":
    raise SystemExit(main())
## @endcond
