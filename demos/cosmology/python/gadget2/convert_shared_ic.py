#!/usr/bin/env python3
## @file convert_shared_ic.py
# @brief Convert the frozen shared cosmology CSV IC to GADGET-2 format-1.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
# @see cosmology_model cosmology_numerics cosmology_spectra
"""Convert the frozen shared cosmology CSV IC to GADGET-2 format-1.

The physical phase-space realization is not regenerated. Positions are
converted from Mpc/h to kpc/h; GADGET-2 format-1 velocities are u=v_pec/sqrt(a)
in km/s, whereas the source CSV stores p=a^2 dx/d(H0 t), with
v_pec=100 p/a km/s. The standard IC reader then multiplies u by a^(3/2) to
obtain GADGET-2's internal canonical velocity variable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np


## @var Schema
# @brief Named Schema protocol/schema value; the source initializer records its exact contents.
Schema = "ippl-gadget2-format1-ic-v1"
## @var UnitLengthCm
# @brief Named UnitLengthCm protocol/schema value; the source initializer records its exact contents.
UnitLengthCm = 3.085678e21       # kpc/h
## @var UnitMassGrams
# @brief Named UnitMassGrams protocol/schema value; the source initializer records its exact contents.
UnitMassGrams = 1.989e43         # 1e10 Msun/h
## @var UnitVelocityCmPerSec
# @brief Named UnitVelocityCmPerSec protocol/schema value; the source initializer records its exact contents.
UnitVelocityCmPerSec = 1.0e5     # km/s
## @var CriticalDensityMsunH2PerMpc3
# @brief Named CriticalDensityMsunH2PerMpc3 protocol/schema value; the source initializer records its exact contents.
CriticalDensityMsunH2PerMpc3 = 2.77536627e11


## @brief Compute the streaming SHA256 digest of the file bytes.
# @see cosmology_tools
#
# @param path Input/output filesystem path; freshness and hash requirements are defined by this routine.
# @return Lowercase hexadecimal SHA256 of the exact retained bytes.
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


## @brief Encode the 256-byte GADGET format-1 header for one collisionless type-1 particle species.
# @see cosmology_tools
#
# @param count Expected number of particles/elements, or byte-count context explicitly declared by the routine.
# @param mass_code Physical GADGET particle mass in code units of 10^10 Msun/h.
# @param a Dimensionless scale factor; must satisfy the calling background/epoch contract.
# @param box_kpc_h Positive periodic box side in kpc/h, as stored in the GADGET header.
# @param omega_m Matter density fraction at a=1; no radiation or neutrino species is added.
# @param omega_lambda Cosmological-constant fraction, equal to one minus Omega_m in the supported flat model.
# @param hubble Dimensionless h=H0/(100 km/s/Mpc); the comoving length convention already includes h^-1.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def gadget_header(*, count: int, mass_code: float, a: float, box_kpc_h: float,
                  omega_m: float, omega_lambda: float, hubble: float) -> bytes:
    npart = [0, count, 0, 0, 0, 0]  # type 1 is collisionless dark matter
    masses = [0.0, mass_code, 0.0, 0.0, 0.0, 0.0]
    fields = [
        struct.pack("<6i", *npart),
        struct.pack("<6d", *masses),
        struct.pack("<2d", a, 1.0 / a - 1.0),
        struct.pack("<2i", 0, 0),
        struct.pack("<6I", *npart),
        struct.pack("<2i", 0, 1),
        struct.pack("<4d", box_kpc_h, omega_m, omega_lambda, hubble),
        struct.pack("<2i", 0, 0),
        struct.pack("<6I", 0, 0, 0, 0, 0, 0),
        struct.pack("<i", 0),
        bytes(60),
    ]
    result = b"".join(fields)
    if len(result) != 256:
        raise AssertionError(f"GADGET-2 header is {len(result)} bytes, expected 256")
    return result


## @brief Write a Fortran-style byte record with matching signed 32-bit leading/trailing sizes.
# @see cosmology_tools
#
# @param stream Open binary or text stream following the routine's declared record/output contract.
# @param data Finite numerical or serialized record data in the routine's explicit schema.
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def write_record(stream, data: bytes) -> None:
    if len(data) > 0x7FFFFFFF:
        raise ValueError("GADGET-2 format-1 record exceeds signed 32-bit length")
    marker = struct.pack("<i", len(data))
    stream.write(marker)
    stream.write(data)
    stream.write(marker)


## @brief Convert the shared canonical CSV to GADGET units and float32 blocks, retaining a quantified round-trip audit.
# @see cosmology_tools
#
# @param input_csv Shared ordered canonical phase-space CSV; positions/momenta are in Mpc/h and masses are unit weights.
# @param output Output path; use a fresh destination where the workflow rejects existing evidence.
# @param redshift Initialization redshift; a=1/(1+redshift).
# @param expected_sha256 Expected SHA256 of the frozen shared initial CSV.
# @param box_mpc_h Positive periodic comoving box side in Mpc/h.
# @param omega_m Matter density fraction at a=1; no radiation or neutrino species is added.
# @param hubble Dimensionless h=H0/(100 km/s/Mpc); the comoving length convention already includes h^-1.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def convert(input_csv: Path, output: Path, *, redshift: float,
            expected_sha256: str, box_mpc_h: float = 168.75,
            omega_m: float = 0.31, hubble: float = 0.675) -> dict:
    input_csv = input_csv.resolve(strict=True)
    output = output.resolve()
    sidecar = output.with_suffix(output.suffix + ".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("Refusing to overwrite a GADGET IC or its sidecar")
    if not math.isfinite(redshift) or redshift < 0:
        raise ValueError("redshift must be finite and nonnegative")
    if not math.isfinite(box_mpc_h) or box_mpc_h <= 0:
        raise ValueError("box size must be positive and finite")
    if not math.isfinite(omega_m) or not 0 < omega_m < 1:
        raise ValueError("omega_m must be between zero and one")
    if not math.isfinite(hubble) or hubble <= 0:
        raise ValueError("hubble parameter must be positive and finite")

    input_hash = sha256(input_csv)
    if input_hash != expected_sha256:
        raise ValueError("shared IC CSV hash does not match the frozen campaign manifest")

    table = np.loadtxt(input_csv, delimiter=",", skiprows=1, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != 8:
        raise ValueError("expected id,x,y,z,px,py,pz,mass CSV with eight columns")
    if not np.isfinite(table).all():
        raise ValueError("source CSV contains nonfinite values")
    count = table.shape[0]
    if count <= 0 or count > 0x7FFFFFFF:
        raise ValueError("particle count is outside GADGET-2's supported format-1 range")
    if not np.array_equal(table[:, 0], np.arange(count, dtype=np.float64)):
        raise ValueError("input particle IDs must be unique, ordered, contiguous uint32 values")
    if not np.all(table[:, 7] == 1.0):
        raise ValueError("source CSV mass weights must all equal one")
    particle_grid = round(count ** (1.0 / 3.0))
    if particle_grid**3 != count:
        raise ValueError("expected a cubic particle lattice so mesh-cell errors are well-defined")
    phase = table[:, 1:7]
    if not np.isfinite(phase).all():
        raise ValueError("source phase space contains nonfinite values")
    if np.any((phase[:, :3] < 0) | (phase[:, :3] >= box_mpc_h)):
        raise ValueError("source positions are not wrapped into the periodic box")

    a = 1.0 / (1.0 + redshift)
    omega_lambda = 1.0 - omega_m
    # Gadget's documented default unit system: kpc/h, 1e10 Msun/h, km/s.
    mass_code = (CriticalDensityMsunH2PerMpc3 * omega_m * box_mpc_h**3
                 / count / 1.0e10)
    positions_kpc_h = (phase[:, :3] * 1000.0).astype("<f4")
    velocity_pec_km_s = (100.0 * phase[:, 3:6] / a).astype(np.float64)
    velocity_gadget = (velocity_pec_km_s / math.sqrt(a)).astype("<f4")
    ids = np.arange(count, dtype="<u4")
    header = gadget_header(count=count, mass_code=mass_code, a=a,
                           box_kpc_h=box_mpc_h * 1000.0, omega_m=omega_m,
                           omega_lambda=omega_lambda, hubble=hubble)

    with output.open("xb") as stream:
        write_record(stream, header)
        write_record(stream, positions_kpc_h.tobytes(order="C"))
        write_record(stream, velocity_gadget.tobytes(order="C"))
        write_record(stream, ids.tobytes(order="C"))

    # Recover the source convention from exactly the float32 data on disk.
    position_recovered = positions_kpc_h.astype(np.float64) / 1000.0
    momentum_recovered = (velocity_gadget.astype(np.float64) * a**1.5 / 100.0)
    position_delta = position_recovered - phase[:, :3]
    position_delta -= box_mpc_h * np.rint(position_delta / box_mpc_h)
    cell = box_mpc_h / particle_grid
    position_rms_cells = float(np.sqrt(np.mean(np.sum(position_delta**2, axis=1))) / cell)
    # Standard GADGET-2 POS blocks are float32 even in a double-precision build.
    # The largest possible three-component RMS round-trip bound is set by half
    # an ulp per component at the periodic-box scale.
    position_limit = float(math.sqrt(3.0) * np.finfo(np.float32).eps
                           * particle_grid / 2.0)
    input_p_norm = float(np.sqrt(np.mean(np.sum(phase[:, 3:6]**2, axis=1))))
    if input_p_norm <= 0:
        output.unlink(missing_ok=True)
        raise ValueError("source momentum field has zero RMS; cannot certify relative round-trip error")
    momentum_rms_relative = float(
        np.sqrt(np.mean(np.sum((momentum_recovered - phase[:, 3:6])**2, axis=1))) / input_p_norm)
    if not math.isfinite(position_rms_cells + momentum_rms_relative):
        output.unlink(missing_ok=True)
        raise ValueError("round-trip errors are nonfinite")

    report = {
        "schema": Schema,
        "input": {"path": str(input_csv), "sha256": input_hash,
                  "bytes": input_csv.stat().st_size},
        "output": {"path": str(output), "sha256": sha256(output),
                   "bytes": output.stat().st_size, "format": 1},
        "initial_conditions": {"particle_count": count, "particle_type": 1,
            "a": a, "redshift": redshift, "box_mpc_over_h": box_mpc_h,
            "box_kpc_over_h": box_mpc_h * 1000.0, "omega_m": omega_m,
            "omega_lambda": omega_lambda, "hubble_param": hubble,
            "mass_code_units_1e10_msun_over_h": mass_code,
            "unit_length_cm": UnitLengthCm, "unit_mass_g": UnitMassGrams,
            "unit_velocity_cm_per_s": UnitVelocityCmPerSec,
            "position_transform": "x_Gadget[kpc/h]=1000*x_CSV[Mpc/h]",
            "velocity_transform": "u_Gadget[km/s]=100*p_CSV/a^(3/2); u=v_pec/sqrt(a)",
            "momentum_transform_after_Gadget_IC_init": "p_Gadget_internal=a^(3/2)*u_Gadget=100*p_CSV",
            "roundtrip_position_rms_force_mesh_cells": position_rms_cells,
            "roundtrip_momentum_relative_rms": momentum_rms_relative,
            "position_storage_precision": "GADGET-2 format-1 POS and VEL records are float32",
            "roundtrip_limit_position_force_mesh_cells": position_limit,
            "roundtrip_limit_momentum_relative_rms": 2e-6},
    }
    if position_rms_cells > position_limit or momentum_rms_relative > 2e-6:
        output.unlink(missing_ok=True)
        raise ValueError(f"GADGET float32 round trip exceeded limit: {position_rms_cells=}, {momentum_rms_relative=}")
    sidecar.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


## @brief Parse the documented command-line interface and execute its selected workflow, retaining nonzero failures.
# @see cosmology_tools
#
# @param argv Command-line argument vector; program-specific parsing is documented by main/usage.
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--redshift", type=float, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--box-mpc-h", type=float, default=168.75)
    parser.add_argument("--omega-m", type=float, default=0.31)
    parser.add_argument("--hubble", type=float, default=0.675)
    args = parser.parse_args(argv)
    result = convert(args.input, args.output, redshift=args.redshift,
                     expected_sha256=args.expected_sha256,
                     box_mpc_h=args.box_mpc_h, omega_m=args.omega_m,
                     hubble=args.hubble)
    print(json.dumps(result, indent=2, allow_nan=False))


## @cond CLI_DISPATCH
if __name__ == "__main__":
    main()
## @endcond
