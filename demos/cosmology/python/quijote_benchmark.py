#!/usr/bin/env python3
## @file quijote_benchmark.py
# @brief Prepare auditable Quijote fiducial runs and compare canonical snapshots.
# @ingroup cosmology_python
# @details External initial states retain canonical momentum p=a*vpec/100,
# with x in Mpc/h and vpec in km/s. No IC generation or rescaling is performed.
# Requested redshifts are converted to synchronized scale-factor endpoints by
# the production application. Numerical convergence requires separate mesh,
# time-step and analysis refinements; one run never certifies a tolerance.
"""Dry-run preparation by default; simulation execution requires explicit --run.

Examples:
  quijote_benchmark.py prepare --ic fiducial.bin --exe /path/Cosmology \
      --output-dir pilot --mesh 512 --steps 512 --ranks 8
  quijote_benchmark.py compare --left pilot/run/snapshots.csv --left-epoch final \
      --right reference-z0.bin --grid 512 --k-max .3 --dk .01 --output compare.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from quijote_analysis import (ParticleSource, analysis_memory_bytes, compare,
                              resolve_sources, sha256)


## @var Fiducial
# @brief Catalogue fiducial values checked against the actual canonical header.
Fiducial = {"total_count": 512**3, "a": 1 / 128, "box_mpc_h": 1000.,
            "omega_m": .3175, "omega_lambda": .6825, "hubble": .6711}
## @var CatalogParameters
# @brief Published catalogue-only parameters absent from Gadget headers; explicitly declared, not verified.
CatalogParameters = {"Omega_bar": .049, "Sigma_8": .834, "n_s": .9624}


## @brief Require imported count/cosmology/epoch; small synthetic fixtures opt out.
# @param header Validated canonical header with counts, scale factor and cosmology.
# @param allow_nonfiducial Explicit opt-out from exact fiducial 512 cubed header requirements.
# @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
def validate_fiducial(header, allow_nonfiducial=False):
    """Require imported count/cosmology/epoch; small synthetic fixtures opt out.

    Gadget headers do not specify baryon fraction, sigma8, n_s or LPT order.
    Those catalog values are recorded as declarations, never header-verified.
    """
    if not (header["flags"] & 1) or header["file_count"] != header["total_count"]:
        raise ValueError("Production IC must be complete, sorted canonical input")
    mismatches = []
    for key, value in Fiducial.items():
        agrees = header[key] == value if key == "total_count" else math.isclose(header[key], value, rel_tol=1e-10, abs_tol=0)
        if not agrees:
            mismatches.append(key)
    if mismatches and not allow_nonfiducial:
        raise ValueError("Not the requested Quijote fiducial 512^3 IC: " + ", ".join(mismatches))
    return mismatches


## @brief Conservative aggregate estimate; rank imbalance/device workspace may add.
# @param header Validated canonical header with counts, scale factor and cosmology.
# @param mesh Even force-mesh side used for aggregate memory estimation.
# @param output_count Number of particle snapshot epochs including the initial state.
# @param ranks Positive MPI process count used in the execution and storage budget.
# @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
def simulation_budget(header, mesh, output_count, ranks):
    """Conservative aggregate estimate; rank imbalance/device workspace may add.

    Uses 64 bytes per force cell plus 112 per particle, a 50% allowance for
    runtime/workspace and 128 MiB per rank. This is not a hardware-fit guarantee.
    """
    if mesh < 4 or mesh % 2 or output_count < 1 or ranks < 1:
        raise ValueError("Invalid budget geometry")
    count = int(header["total_count"])
    base = 64 * mesh**3 + 112 * count
    return {"aggregate_memory_bytes": int(1.5 * base + ranks * 128 * 1024**2),
            "mean_memory_per_rank_bytes": int(1.5 * base / ranks + 128 * 1024**2),
            "snapshot_bytes": output_count * (56 * count + ranks * 128),
            "model": "1.5*(64*mesh^3+112*N)+128MiB/rank; excludes filesystem cache",
            "limitation": "estimate only; rank imbalance, FFT workspace and device memory require measured pilot"}


## @brief Record commit and dirty state without modifying the checkout.
# @param directory Source checkout inspected with read-only Git commands.
# @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
def revision_record(directory):
    """Record commit and dirty state without modifying the checkout."""
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=directory,
                                  capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=directory,
                                capture_output=True, text=True, check=True).stdout
        return {"revision": revision, "dirty": bool(status), "status": status.splitlines()}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


## @brief Evaluate  write json.
# @param path Input or output filesystem path; the calling contract determines freshness and format.
# @param value JSON-serializable provenance record; nonfinite numbers are rejected.
# @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
def _write_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


## @brief Require the declared epochs and complete binary shard sets after exit 0.
# @param run_dir Simulation output directory containing snapshots.csv and binary rank shards.
# @param redshifts Distinct requested output redshifts in descending order.
# @param header Validated canonical header with counts, scale factor and cosmology.
# @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
def completed_output_record(run_dir, redshifts, header):
    """Require the declared epochs and complete binary shard sets after exit 0.

    This is an output-contract check, not an accuracy/convergence test. Each
    epoch is checked for complete IDs and finite payload, with bounded chunks.
    Counts and imported particle mass remain exact. Background and epoch
    metadata allow relative 1e-10 roundoff; this is not a numerical accuracy budget.
    """
    import csv
    path = run_dir / "snapshots.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = [header["a"], *[1 / (1 + z) for z in redshifts]]
    if len(rows) != len(expected) or len({row["name"] for row in rows}) != len(rows):
        raise ValueError("Completed run does not contain each requested epoch exactly once")
    records = []
    for row, a in zip(rows, expected):
        if not math.isclose(float(row["a"]), a, rel_tol=1e-10, abs_tol=0):
            raise ValueError("Completed run snapshot epochs disagree with the requested schedule")
        paths, declared_a = resolve_sources([str(path)], row["name"])
        source = ParticleSource(paths)
        if not math.isclose(source.header["a"], declared_a, rel_tol=1e-10, abs_tol=0):
            raise ValueError("Completed output epoch disagrees with its snapshot manifest")
        for key in ("total_count", "particle_mass_msun_h"):
            if source.header[key] != header[key]:
                raise ValueError(f"Completed output changed {key}")
        for key in ("box_mpc_h", "omega_m", "omega_lambda", "hubble"):
            if not math.isclose(source.header[key], header[key], rel_tol=1e-10, abs_tol=0):
                raise ValueError(f"Completed output changed {key}")
        records.append({"name": row["name"], "a": a,
                        "shards": source.validate()})
    return {"manifest": str(path), "sha256": sha256(path), "epochs": records}


## @brief Execute only a frozen prepared plan after verifying input/executable hashes.
# @param manifest_path Frozen benchmark.json path created by the prepare command.
# @param run False performs verification only; true explicitly authorizes execution.
# @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
def execute_prepared(manifest_path, run=False):
    """Execute only a frozen prepared plan after verifying input/executable hashes.

    A prepared dry run can be inspected and then executed without creating a
    second directory. Started/failed/completed plans cannot be silently rerun.
    """
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "ippl-quijote-benchmark-v1" or manifest.get("state") != "prepared":
        raise ValueError("Only a prepared, not previously started benchmark can be executed")
    checks = [(manifest["config_path"], manifest["config_sha256"]),
              (manifest["executable"]["path"], manifest["executable"]["sha256"]),
              (manifest["ic"]["path"], manifest["ic"]["sha256"])]
    for path, digest in checks:
        if sha256(path) != digest:
            raise ValueError(f"Prepared input or executable changed: {path}")
    command = manifest["command"]
    expected_tail = [manifest["executable"]["path"], manifest["config_path"]]
    if (not isinstance(command, list) or command[-2:] != expected_tail
            or len(command) not in (2, 5)
            or (len(command) == 5 and (command[1] != "-n" or not command[2].isdigit() or int(command[2]) < 1))):
        raise ValueError("Prepared command does not match the recorded executable/config")
    output = manifest_path.parent
    run_dir = Path(manifest["config"]["output"])
    if (run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir()))) or (output / "run.log").exists():
        raise FileExistsError("Prepared simulation output already exists")
    free = shutil.disk_usage(output).free
    required = manifest["budget"]["snapshot_bytes"] + manifest["budget"]["disk_reserve_gib"] * 1024**3
    if free < required:
        raise OSError("Output space no longer satisfies the prepared disk budget")
    if not run:
        return manifest
    manifest["state"] = "running"
    manifest["executed"] = True
    manifest["started_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["execution_environment"] = {key: os.environ.get(key) for key in
                                         ("OMP_NUM_THREADS", "OMP_PROC_BIND", "CUDA_VISIBLE_DEVICES")}
    _write_json(manifest_path, manifest)
    try:
        with (output / "run.log").open("x") as log:
            process = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=False)
        manifest["returncode"] = process.returncode
        if process.returncode == 0:
            redshifts = [float(value) for value in manifest["config"]["output_redshifts"].split(",")]
            manifest["outputs"] = completed_output_record(run_dir, redshifts, manifest["ic"]["header"])
        manifest["state"] = "completed" if process.returncode == 0 else "failed"
    except BaseException as error:
        manifest["state"] = "interrupted_or_failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        _write_json(manifest_path, manifest)
        raise
    manifest["log"] = {"path": str(output / "run.log"), "sha256": sha256(output / "run.log")}
    _write_json(manifest_path, manifest)
    if process.returncode:
        raise RuntimeError(f"Simulation exited {process.returncode}; see {output / 'run.log'}")
    return manifest


## @brief Write a fresh config/manifest; run only when args.run is explicitly true.
# @param args Validated CLI namespace for the documented prepare or comparison workflow.
# @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
def prepare(args):
    """Write a fresh config/manifest; run only when args.run is explicitly true."""
    import quijote_io
    ic = Path(args.ic).resolve(strict=True)
    header = quijote_io.read_header(ic)
    mismatches = validate_fiducial(header, args.allow_nonfiducial)
    if args.mesh < 4 or args.mesh % 2 or args.steps < 1 or args.ranks < 1 or args.diagnostics_every < 1:
        raise ValueError("Require even mesh >=4 and positive steps/ranks/diagnostic interval")
    redshifts = list(args.redshifts)
    if (not redshifts or not all(math.isfinite(z) and z >= 0 for z in redshifts)
            or len(set(redshifts)) != len(redshifts)
            or any(1 / (1 + z) <= header["a"] for z in redshifts)):
        raise ValueError("Output redshifts must be distinct and later than the initial state")
    redshifts.sort(reverse=True)
    executable = Path(args.exe).resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("Simulation executable is not an executable file")
    output = Path(args.output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Benchmark directory must be fresh or empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    budget = simulation_budget(header, args.mesh, 1 + len(redshifts), args.ranks)
    if args.memory_limit_gib is not None:
        if args.memory_limit_gib <= 0 or not math.isfinite(args.memory_limit_gib):
            raise ValueError("Aggregate memory budget must be finite and positive")
        if budget["aggregate_memory_bytes"] > args.memory_limit_gib * 1024**3:
            raise MemoryError("Estimated aggregate simulation memory exceeds the declared budget")
    free = shutil.disk_usage(output.parent).free
    if args.disk_reserve_gib < 0 or not math.isfinite(args.disk_reserve_gib):
        raise ValueError("Disk reserve must be finite and nonnegative")
    required = budget["snapshot_bytes"] + args.disk_reserve_gib * 1024**3
    if free < required:
        raise OSError(f"Insufficient output space: require {required / 1024**3:.2f} GiB including reserve")
    # Validate actual particles as well as the header before advertising ready ICs.
    source = ParticleSource([ic], chunk_size=args.chunk_size)
    source.validate()
    converter = ic.with_name(ic.name + ".json")
    converter_record = None
    if converter.exists():
        data = json.loads(converter.read_text())
        if data["output"]["sha256"] != source.provenance[0]["sha256"]:
            raise ValueError("IC no longer matches converter provenance")
        converter_record = {"path": str(converter), "sha256": sha256(converter), "manifest": data}
    run_dir = output / "run"
    parameter_path = output / "input.par"
    parameters = {"ic_mode": "external", "ic_file": str(ic), "particle_count": header["total_count"],
                  "np": args.mesh, "nt": args.steps, "box_size": header["box_mpc_h"],
                  "z_in": 1 / header["a"] - 1, "z_fi": min(redshifts),
                  "hubble": header["hubble"], "Omega_m": header["omega_m"], **CatalogParameters,
                  "output_redshifts": ",".join(format(z, ".17g") for z in redshifts),
                  "snapshot_format": "binary", "output": str(run_dir),
                  "diagnostics_every": args.diagnostics_every, "write_particles": "true"}
    # The production parser strips # and //; reject unrepresentable filesystem names.
    if any("\n" in str(value) or "#" in str(value) or "//" in str(value) or '"' in str(value)
           for value in parameters.values()):
        raise ValueError("Parameter values contain characters unsupported by the config parser")
    lines = [f"{name} = {value}" for name, value in parameters.items()]
    command = [str(executable), str(parameter_path)]
    if args.ranks > 1:
        launcher = shutil.which(args.mpiexec)
        if not launcher:
            raise FileNotFoundError(f"MPI launcher not found: {args.mpiexec}")
        command = [launcher, "-n", str(args.ranks), *command]
    output.mkdir(exist_ok=True)
    parameter_path.write_text("\n".join(lines) + "\n")
    manifest = {"schema": "ippl-quijote-benchmark-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "state": "prepared", "executed": False, "header_matches_fiducial": not mismatches,
                "nonfiducial_fields": mismatches, "ic": source.provenance[0],
                "converter": converter_record, "config": parameters,
                "config_path": str(parameter_path), "config_sha256": sha256(parameter_path),
                "executable": {"path": str(executable), "sha256": sha256(executable)},
                "code": revision_record(args.source_dir), "command": command,
                "environment": {key: os.environ.get(key) for key in
                                ("OMP_NUM_THREADS", "OMP_PROC_BIND", "CUDA_VISIBLE_DEVICES")},
                "budget": {**budget, "free_disk_bytes": free, "disk_reserve_gib": args.disk_reserve_gib,
                           "declared_aggregate_memory_limit_gib": args.memory_limit_gib},
                "header_verified": list(Fiducial),
                "catalog_only_parameters": CatalogParameters,
                "catalogue_identity": "unverified: matching headers do not distinguish fiducial from fiducial_ZA",
                "qualification": "imported-state comparison; no force/time convergence or performance claim",
                "notes": ["Header does not establish LPT order or catalog-only power-spectrum parameters.",
                          "No parallel HDF5; canonical binary chunks are used."]}
    manifest_path = output / "benchmark.json"
    _write_json(manifest_path, manifest)
    if args.run:
        return execute_prepared(manifest_path, run=True)
    return manifest


## @brief Physical bins with a possibly shorter final bin; upper bounds excluded.
# @param k_min Inclusive lower shell edge in h/Mpc, nonnegative.
# @param k_max Exclusive upper shell edge in h/Mpc, greater than k_min.
# @param dk Positive shell width in h/Mpc; final shell may be shorter.
# @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
def physical_edges(k_min, k_max, dk):
    """Physical bins with a possibly shorter final bin; upper bounds excluded."""
    if not all(math.isfinite(value) for value in (k_min, k_max, dk)) or not (0 <= k_min < k_max and dk > 0):
        raise ValueError("Require 0 <= k_min < k_max and dk > 0")
    count = int(math.ceil((k_max - k_min) / dk))
    if count > 1_000_000:
        raise ValueError("Too many spectral bins")
    return np.concatenate((k_min + np.arange(count) * dk, [k_max]))


## @brief Compare canonical reference and simulation files with identical estimators.
# @param args Validated CLI namespace for the documented prepare or comparison workflow.
# @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
def compare_command(args):
    """Compare canonical reference and simulation files with identical estimators."""
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    lp, la = resolve_sources(args.left, args.left_epoch)
    rp, ra = resolve_sources(args.right, args.right_epoch)
    left = ParticleSource(lp, args.chunk_size, la)
    right = ParticleSource(rp, args.chunk_size, ra)
    if args.memory_limit_gib <= 0 or not math.isfinite(args.memory_limit_gib):
        raise ValueError("Analysis memory budget must be finite and positive")
    report = compare(left, right, grid=args.grid, edges=physical_edges(args.k_min, args.k_max, args.dk),
                     workers=args.workers, memory_limit_bytes=args.memory_limit_gib * 1024**3,
                     shot_noise=args.shot_noise,
                     line_of_sight=None if args.line_of_sight is None else "xyz".index(args.line_of_sight))
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, report)
    return report


## @brief Build the prepare, frozen execute and compare command-line interfaces.
# @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
def parser():
    root = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = root.add_subparsers(dest="command_name", required=True)
    prepare_parser = commands.add_parser("prepare", help="Write config/provenance; dry run unless --run")
    prepare_parser.add_argument("--ic", required=True)
    prepare_parser.add_argument("--exe", required=True)
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--mesh", type=int, required=True, help="Independent force-grid side")
    prepare_parser.add_argument("--steps", type=int, required=True, help="Requested base time intervals; not accuracy-qualified")
    prepare_parser.add_argument("--ranks", type=int, default=1)
    prepare_parser.add_argument("--mpiexec", default="mpiexec")
    prepare_parser.add_argument("--redshifts", type=float, nargs="+", default=[1., .5, 0.])
    prepare_parser.add_argument("--diagnostics-every", type=int, default=10)
    prepare_parser.add_argument("--chunk-size", type=int, default=262144)
    prepare_parser.add_argument("--memory-limit-gib", type=float, help="Aggregate simulation memory cap")
    prepare_parser.add_argument("--disk-reserve-gib", type=float, default=1.)
    prepare_parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[3])
    prepare_parser.add_argument("--allow-nonfiducial", action="store_true", help="Explicitly permit synthetic/small or other imported cases")
    prepare_parser.add_argument("--run", action="store_true", help="Actually launch this prepared simulation")
    prepare_parser.set_defaults(function=prepare)
    execution = commands.add_parser("execute", help="Verify/execute an existing frozen prepared benchmark")
    execution.add_argument("--manifest", required=True)
    execution.add_argument("--run", action="store_true", help="Actually launch; otherwise verify only")
    execution.set_defaults(function=lambda args: execute_prepared(args.manifest, args.run))
    comparison = commands.add_parser("compare", help="Bounded-chunk real-space power and cross-correlation")
    comparison.add_argument("--left", nargs="+", required=True, help="Canonical files/globs or snapshots.csv/converter JSON")
    comparison.add_argument("--right", nargs="+", required=True)
    comparison.add_argument("--left-epoch")
    comparison.add_argument("--right-epoch")
    comparison.add_argument("--grid", type=int, required=True, help="Independent analysis-grid side")
    comparison.add_argument("--k-min", type=float, default=0.)
    comparison.add_argument("--k-max", type=float, required=True)
    comparison.add_argument("--dk", type=float, required=True)
    comparison.add_argument("--workers", type=int, default=1)
    comparison.add_argument("--chunk-size", type=int, default=262144)
    comparison.add_argument("--memory-limit-gib", type=float, default=8.)
    comparison.add_argument("--shot-noise", choices=("raw", "poisson"), default="raw")
    comparison.add_argument("--line-of-sight", choices=("x", "y", "z"), help="Optional RSD shift and raw P2/P4 multipoles")
    comparison.add_argument("--output", required=True)
    comparison.set_defaults(function=compare_command)
    return root


## @brief Dispatch the explicitly selected CLI action and print its output record.
# @param argv Optional explicit CLI argument list; None uses process arguments.
# @return The result described by the function contract; errors reject unsupported or inconsistent inputs.
def main(argv=None):
    args = parser().parse_args(argv)
    result = args.function(args)
    if args.command_name in ("prepare", "execute"):
        print(json.dumps({"state": result["state"], "executed": result["executed"],
                          "command": result["command"], "budget": result["budget"]}, indent=2))
    else:
        print(json.dumps({"output": args.output, "shells": len(result["rows"]),
                          "estimated_memory_bytes": result["estimated_memory_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
