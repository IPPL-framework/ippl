#!/usr/bin/env python3
"""Verify a completed matched Gadget-2 256^3 run and measure its z=0 spectrum."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np


SharedRoot = Path("/data/user/adelmann/cosmology-zeldovich-np256-20261006")
GadgetRoot = Path("/data/user/adelmann/gadget2-zeldovich-np256-20261006")
InputCsvSha256 = "3b4a3e1864ad535369444b98779c117750e1e980cbe7d01afedc20f8329cda11"
ParticleGrid = 256
MeshGrid = 256
Cutoff = 96
BoxMpcH = 168.75
OmegaMatter = .31
OmegaLambda = .69
HubbleParam = .675
SofteningKpcH = BoxMpcH * 1000.0 / ParticleGrid * .02


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_record(stream, expected_bytes: int | None = None) -> bytes:
    marker = stream.read(4)
    if len(marker) != 4:
        raise ValueError("truncated GADGET Fortran record marker")
    size = struct.unpack("<i", marker)[0]
    if size < 0 or (expected_bytes is not None and size != expected_bytes):
        raise ValueError(f"unexpected GADGET record length {size}; expected {expected_bytes}")
    data = stream.read(size)
    trailer = stream.read(4)
    if len(data) != size or len(trailer) != 4 or struct.unpack("<i", trailer)[0] != size:
        raise ValueError("truncated GADGET record or mismatching trailing marker")
    return data


def parse_parameters(path: Path) -> dict[str, str]:
    result = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split(maxsplit=1)
        if len(fields) != 2 or fields[0] in result:
            raise ValueError(f"malformed/duplicate GADGET parameter at line {line_number}")
        result[fields[0]] = fields[1]
    return result


def read_snapshot(path: Path):
    count = ParticleGrid**3
    with path.open("rb") as stream:
        header = read_record(stream, 256)
        positions_raw = read_record(stream, count * 3 * 4)
        velocities_raw = read_record(stream, count * 3 * 4)
        ids_raw = read_record(stream, count * 4)
        if stream.read(1):
            raise ValueError("unexpected trailing bytes after GADGET format-1 records")
    counts = struct.unpack_from("<6i", header, 0)
    masses = struct.unpack_from("<6d", header, 24)
    a, redshift = struct.unpack_from("<2d", header, 72)
    totals = struct.unpack_from("<6I", header, 96)
    box, omega_m, omega_lambda, hubble = struct.unpack_from("<4d", header, 128)
    expected_counts = (0, count, 0, 0, 0, 0)
    if counts != expected_counts or totals != expected_counts or masses[1] <= 0:
        raise ValueError("snapshot is not exactly one 256^3 collisionless dark-matter species")
    if (not math.isclose(a, 1.0, rel_tol=0, abs_tol=1e-12)
            or not math.isclose(redshift, 0, rel_tol=0, abs_tol=1e-12)
            or not math.isclose(box, BoxMpcH * 1000, rel_tol=0, abs_tol=1e-8)
            or not math.isclose(omega_m, OmegaMatter, rel_tol=0, abs_tol=1e-12)
            or not math.isclose(omega_lambda, OmegaLambda, rel_tol=0, abs_tol=1e-12)
            or not math.isclose(hubble, HubbleParam, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("snapshot header does not match the requested z=0 cosmology")
    positions = np.frombuffer(positions_raw, dtype="<f4").reshape(count, 3).astype(np.float64)
    positions *= .001  # snapshot POS block: kpc/h -> Mpc/h
    velocities = np.frombuffer(velocities_raw, dtype="<f4")
    ids = np.frombuffer(ids_raw, dtype="<u4")
    if (not np.isfinite(positions).all() or not np.isfinite(velocities).all()
            or np.any((positions < 0) | (positions >= BoxMpcH))
            or not np.array_equal(np.sort(ids), np.arange(count, dtype=np.uint32))):
        raise ValueError("snapshot contains invalid positions, velocities, or particle IDs")
    metadata = {"counts_by_type": counts, "total_counts_by_type": totals,
        "mass_table": masses, "scale_factor": a, "redshift": redshift,
        "box_kpc_over_h": box, "omega_m": omega_m,
        "omega_lambda": omega_lambda, "hubble_param": hubble}
    return metadata, positions


def validate_parameters(run: dict) -> dict:
    path = Path(run["parameter_file"])
    if sha256(path) != run["parameter_sha256"]:
        raise ValueError("GADGET parameter-file checksum mismatch")
    parameters = parse_parameters(path)
    exact_strings = {"InitCondFile": run["converted_ic"],
        "OutputDir": run["output_directory"] + "/",
        "OutputListFilename": run["output_list"], "ResubmitCommand": "none"}
    for name, expected in exact_strings.items():
        if parameters.get(name) != expected:
            raise ValueError(f"GADGET parameter {name} mismatch")
    expected_values = {"TimeBegin": .01, "TimeMax": 1., "Omega0": .31,
        "OmegaLambda": .69, "HubbleParam": .675, "BoxSize": 168750.,
        "ErrTolIntAccuracy": .025, "MaxRMSDisplacementFac": .2,
        "MaxSizeTimestep": .025, "ErrTolTheta": .5, "ErrTolForceAcc": .005,
        "SofteningHalo": SofteningKpcH, "SofteningHaloMaxPhys": SofteningKpcH}
    for name, expected in expected_values.items():
        if not math.isclose(float(parameters[name]), expected, rel_tol=0, abs_tol=1e-14):
            raise ValueError(f"GADGET parameter {name} differs from the recorded protocol")
    expected_integer = {"ComovingIntegrationOn": 1, "PeriodicBoundariesOn": 1,
        "OutputListOn": 0, "ICFormat": 1, "SnapFormat": 1,
        "TypeOfTimestepCriterion": 0, "NumFilesPerSnapshot": 1,
        "NumFilesWrittenInParallel": 1, "ResubmitOn": 0}
    for name, expected in expected_integer.items():
        if int(parameters[name]) != expected:
            raise ValueError(f"GADGET parameter {name} differs from the protocol")
    if int(parameters.get("TimeLimitCPU", "0")) < 86400:
        raise ValueError("GADGET internal watchdog is shorter than one day")
    if sha256(Path(run["output_list"])) != run["output_list_sha256"]:
        raise ValueError("GADGET output-list checksum mismatch")
    return {"values": expected_values | expected_integer,
            "time_limit_cpu_seconds": int(parameters["TimeLimitCPU"])}


def analyze(campaign_path: Path, output_path: Path) -> dict:
    campaign_path = campaign_path.resolve(strict=True)
    output_path = output_path.resolve()
    if not campaign_path.is_relative_to(GadgetRoot) or not output_path.is_relative_to(GadgetRoot):
        raise ValueError("GADGET analysis inputs and outputs must remain under the isolated campaign root")
    campaign_bytes = campaign_path.read_bytes()
    campaign = json.loads(campaign_bytes)
    run = campaign.get("run", {})
    if (campaign.get("status") != "complete" or campaign.get("complete") is not True
            or run.get("state") != "complete" or run.get("return_code") != 0
            or run.get("timed_out") is not False):
        raise ValueError("GADGET-2 run is not complete and successful")
    if (run.get("particle_grid") != ParticleGrid or run.get("pm_grid") != MeshGrid
            or run.get("particle_count") != ParticleGrid**3
            or run.get("input_csv_sha256") != InputCsvSha256
            or run.get("initial_redshift") != 99 or run.get("final_redshift") != 0
            or run.get("ranks") != 8 or run.get("threads_per_rank") != 1):
        raise ValueError("GADGET-2 run identity differs from the matched 256^3 protocol")
    if (run.get("execution_mode") != "authorized_login"
            or not str(run.get("execution_host", "")).startswith("merlin-l-")
            or run.get("job_id") is not None):
        raise ValueError("GADGET-2 run was not executed on the authorized Merlin login host")
    validate_parameters(run)
    for key in ("executable", "build_makefile", "build_log", "controller_source",
                "converter_source", "batch_source", "login_source", "pmgrid_patch", "converter_sidecar"):
        path = Path(run[key])
        if sha256(path) != run[key + "_sha256"]:
            raise ValueError(f"GADGET run provenance changed: {key}")
    for key in ("gadget2_source_archive", "fftw2_source_archive"):
        if sha256(Path(run[key])) != run[key + "_sha256"]:
            raise ValueError(f"GADGET source archive changed: {key}")
    if "-DPMGRID=256" not in Path(run["build_makefile"]).read_text():
        raise ValueError("recorded GADGET build does not use PMGRID=256")

    sidecar_path = Path(run["converted_ic"] + ".json")
    sidecar = json.loads(sidecar_path.read_text())
    if (sha256(sidecar_path) != run["converter_sidecar_sha256"]
            or sidecar["input"]["sha256"] != InputCsvSha256
            or sha256(Path(sidecar["input"]["path"])) != InputCsvSha256
            or sidecar["output"]["sha256"] != run["converted_ic_sha256"]
            or sha256(Path(run["converted_ic"])) != run["converted_ic_sha256"]
            or sidecar["initial_conditions"]["particle_count"] != ParticleGrid**3):
        raise ValueError("GADGET format-1 input did not preserve the shared IC")

    shared_manifest = json.loads((SharedRoot / "ics/ic-manifest.json").read_text())
    if sha256(SharedRoot / "ics/ic-manifest.json") != run["input_manifest_sha256"]:
        raise ValueError("shared IC manifest changed after the run")
    plot_manifest_path = SharedRoot / "plots/z99-np256-ippl-fastpm-v1/manifest.json"
    plot_values_path = SharedRoot / "plots/z99-np256-ippl-fastpm-v1/plotted_values.json"
    if (sha256(plot_manifest_path) != run["reference_plot_manifest_sha256"]
            or sha256(plot_values_path) != run["reference_plot_values_sha256"]):
        raise ValueError("IPPL/FastPM reference plot inputs changed after Gadget run")
    plot_manifest = json.loads(plot_manifest_path.read_text())
    for name, expected in plot_manifest["input_hashes"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"IPPL/FastPM provenance check failed: {name}")
    fastpm_shards = campaign["reference_audit"].get("fastpm_final_shards", [])
    if len(fastpm_shards) != 8:
        raise ValueError("campaign lacks hashes for the eight FastPM final rank shards")
    for item in fastpm_shards:
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"FastPM final snapshot changed since the matched comparison: {item['path']}")
    plot_values = json.loads(plot_values_path.read_text())
    if (plot_values["shared_input_sha256"] != InputCsvSha256
            or plot_values["particle_grid"] != ParticleGrid
            or plot_values["force_mesh_grid"] != MeshGrid
            or plot_values["ic_cutoff_fundamental"] != Cutoff):
        raise ValueError("IPPL/FastPM reference spectrum uses a different IC or resolution")
    ippl_run_path = SharedRoot / "runs/z99-ippl-a100-p256-m256/run.json"
    fastpm_run_path = SharedRoot / "runs/z99-fastpm-cpu-p256-m256/run.json"
    if sha256(ippl_run_path) != run["ippl_run_json_sha256"] or sha256(fastpm_run_path) != run["fastpm_run_json_sha256"]:
        raise ValueError("IPPL/FastPM run metadata changed after the Gadget run")
    for path, status_field in ((ippl_run_path, "solver_exit_status"),
                               (fastpm_run_path, "slurm_exit_status")):
        item = json.loads(path.read_text())
        if (item.get("state") != "complete" or item.get(status_field) != 0
                or item.get("input_csv_sha256") != InputCsvSha256):
            raise ValueError(f"reference run is incomplete or does not use the shared IC: {path}")

    snapshots = run.get("snapshots", [])
    if len(snapshots) != 1:
        raise ValueError("expected exactly one GADGET final snapshot")
    snapshot = Path(snapshots[0]["path"])
    if snapshot.stat().st_size != snapshots[0]["bytes"] or sha256(snapshot) != snapshots[0]["sha256"]:
        raise ValueError("GADGET snapshot checksum or size mismatch")
    header, positions = read_snapshot(snapshot)

    spectrum_path = SharedRoot / "source/demos/cosmology/analyze_zeldovich_benchmark.py"
    if sha256(spectrum_path) != run["spectrum_source_sha256"]:
        raise ValueError("shared CIC estimator source changed after Gadget run")
    sys.path.insert(0, str(spectrum_path.parent))
    shared = importlib.import_module("analyze_zeldovich_benchmark")
    if Path(shared.__file__).resolve() != spectrum_path.resolve():
        raise ValueError("unexpected CIC spectrum implementation imported")
    spectrum = shared.cic_power(positions, particle_grid=ParticleGrid,
        mesh_grid=MeshGrid, box_size=BoxMpcH, cutoff=Cutoff)
    if len(spectrum) != Cutoff or [row["shell"] for row in spectrum] != list(range(1, Cutoff + 1)):
        raise ValueError("CIC estimator did not return all 96 shells")

    shell_values = plot_values["shells"]
    k_values = plot_values["k_h_per_mpc"]
    if shell_values != list(range(1, Cutoff + 1)) or len(k_values) != Cutoff:
        raise ValueError("reference plot does not contain the expected shell centers")
    comparisons = []
    for label in ("IPPL · A100, one rank · z=0", "FastPM · CPU, 8 ranks · z=0"):
        reference = plot_values["powers_shot_subtracted_mpc_over_h_cubed"][label]
        offsets = []
        for row, k, power in zip(spectrum, k_values, reference):
            if not math.isclose(row["k_h_per_mpc"], k, rel_tol=0, abs_tol=1e-14):
                raise ValueError("GADGET and PM spectra have different Fourier shell centers")
            current = row["P_shot_subtracted"]
            if current > 0 and power > 0:
                offsets.append({"shell": row["shell"], "k_h_per_mpc": k,
                    "gadget_power": current, "reference_power": power,
                    "gadget_over_reference": current / power,
                    "relative_difference": current / power - 1.0})
        diffs = [item["relative_difference"] for item in offsets]
        comparisons.append({"reference": label, "positive_shell_count": len(offsets),
            "shells": offsets,
            "max_abs_relative_power_difference": max(abs(x) for x in diffs),
            "rms_relative_power_difference": math.sqrt(sum(x*x for x in diffs) / len(diffs)),
            "first_shell_relative_power_difference": diffs[0]})

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite analysis output {output_path}")
    report = {"schema": "ippl-gadget2-np256-z99-z0-spectrum-v1",
        "campaign": str(campaign_path), "campaign_sha256": hashlib.sha256(campaign_bytes).hexdigest(),
        "campaign_complete": True, "solver_return_code": run["return_code"],
        "solver_wall_seconds": run["solver_wall_seconds"], "ranks": run["ranks"],
        "threads_per_rank": run["threads_per_rank"], "initial_redshift": 99,
        "final_redshift": 0, "particle_grid": ParticleGrid, "mesh_grid": MeshGrid,
        "particle_count": ParticleGrid**3, "cutoff": Cutoff,
        "shared_input_csv_sha256": InputCsvSha256,
        "fastpm_final_shards": fastpm_shards,
        "ic_seed": shared_manifest["seed"], "ic_manifest_sha256": sha256(SharedRoot / "ics/ic-manifest.json"),
        "converted_input_sha256": run["converted_ic_sha256"],
        "input_roundtrip": sidecar["initial_conditions"],
        "snapshot": {"path": str(snapshot), "sha256": sha256(snapshot),
            "bytes": snapshot.stat().st_size, "header": header,
            "particle_count": ParticleGrid**3},
        "force_model": "GADGET-2.0.7 TreePM; PMGRID=256; short-range tree enabled",
        "spectrum_definition": "periodic 256^3 CIC density mesh, CIC assignment-window power deconvolution, Poisson shot-noise subtraction; same estimator for all codes",
        "spectrum_estimator_source": str(spectrum_path),
        "spectrum_estimator_source_sha256": sha256(spectrum_path),
        "final_spectrum": spectrum, "comparisons": comparisons,
        "limitations": ["single shared Gaussian realization; no ensemble uncertainty estimate",
            "GADGET-2 TreePM includes short-range tree forces while IPPL/FastPM are plain PM",
            "initial 1LPT spectrum is truncated at |n|<=96; shells are a characterization, not a continuum-accuracy claim",
            "auto-power agreement does not establish particle-by-particle trajectory agreement"]}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=GadgetRoot / "campaign.json")
    parser.add_argument("--output", type=Path,
        default=GadgetRoot / "analysis/gadget2-np256-z99-z0-results.json")
    arguments = parser.parse_args()
    result = analyze(arguments.campaign, arguments.output)
    print(json.dumps({"analysis": str(arguments.output),
        "max_abs_relative_difference_vs_ippl": result["comparisons"][0]["max_abs_relative_power_difference"],
        "max_abs_relative_difference_vs_fastpm": result["comparisons"][1]["max_abs_relative_power_difference"]}, indent=2))
