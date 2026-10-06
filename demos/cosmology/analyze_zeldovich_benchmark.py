#!/usr/bin/env python3
"""Audit saved broadband IPPL/FastPM snapshots and compute CIC power spectra."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess

import numpy as np
import pandas as pd

from gaussian_fixture import bbks_spectrum
from validate_linear import growth_reference


Root = Path("/data/user/adelmann/cosmology-zeldovich-broadband-20261004")
Schema = "ippl-zeldovich-broadband-analysis-v1"
Ranks, ParticleGrid, MeshGrid, Checkpoints = 8, 128, 128, 24
BoxSize, OmegaMatter, Cutoff = 168.75, .31, 48
ExpectedRuns = {"z99_ippl", "z99_fastpm", "z49_ippl", "z49_fastpm"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_hash_uncompressed(path):
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_snapshot(record, checkpoint):
    prefix = f"particles_checkpoint{checkpoint:04d}_rank"
    rows = [item for item in record["snapshots"]
            if Path(item["path"]).name.startswith(prefix)]
    if len(rows) != Ranks or {int(Path(row["path"]).name.split("rank")[-1].split(".")[0])
                              for row in rows} != set(range(Ranks)):
        raise ValueError(f"{record['case']}: incomplete checkpoint {checkpoint} rank shards")
    frames = []
    for item in rows:
        path = Path(item["path"])
        if sha256(path) != item["sha256"] or file_hash_uncompressed(path) != item["csv_sha256"]:
            raise ValueError(f"{record['case']}: corrupt archived particle shard {path}")
        frame = pd.read_csv(path, compression="gzip", dtype={"id": np.uint64},
                            float_precision="round_trip")
        if list(frame.columns) != ["id", "x", "y", "z", "px", "py", "pz"]:
            raise ValueError(f"{record['case']}: unexpected particle snapshot columns")
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True).sort_values("id", kind="stable").reset_index(drop=True)
    ids = frame.id.to_numpy(dtype=np.uint64)
    phase = frame[["x", "y", "z", "px", "py", "pz"]].to_numpy(dtype=np.float64)
    if (len(frame) != ParticleGrid**3 or not np.array_equal(ids, np.arange(ParticleGrid**3))
            or not np.isfinite(phase).all()):
        raise ValueError(f"{record['case']}: incomplete, duplicate or nonfinite phase space")
    if np.any((phase[:, :3] < 0) | (phase[:, :3] >= BoxSize)):
        raise ValueError(f"{record['case']}: particle positions are outside the periodic box")
    return ids, phase[:, :3], phase[:, 3:]


def cic_power(positions, *, particle_grid=ParticleGrid, mesh_grid=MeshGrid,
              box_size=BoxSize, cutoff=Cutoff):
    """Periodic CIC density power, assignment-window corrected, Poisson-subtracted."""
    m, n, box = mesh_grid, particle_grid, box_size
    mean = n**3 / m**3
    counts = np.zeros((m, m, m), dtype=np.float64)
    coordinate = np.remainder(np.asarray(positions, dtype=np.float64), box) * (m / box)
    lower = np.floor(coordinate).astype(np.int32)
    fraction = coordinate - lower
    flat = counts.ravel()
    for dz in (0, 1):
        wz = (1 - fraction[:, 2]) if dz == 0 else fraction[:, 2]
        iz = (lower[:, 2] + dz) % m
        for dy in (0, 1):
            wy = (1 - fraction[:, 1]) if dy == 0 else fraction[:, 1]
            iy = (lower[:, 1] + dy) % m
            for dx in (0, 1):
                wx = (1 - fraction[:, 0]) if dx == 0 else fraction[:, 0]
                ix = (lower[:, 0] + dx) % m
                linear = (iz * m + iy) * m + ix
                weights = wx * wy * wz
                flat += np.bincount(linear, weights=weights, minlength=m**3)
    if not math.isclose(float(counts.sum()), float(n**3), rel_tol=0, abs_tol=1e-7):
        raise ValueError("CIC deposition did not conserve particle count")
    delta = counts / mean - 1.0
    del counts
    coefficients = np.fft.rfftn(delta) / m**3
    del delta
    nx = np.rint(np.fft.fftfreq(m) * m).astype(np.int32)[:, None, None]
    ny = np.rint(np.fft.fftfreq(m) * m).astype(np.int32)[None, :, None]
    nz = np.rint(np.fft.rfftfreq(m) * m).astype(np.int32)[None, None, :]
    radiusSquared = nx.astype(np.int64)**2 + ny.astype(np.int64)**2 + nz.astype(np.int64)**2
    radius = np.sqrt(radiusSquared)
    shell = np.floor(radius + .5).astype(np.int16)
    # CIC amplitude is prod sinc(n_i/NM)^2; power correction is its square.
    windowPower = (np.sinc(nx / m)**4 * np.sinc(ny / m)**4 * np.sinc(nz / m)**4)
    modePower = BoxSize**3 * np.abs(coefficients)**2 / windowPower
    multiplicity = np.full(coefficients.shape, 2, dtype=np.float64)
    multiplicity[..., 0] = 1
    if m % 2 == 0:
        multiplicity[..., -1] = 1
    maxModes = cutoff
    valid = (shell > 0) & (shell <= maxModes)
    bins = shell[valid].astype(np.int32)
    weights = multiplicity[valid]
    weightedCount = np.bincount(bins, weights=weights, minlength=maxModes + 1)
    weightedK = np.bincount(bins, weights=weights * radius[valid], minlength=maxModes + 1)
    raw = np.bincount(bins, weights=weights * modePower[valid], minlength=maxModes + 1)
    shotNoise = box**3 / n**3
    rows = []
    fundamental = 2 * math.pi / box
    for index in range(1, maxModes + 1):
        if weightedCount[index] <= 0:
            continue
        measured = float(raw[index] / weightedCount[index])
        rows.append({"shell": index, "k_h_per_mpc": float(weightedK[index] / weightedCount[index] * fundamental),
                     "P_cic_deconvolved_raw": measured,
                     "shot_noise": shotNoise, "P_shot_subtracted": measured - shotNoise,
                     "full_lattice_mode_count": int(weightedCount[index])})
    return rows


def load_table(record):
    path = Path(record["checkpoints"])
    if sha256(path) != record["checkpoint_table_sha256"]:
        raise ValueError(f"{record['case']}: checkpoint table hash mismatch")
    table = pd.read_csv(path, float_precision="round_trip")
    if (len(table) != Checkpoints + 1 or list(table.checkpoint) != list(range(Checkpoints + 1))
            or not np.isfinite(table.to_numpy(dtype=float)).all()):
        raise ValueError(f"{record['case']}: incomplete checkpoint table")
    return table


def periodic_difference(left, right):
    delta = np.asarray(left) - np.asarray(right)
    return delta - BoxSize * np.rint(delta / BoxSize)


def vector_rms(values):
    return float(np.sqrt(np.mean(np.sum(np.asarray(values)**2, axis=-1))))


def analyze(campaign_path: Path, output_path: Path):
    if socket.gethostname().split(".")[0] != "merlin-l-001":
        raise RuntimeError("Scientific analysis is restricted to the authorized Merlin host")
    campaign_path = campaign_path.resolve(strict=True)
    if not campaign_path.is_relative_to(Root):
        raise ValueError("Campaign input must stay below its authorized Merlin root")
    campaign = json.loads(campaign_path.read_text())
    if campaign.get("state") != "complete" or not campaign.get("complete"):
        raise ValueError("Refusing to analyze a partial or failed campaign")
    if campaign.get("provenance") != campaign.get("provenance_unchanged_after"):
        raise ValueError("Frozen campaign source or executable provenance changed during evolution")
    for key in ("ippl_executable", "fastpm_executable", "fastpm_manifest"):
        item = campaign["provenance"][key]
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen binary/manifest changed: {item['path']}")
    model = campaign["provenance"]["ippl_model_source"]
    if (subprocess.check_output(["git", "-C", model["path"], "rev-parse", "HEAD"], text=True).strip()
            != model["commit"]):
        raise ValueError("Frozen IPPL model source commit changed")
    subprocess.run(["git", "-C", model["path"], "diff", "--exit-code", "HEAD", "--"], check=True)
    modelFiles = {
        "cosmology_simulation_sha256": "CosmologySimulation.h",
        "cosmology_physics_sha256": "CosmologyPhysics.h",
        "cosmology_config_sha256": "CosmologyConfig.h",
        "evolution_adapter_sha256": "tests/CompareCosmologyEvolution.cpp",
    }
    for key, filename in modelFiles.items():
        path = Path(model["path"]) / "demos/cosmology" / filename
        if sha256(path) != model[key]:
            raise ValueError(f"Frozen IPPL model source changed: {path}")
    sourceRoot = Root / "source"
    for relative, expected in campaign["provenance"]["controller_source_hashes"].items():
        if sha256(sourceRoot / relative) != expected:
            raise ValueError(f"Campaign controller/analysis source changed: {relative}")
    runs = {record["case"]: record for record in campaign["runs"]}
    if set(runs) != ExpectedRuns or any(row["state"] != "complete" for row in runs.values()):
        raise ValueError("Expected exactly four complete IPPL/FastPM runs")
    icPath = Root / "ics/ic-manifest.json"
    if sha256(icPath) != campaign["initial_conditions"]["manifest_sha256"]:
        raise ValueError("Initial-condition manifest changed")
    icManifest = json.loads(icPath.read_text())
    if icManifest["coefficient_sha256"] != campaign["initial_conditions"]["realization_sha256"]:
        raise ValueError("Shared Fourier realization hash differs")
    fixtureByZ = {int(item["redshift_initial"]): item for item in icManifest["fixtures"]}
    sourceReportHash = sha256(campaign_path)
    environment = campaign["protocol"]
    allSpectra = {}
    runAudits = []
    for name in sorted(ExpectedRuns):
        record = runs[name]
        z = int(record["redshift_initial"])
        expectedInput = fixtureByZ[z]
        inputPath = Root / "ics" / expectedInput["csv"]
        if (sha256(inputPath) != expectedInput["csv_sha256"]
                or record["input_sha256"] != expectedInput["csv_sha256"]):
            raise ValueError(f"{name}: shared input CSV hash mismatch")
        table = load_table(record)
        outputDir = Path(record["output"])
        metadataPath = outputDir / "metadata.txt"
        if sha256(metadataPath) != record["output_metadata_sha256"]:
            raise ValueError(f"{name}: output metadata hash mismatch")
        expectedA = record["a_initial"] * np.exp(
            np.arange(Checkpoints + 1) * math.log(1.0 / record["a_initial"]) / Checkpoints)
        if float(np.max(np.abs(table.a.to_numpy() / expectedA - 1))) > 3e-13:
            raise ValueError(f"{name}: checkpoint epoch schedule mismatch")
        spectra = []
        for checkpoint in range(Checkpoints + 1):
            _, positions, momenta = read_snapshot(record, checkpoint)
            spectra.append(cic_power(positions))
            if checkpoint == 0:
                # Verify both codes wrote the exact shared particle state at input.
                original = pd.read_csv(inputPath, dtype={"id": np.uint64}, float_precision="round_trip")
                inputPosition = original[["x", "y", "z"]].to_numpy(dtype=np.float64)
                inputMomentum = original[["px", "py", "pz"]].to_numpy(dtype=np.float64)
                positionError = vector_rms(periodic_difference(positions, inputPosition)) / (BoxSize / MeshGrid)
                momentumError = vector_rms(momenta - inputMomentum) / vector_rms(inputMomentum)
                del original, inputPosition, inputMomentum
                if positionError > 1e-9 or momentumError > 1e-6:
                    raise ValueError(f"{name}: imported checkpoint differs from the shared initial state")
                record["initial_state_roundtrip"] = {
                    "position_rms_force_mesh_cells": positionError,
                    "momentum_relative_rms": momentumError}
        allSpectra[name] = spectra
        maximumCheckpointMassError = float(np.max(np.abs(table.mass_error.to_numpy())))
        metadataMassError = (float(record["metadata"]["maximum_mass_error"])
                             if "maximum_mass_error" in record["metadata"] else None)
        runAudits.append({"case": name, "metadata_sha256": record["output_metadata_sha256"],
                          "checkpoint_table_sha256": record["checkpoint_table_sha256"],
                          "snapshot_shards_verified": len(record["snapshots"]),
                          "initial_state_roundtrip": record["initial_state_roundtrip"],
                          "maximum_mass_error_metadata": metadataMassError,
                          "maximum_mass_error_at_checkpoints": maximumCheckpointMassError,
                          "scale_factors": table.a.to_list(),
                          "rank_count": int(record["metadata"]["ranks"]),
                          "threads_per_rank": int(record["metadata"]["threads"])})

    comparisons = []
    for z in (99, 49):
        leftName, rightName = f"z{z}_ippl", f"z{z}_fastpm"
        left = allSpectra[leftName][-1]
        right = allSpectra[rightName][-1]
        byLeft = {row["shell"]: row for row in left}
        byRight = {row["shell"]: row for row in right}
        powerPairs = []
        for shell in sorted(set(byLeft) & set(byRight)):
            a, b = byLeft[shell], byRight[shell]
            if a["P_shot_subtracted"] > 0 and b["P_shot_subtracted"] > 0:
                powerPairs.append({"shell": shell, "k_h_per_mpc": a["k_h_per_mpc"],
                                   "P_ippl": a["P_shot_subtracted"],
                                   "P_fastpm": b["P_shot_subtracted"],
                                   "ratio_ippl_over_fastpm": a["P_shot_subtracted"] / b["P_shot_subtracted"]})
        positions1, moments1 = read_snapshot(runs[leftName], Checkpoints)[1:]
        positions2, moments2 = read_snapshot(runs[rightName], Checkpoints)[1:]
        positionDifference = periodic_difference(positions1, positions2)
        momentumDifference = moments1 - moments2
        momentumReference = vector_rms(moments2)
        comparisons.append({"redshift_initial": z,
            "position_rms_mpc_over_h": vector_rms(positionDifference),
            "position_rms_mesh_cells": vector_rms(positionDifference) / (BoxSize / MeshGrid),
            "momentum_rms_relative_to_fastpm": vector_rms(momentumDifference) / momentumReference,
            "positive_shot_subtracted_power_shells": powerPairs,
            "maximum_absolute_power_fractional_difference": (
                max(abs(row["ratio_ippl_over_fastpm"] - 1) for row in powerPairs)
                if powerPairs else None)})

    output = {"schema": Schema, "campaign": str(campaign_path),
        "campaign_sha256": sourceReportHash,
        "campaign_complete": True, "campaign_passed_operational_checks": True,
        "protocol": environment,
        "initial_condition_manifest": {"path": str(icPath), "sha256": sha256(icPath),
            "coefficient_sha256": icManifest["coefficient_sha256"],
            "band": icManifest["mode_band"], "spectrum": icManifest["spectrum"],
            "sigma8": icManifest["cosmology"]["Sigma_8"],
            "cosmology": icManifest["cosmology"],
            "redshift_fixtures": [{key: value for key, value in row.items()
                                    if key in ("redshift_initial", "a_initial", "csv_sha256",
                                               "minimum_sampled_initial_map_eigenvalue",
                                               "minimum_sampled_initial_map_jacobian",
                                               "initial_displacement_vector_rms_mpc_over_h",
                                               "initial_max_displacement_over_particle_spacing",
                                               "momentum_rounding_relative_l2")}
                                    for row in icManifest["fixtures"]]},
        "definition": {"deposition": "periodic CIC on 128^3 mesh; counts normalized by mean",
            "fft": "rfftn; full real-mode multiplicities restored",
            "power": "P=V*<|delta_k|^2>, CIC window power deconvolved; Poisson 1/nbar subtracted",
            "shells": "integer |n| shells, centers from measured weighted mean |n| times 2pi/L; |n|<=48",
            "shot_noise_mpc_over_h_cubed": BoxSize**3 / ParticleGrid**3,
            "qualification": "measured realization and code differences; no new pass thresholds"},
        "runs": runAudits,
        "final_spectra": {name: rows[-1] for name, rows in allSpectra.items()},
        "checkpoint_spectra": {name: rows for name, rows in allSpectra.items()},
        "ippl_fastpm_comparisons": comparisons,
        "limitations": campaign["limitations"] + [
            "CIC window correction and Poisson shot-noise subtraction do not remove finite-mesh aliasing",
            "High-k bins approach the particle Nyquist and should be interpreted as characterization",
            "z99 versus z49 is a starting-epoch sensitivity, not a convergence-order proof"],
    }
    output_path = output_path.resolve()
    if not output_path.is_relative_to(Root) or output_path.exists():
        raise ValueError("Analysis output must be a new path under the remote campaign root")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Root / "campaign.json")
    parser.add_argument("--output", type=Path, default=Root / "analysis/results.json")
    args = parser.parse_args()
    result = analyze(args.campaign, args.output)
    print(json.dumps({"output": str(args.output), "campaign_sha256": result["campaign_sha256"],
                      "runs_audited": len(result["runs"]),
                      "paired_comparisons": result["ippl_fastpm_comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
