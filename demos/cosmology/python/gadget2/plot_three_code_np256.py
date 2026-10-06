#!/usr/bin/env python3
## @file plot_three_code_np256.py
# @brief Plot matched 256^3-particle IPPL, FastPM, and GADGET-2 z=0 spectra.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
# @see cosmology_model cosmology_numerics cosmology_spectra
"""Plot matched 256^3-particle IPPL, FastPM, and GADGET-2 z=0 spectra."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


## @var SharedRoot
# @brief Named SharedRoot protocol/schema value; the source initializer records its exact contents.
SharedRoot = Path("/data/user/adelmann/cosmology-zeldovich-np256-20261006")
## @var GadgetRoot
# @brief Named GadgetRoot protocol/schema value; the source initializer records its exact contents.
GadgetRoot = Path("/data/user/adelmann/gadget2-zeldovich-np256-20261006")
## @var InputCsvSha256
# @brief Named InputCsvSha256 protocol/schema value; the source initializer records its exact contents.
InputCsvSha256 = "3b4a3e1864ad535369444b98779c117750e1e980cbe7d01afedc20f8329cda11"
## @var Schema
# @brief Named Schema protocol/schema value; the source initializer records its exact contents.
Schema = "ippl-fastpm-gadget2-np256-z99-z0-power-v1"


## @brief Compute the streaming SHA256 digest of the file bytes.
# @see cosmology_tools
#
# @param path Input/output filesystem path; freshness and hash requirements are defined by this routine.
# @return Lowercase hexadecimal SHA256 of the exact retained bytes.
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


## @brief Render the documented module workflow.
# @see cosmology_tools
#
# @param campaign_path Retained campaign JSON path; completion and provenance are verified before scientific analysis.
# @param analysis_path Retained analysis JSON path whose originating campaign/hash must match.
# @param output_dir Output directory; use a fresh location when required by the workflow.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def render(campaign_path: Path, analysis_path: Path, output_dir: Path) -> dict:
    campaign_path = campaign_path.resolve(strict=True)
    analysis_path = analysis_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    if (not campaign_path.is_relative_to(GadgetRoot)
            or not analysis_path.is_relative_to(GadgetRoot)
            or not output_dir.is_relative_to(GadgetRoot) or output_dir.exists()):
        raise ValueError("reports and a fresh plot directory must stay under the Gadget campaign root")
    campaign_bytes = campaign_path.read_bytes()
    analysis_bytes = analysis_path.read_bytes()
    campaign, gadget = json.loads(campaign_bytes), json.loads(analysis_bytes)
    run = campaign.get("run", {})
    if (campaign.get("status") != "complete" or campaign.get("complete") is not True
            or gadget.get("campaign_complete") is not True
            or gadget.get("campaign_sha256") != hashlib.sha256(campaign_bytes).hexdigest()
            or gadget.get("shared_input_csv_sha256") != InputCsvSha256
            or gadget.get("particle_grid") != 256 or gadget.get("mesh_grid") != 256
            or gadget.get("cutoff") != 96):
        raise ValueError("GADGET campaign/analysis is incomplete or mismatched")
    if (sha256(Path(gadget["spectrum_estimator_source"]))
            != gadget["spectrum_estimator_source_sha256"]):
        raise ValueError("shared CIC spectrum source changed after Gadget analysis")
    ic_manifest_path = SharedRoot / "ics/ic-manifest.json"
    ic_path = SharedRoot / "ics/shared-z99.csv"
    reference_manifest = SharedRoot / "plots/z99-np256-ippl-fastpm-v1/manifest.json"
    reference_values_path = SharedRoot / "plots/z99-np256-ippl-fastpm-v1/plotted_values.json"
    ippl_run_path = SharedRoot / "runs/z99-ippl-a100-p256-m256/run.json"
    fastpm_run_path = SharedRoot / "runs/z99-fastpm-cpu-p256-m256/run.json"
    reference_manifest_data = json.loads(reference_manifest.read_text())
    reference = json.loads(reference_values_path.read_text())
    if (reference_manifest_data.get("campaign_complete") is not True
            or reference["shared_input_sha256"] != InputCsvSha256
            or reference["particle_grid"] != 256 or reference["force_mesh_grid"] != 256
            or reference["ic_cutoff_fundamental"] != 96):
        raise ValueError("matched IPPL/FastPM reference spectrum is not valid for this run")
    inputs = {str(path): sha256(path) for path in (campaign_path, analysis_path,
        ic_manifest_path, ic_path, reference_manifest, reference_values_path,
        ippl_run_path, fastpm_run_path, Path(run["executable"]),
        Path(gadget["snapshot"]["path"]), Path(gadget["spectrum_estimator_source"]))}
    for item in campaign.get("reference_audit", {}).get("fastpm_final_shards", []):
        shard = Path(item["path"])
        inputs[str(shard)] = sha256(shard)
        if inputs[str(shard)] != item["sha256"]:
            raise ValueError(f"FastPM final shard changed after the matched comparison: {shard}")
    if inputs[str(ic_path)] != InputCsvSha256:
        raise ValueError("shared initial condition hash mismatch")
    if inputs[str(Path(gadget["snapshot"]["path"]))] != gadget["snapshot"]["sha256"]:
        raise ValueError("GADGET snapshot changed after analysis")
    for name, expected in reference_manifest_data["input_hashes"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"IPPL/FastPM source evidence changed: {name}")
    if sha256(Path(run["executable"])) != run["executable_sha256"]:
        raise ValueError("GADGET executable hash changed after the run")

    shells = np.asarray(reference["shells"], dtype=np.int32)
    k = np.asarray(reference["k_h_per_mpc"], dtype=np.float64)
    spectra_ref = reference["powers_shot_subtracted_mpc_over_h_cubed"]
    labels = {"ippl": "IPPL · A100, one rank · z=0",
        "fastpm": "FastPM · CPU, 8 ranks · z=0", "gadget2": "GADGET-2 · TreePM · z=0"}
    powers = {"ippl": np.asarray(spectra_ref[labels["ippl"]], dtype=np.float64),
        "fastpm": np.asarray(spectra_ref[labels["fastpm"]], dtype=np.float64),
        "gadget2": np.asarray([row["P_shot_subtracted"] for row in gadget["final_spectrum"]],
                               dtype=np.float64)}
    gadget_shells = [row["shell"] for row in gadget["final_spectrum"]]
    gadget_k = np.asarray([row["k_h_per_mpc"] for row in gadget["final_spectrum"]])
    if (len(shells) != 96 or gadget_shells != shells.tolist()
            or not np.allclose(k, gadget_k, rtol=0, atol=1e-14)):
        raise ValueError("GADGET and PM codes do not share all 96 CIC shell centers")
    for name, record_path, hash_key in (("ippl", ippl_run_path, "ippl_run_json_sha256"),
                                        ("fastpm", fastpm_run_path, "fastpm_run_json_sha256")):
        if sha256(record_path) != run[hash_key]:
            raise ValueError(f"{name} run record changed since GADGET validation")

    colors = {"ippl": "#0072B2", "fastpm": "#D55E00", "gadget2": "#009E73"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    fig, (axis, ratio_axis) = plt.subplots(2, 1, figsize=(9.3, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": .08})
    positive = {name: values > 0 for name, values in powers.items()}
    for name in ("ippl", "fastpm", "gadget2"):
        axis.plot(k[positive[name]], powers[name][positive[name]], color=colors[name],
                  lw=1.8, marker="o", ms=2.8, markevery=4, label=labels[name])
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_ylabel(r"$P(k)\ [ (\mathrm{Mpc}/h)^3 ]$")
    axis.set_title("z = 0 matter power · matched 256³ Zel’dovich realization", pad=12)
    axis.grid(True, which="major", color="#dddddd", lw=.55)
    axis.legend(loc="best", framealpha=.96)
    ratio_rows = {}
    for name in ("ippl", "fastpm"):
        valid = positive[name] & positive["gadget2"]
        ratio = powers[name][valid] / powers["gadget2"][valid] - 1.0
        ratio_axis.plot(k[valid], 100 * ratio, color=colors[name], lw=1.5,
                        marker="o", ms=2.5, markevery=4,
                        label=f"{labels[name].split(' · ')[0]} / GADGET-2 − 1")
        ratio_rows[name] = {"shells": shells[valid].tolist(),
            "k_h_per_mpc": k[valid].tolist(), "relative_power_difference": ratio.tolist(),
            "max_abs_relative_difference": float(np.max(np.abs(ratio))),
            "rms_relative_difference": float(np.sqrt(np.mean(ratio**2)))}
    ratio_axis.axhline(0, color="#555555", lw=.8, zorder=-2)
    ratio_axis.set_xscale("log")
    ratio_axis.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    ratio_axis.set_ylabel("Power offset [%]")
    ratio_axis.grid(True, which="major", color="#dddddd", lw=.55)
    ratio_axis.legend(loc="best", framealpha=.96, fontsize=8.5)
    ratio_axis.set_xlim(k[0] * .92, k[-1] * 1.08)
    note = ("Same 256³ phase-space CSV (SHA-256 3b4a3e1864ad…), seed 20261003; "
        "zᵢ=99, |n|≤96, L=168.75 Mpc/h. All: 256³ particles and common 256³ CIC analysis mesh.\n"
        "IPPL/FastPM: 2400 steps; GADGET-2: adaptive synchronized TreePM steps, short-range tree; "
        "softening 13.18 kpc/h. Common CIC assignment, window deconvolution and Poisson shot-noise subtraction.\n"
        "One realization; compare spectra, not trajectories. High-|n| modes 49–96 extend the earlier 128³ IC band.")
    fig.text(.5, .012, note, ha="center", va="bottom", fontsize=7.4, linespacing=1.25)
    fig.subplots_adjust(left=.13, right=.97, top=.92, bottom=.22)

    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = []
    try:
        for extension in ("png", "svg"):
            target = output_dir / f"three-code-np256-z99-z0-power.{extension}"
            fig.savefig(target, dpi=240, metadata={"Date": None} if extension == "svg" else None)
            outputs.append(target)
    finally:
        plt.close(fig)
    values = {"schema": Schema, "input_hashes": inputs,
        "shared_input_csv_sha256": InputCsvSha256, "particle_grid": 256,
        "force_mesh_grid": 256, "initial_redshift": 99, "final_redshift": 0,
        "shells": shells.tolist(), "k_h_per_mpc": k.tolist(),
        "powers_shot_subtracted_mpc_over_h_cubed": powers,
        "positive_shells": {name: mask.tolist() for name, mask in positive.items()},
        "relative_power_differences_vs_gadget": ratio_rows,
        "limitations": gadget["limitations"]}
    values_path = output_dir / "plotted_values.json"
    values_path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")
    outputs.append(values_path)
    for name, expected in inputs.items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"input changed during plot generation: {name}")
    manifest_runs = {
        "ippl": {"force_model": "IPPL periodic Particle-Mesh", "wall_seconds": None},
        "fastpm": {"force_model": "FastPM plain Particle-Mesh", "wall_seconds": None},
        "gadget2": {"force_model": gadget["force_model"],
                    "wall_seconds": run.get("solver_wall_seconds")}}
    manifest = {"schema": Schema, "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_complete": True, "shared_input_csv_sha256": InputCsvSha256,
        "inputs": inputs, "runs": manifest_runs,
        "power_definition": gadget["spectrum_definition"],
        "limitations": gadget["limitations"],
        "outputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                    for path in outputs]}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return {"manifest": manifest, "manifest_path": str(manifest_path),
        "plot": str(outputs[0]), "max_abs_relative_difference_vs_ippl": ratio_rows["ippl"]["max_abs_relative_difference"],
        "rms_relative_difference_vs_ippl": ratio_rows["ippl"]["rms_relative_difference"],
        "max_abs_relative_difference_vs_fastpm": ratio_rows["fastpm"]["max_abs_relative_difference"],
        "rms_relative_difference_vs_fastpm": ratio_rows["fastpm"]["rms_relative_difference"]}


## @cond CLI_DISPATCH
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=GadgetRoot / "campaign.json")
    parser.add_argument("--analysis", type=Path,
        default=GadgetRoot / "analysis/gadget2-np256-z99-z0-results.json")
    parser.add_argument("--output-dir", type=Path,
        default=GadgetRoot / "plots/three-code-np256-z99-z0-v1")
    args = parser.parse_args()
    print(json.dumps(render(args.campaign, args.analysis, args.output_dir), indent=2, allow_nan=False))
## @endcond
