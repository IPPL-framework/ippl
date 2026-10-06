#!/usr/bin/env python3
## @file plot_zarija.py
# @brief Reproduce static scientific figures from an existing matched-Zarija campaign.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
"""Reproduce static scientific figures from an existing matched-Zarija campaign.

No simulations are run and no acceptance limits are changed. Power is recovered
from Lagrangian IC displacements, not from an Eulerian density deposition.
Output: PNG/SVG figures, plotted numbers, and source/artifact hashes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

## @cond RUNTIME_SETTINGS
sys.dont_write_bytecode = True
## @endcond

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validate_linear import bbks_spectrum, growth_reference
from validate_zarija import Geometry, read_ippl, read_zarija, recover, sha256, table_spectrum


## @var Colors
# @brief Named Colors protocol/schema value; the source initializer records its exact contents.
Colors = {"ippl": "#0072B2", "zarija": "#D55E00"}
## @var Markers
# @brief Named Markers protocol/schema value; the source initializer records its exact contents.
Markers = {"ippl": "o", "zarija": "s"}


## @brief Evaluate the save figure helper in the documented module workflow.
# @see cosmology_tools
#
# @param figure Matplotlib figure receiving verified saved evidence.
# @param directory Artifact directory following this module's ownership/freshness contract.
# @param name Stable artifact/run/check identifier as defined by the caller.
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def save_figure(figure, directory, name):
    figure.savefig(directory/f"{name}.png", dpi=220, facecolor="white")
    figure.savefig(directory/f"{name}.svg", facecolor="white")
    plt.close(figure)


## @brief Evaluate the shell data helper in the documented module workflow.
# @see cosmology_tools
#
# @param campaign Retained campaign object/report with declared configuration, source hashes and run states.
# @param resultPath Destination for the retained finite JSON comparison report.
# @param sourceHashes Expected source/artifact hash map frozen before execution.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def shell_data(campaign, resultPath, sourceHashes):
    parameters = campaign["parameters"]
    geometry = Geometry(parameters["np"], parameters["box_size"])
    table = resultPath.parent/"provenance"/"transfer.tf"
    sourceHashes[str(table)] = sha256(table)
    if sourceHashes[str(table)] != campaign["provenance"]["captured_transfer_table"]["sha256"]:
        raise ValueError("Captured transfer table no longer matches the validated campaign")
    wave = np.sqrt(geometry.k2[geometry.unique])
    radius = geometry.radius[geometry.unique]
    edges = campaign["shell_edges_integer_k"]
    rows = []
    # Show z=49 alone: z=200 repeats the same phases and is not an extra ensemble.
    for flag in campaign["transfer_flags"]:
        theory = (bbks_spectrum(wave, parameters) if flag == 4 else
                  table_spectrum(wave, parameters, table))
        for code in ("ippl", "zarija"):
            runs = [run for run in campaign["runs"] if run["code"] == code and run["ranks"] == 1
                    and run["parameters"]["TFFlag"] == flag and run["parameters"]["z_in"] == 49]
            if sorted(run["parameters"]["seed"] for run in runs) != sorted(campaign["seeds"]):
                raise ValueError("Incomplete or duplicated independent-seed ensemble")
            powers = []
            for run in runs:
                directory = resultPath.parent/run["name"]
                a = 1/(1+run["parameters"]["z_in"])
                paths = (sorted((directory/"output").glob("particles_initial_rank*.csv"))
                         if code == "ippl" else sorted(directory.glob("particles.bin.*")))
                sourceHashes.update({str(path): sha256(path) for path in paths})
                snapshot = (read_ippl(directory/"output", 1, geometry) if code == "ippl" else
                            read_zarija(directory/"particles", 1, geometry, a))
                delta, _ = recover(snapshot, geometry, code, a, parameters["Omega_m"])
                powers.append(geometry.box**3*np.abs(delta[geometry.unique])**2)
            powers = np.asarray(powers)
            for lower, upper in zip(edges[:-1], edges[1:]):
                mask = (radius >= lower) & (radius < upper)
                count = int(mask.sum())
                pairs = len(runs)*count
                measuredRatio = float(np.mean(powers[:, mask]/theory[mask]))
                checkName = f"{code}_tf{flag}_z49: power shell [{lower:g},{upper:g})"
                checks = [check for check in campaign["checks"] if check["name"] == checkName]
                if len(checks) != 1 or checks[0]["independent_pairs"] != pairs:
                    raise ValueError(f"Missing/mismatched saved shell check: {checkName}")
                if abs(measuredRatio-checks[0]["mean_power"]) > 1e-12:
                    raise ValueError("Reconstructed power disagrees with the saved validation")
                rows.append({"code": code, "transfer_flag": flag, "z": 49,
                             "k_mean": float(wave[mask].mean()), "pairs": pairs,
                             "shell_lower": lower, "shell_upper": upper,
                             "measured_power": float(powers[:, mask].mean()),
                             "expected_power": float(theory[mask].mean()),
                             # Exact Gaussian shell variance for varying P(k).
                             "power_standard_error": float(np.sqrt(np.sum(theory[mask]**2))
                                                            /(np.sqrt(len(runs))*count)),
                             "mean_power_ratio": measuredRatio,
                             "ratio_standard_error": 1/np.sqrt(pairs),
                             "ratio_acceptance_half_width": checks[0]["tolerance"]})
    return pd.DataFrame(rows)


## @brief Evaluate the power figure helper in the documented module workflow.
# @see cosmology_tools
#
# @param data Finite numerical or serialized record data in the routine's explicit schema.
# @param campaign Retained campaign object/report with declared configuration, source hashes and run states.
# @param directory Artifact directory following this module's ownership/freshness contract.
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def power_figure(data, campaign, directory):
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex="col",
                                gridspec_kw={"height_ratios": [1.4, 1]}, layout="constrained")
    for column, (flag, label) in enumerate(((4, "BBKS"), (0, "Supplied transfer table"))):
        expected = data[(data.transfer_flag == flag) & (data.code == "ippl")]
        axes[0, column].plot(expected.k_mean, expected.expected_power, color="#333333",
                             linewidth=1.4, marker="_", markersize=13, label="Independent shell expectation")
        ref = data[(data.transfer_flag == flag) & (data.code == "zarija")]
        limit = 100*ref.ratio_acceptance_half_width.to_numpy()
        axes[1, column].fill_between(ref.k_mean, -limit, limit, color="#e9edf1",
                                     label=r"Acceptance: $6\sigma$ + reference floor")
        axes[1, column].axhline(0, color="#5b6570", linewidth=1)
        for code, offset in (("ippl", 0.985), ("zarija", 1.015)):
            part = data[(data.transfer_flag == flag) & (data.code == code)]
            style = dict(fmt=Markers[code], color=Colors[code], markersize=5,
                         capsize=3, elinewidth=1.3, label=code.upper() if code == "ippl" else "Zarija")
            axes[0, column].errorbar(part.k_mean*offset, part.measured_power,
                                    yerr=part.power_standard_error, **style)
            axes[1, column].errorbar(part.k_mean*offset, 100*(part.mean_power_ratio-1),
                                    yerr=100*part.ratio_standard_error, **style)
        axes[0, column].set(title=label, yscale="log", xscale="log")
        axes[1, column].set(xscale="log", xlabel=r"Wavenumber $k$ [$h\,\mathrm{Mpc}^{-1}$]")
        axes[0, column].legend(fontsize=8.5, loc="upper right")
        axes[1, column].legend(fontsize=8.2, loc="lower right")
    axes[0, 0].set_ylabel(r"Linear IC $P_0(k)$ [$(\mathrm{Mpc}/h)^3$]")
    axes[1, 0].set_ylabel(r"Mean per-mode power excess [%]")
    figure.suptitle("Matched initial-condition power spectra", fontsize=17, weight="bold")
    figure.supxlabel(
        f"z = 49; {len(campaign['seeds'])} seeds per code; error bars: Gaussian 1σ. "
        "Horizontal offsets only separate markers.\n"
        "Recovered from 1LPT displacements, extrapolated to z = 0; not evolved Eulerian power. "
        "DC / Nyquist planes excluded.", fontsize=9)
    save_figure(figure, directory, "matched_ic_power")


## @brief Evaluate the growth figure helper in the documented module workflow.
# @see cosmology_tools
#
# @param scalar Measured scalar diagnostic in the caller's units.
# @param parameters Named protocol or cosmology parameters; unsupported keys are rejected by the calling validator.
# @param directory Artifact directory following this module's ownership/freshness contract.
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def growth_figure(scalar, parameters, directory):
    figure, axes = plt.subplots(2, 1, figsize=(8.8, 7), sharex=True,
                                gridspec_kw={"height_ratios": [1.3, 1]}, layout="constrained")
    growth = scalar[scalar.quantity == "growth_D"].sort_values("argument", ascending=False)
    a = 1/(1+growth.argument.to_numpy())
    curveA = np.geomspace(a.min(), 1, 200)
    curveD = [growth_reference(value, parameters["Omega_m"])[0] for value in curveA]
    axes[0].plot(curveA, curveD, color="#6c737b", linewidth=1.3, label="Independent growing-mode integral")
    axes[0].plot(a, growth.ippl, "o", color=Colors["ippl"], markersize=8, label="IPPL")
    axes[0].plot(a, growth.zarija, "s", color=Colors["zarija"], markerfacecolor="none", markersize=10, label="Zarija")
    axes[0].set(yscale="log", ylabel=r"Growth factor $D(a)$, with $D(1)=1$")
    axes[0].legend(loc="upper left", fontsize=9)
    for quantity, label, color, marker in (
            ("growth_D", r"$D$", "#0072B2", "o"),
            ("growth_Ddot", r"$\mathrm{d}D/\mathrm{d}(H_0t)$", "#D55E00", "s"),
            ("growth_f", r"$f=\mathrm{d}\ln D/\mathrm{d}\ln a$", "#009E73", "^")):
        part = scalar[scalar.quantity == quantity].sort_values("argument", ascending=False)
        axes[1].plot(1/(1+part.argument), 1e6*(part.zarija/part.ippl-1),
                     color=color, marker=marker, linewidth=1.2, label=label)
    axes[1].axhline(0, color="#6c737b", linewidth=1)
    axes[1].set(ylabel="(Zarija / IPPL − 1) [ppm]", xlabel=r"Scale factor $a=1/(1+z)$")
    axes[1].legend(fontsize=9, loc="lower right")
    axes[1].text(0.025, 0.94, "Acceptance: ±20 ppm (outside this zoom)",
                 transform=axes[1].transAxes, va="top", fontsize=9, color="#525b66")
    axes[0].set_xscale("log")
    figure.suptitle("Linear growth: agreement below one part per million", fontsize=16, weight="bold")
    figure.supxlabel("Background functions, not a measured particle-evolution trajectory. "
                      "Markers are the four tested redshifts; connecting lines guide the eye.", fontsize=9)
    save_figure(figure, directory, "matched_growth")


## @brief Evaluate the rank figure helper in the documented module workflow.
# @see cosmology_tools
#
# @param campaign Retained campaign object/report with declared configuration, source hashes and run states.
# @param directory Artifact directory following this module's ownership/freshness contract.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def rank_figure(campaign, directory):
    rows = []
    for flag in campaign["transfer_flags"]:
        for redshift in campaign["redshifts"]:
            for ranks in (2, 3, 4):
                prefix = f"ippl_tf{flag}_z{redshift:g}_s{campaign['seeds'][0]}_r{ranks}:"
                matching = [check for check in campaign["checks"] if check["name"].startswith(prefix)
                            and "identical initial phase space" in check["name"]]
                if len(matching) != 1:
                    raise ValueError("Missing or duplicate rank comparison")
                rows.append({"transfer_flag": flag, "redshift": redshift, "ranks": ranks,
                             **{key: matching[0][key] for key in
                                ("position_rms", "momentum_rms", "position_limit", "momentum_limit")}})
    data = pd.DataFrame(rows)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, layout="constrained")
    styles = zip(data.groupby(["transfer_flag", "redshift"], sort=False),
                 ("#0072B2", "#56B4E9", "#D55E00", "#CC79A7"), ("o", "s", "^", "D"))
    for ((flag, redshift), part), color, marker in styles:
        label = f"{'BBKS' if flag == 4 else 'Table'}, z = {redshift:g}"
        for axis, name in zip(axes, ("position", "momentum")):
            fraction = part[f"{name}_rms"]/part[f"{name}_limit"]
            # Do not silently floor zeros to positive numbers on a logarithmic axis.
            if (fraction <= 0).any():
                raise ValueError("Exact-zero rank residual requires an explicitly labelled log-axis marker")
            axis.plot(part.ranks, fraction, color=color, marker=marker, linewidth=1.1, label=label)
    for axis, label in zip(axes, ("Particle positions", "Canonical momenta")):
        axis.axhline(1, linestyle="--", color="#5b6570", linewidth=1, label="Acceptance limit")
        axis.set(title=label, xlabel="MPI ranks (compared with 1 rank)", yscale="log", xticks=[2, 3, 4])
        axis.legend(fontsize=8.3, loc="center right")
    axes[0].set_ylabel("RMS difference / acceptance limit")
    figure.suptitle("IPPL initial conditions are MPI-rank consistent", fontsize=16, weight="bold")
    figure.supxlabel("Same seed and cosmology; particles matched by ID, with periodic position differences.\n"
                      "This checks numerical reproducibility, not parallel performance or scaling.", fontsize=9)
    save_figure(figure, directory, "mpi_rank_consistency")
    return data


## @brief Parse the documented command-line interface and execute its selected workflow, retaining nonzero failures.
# @see cosmology_tools
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True, help="Initializer results.json")
    parser.add_argument("--physics", type=Path, required=True, help="Scalar-probe results.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    campaignPath, physicsPath = args.campaign.resolve(), args.physics.resolve()
    campaign = json.loads(campaignPath.read_text())
    physics = json.loads(physicsPath.read_text())
    if not campaign["passed"] or not physics["passed"]:
        raise ValueError("This qualification plot requires passing source campaigns")
    directory = args.output_dir.resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Output directory must be new or empty")
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.6,
                         "axes.titleweight": "semibold", "svg.fonttype": "none"})
    hashes = {str(path): sha256(path) for path in (campaignPath, physicsPath, Path(__file__))}
    for name, filename in (("analysis_script", "validate_zarija.py"),
                           ("independent_reference_script", "validate_linear.py")):
        path = Path(__file__).with_name(filename)
        hashes[str(path)] = sha256(path)
        if hashes[str(path)] != campaign["provenance"][name]["sha256"]:
            raise ValueError(f"{filename} no longer matches the validated analysis")
    scalarPath = physicsPath.parent/"bbks"/"comparison.csv"
    hashes[str(scalarPath)] = sha256(scalarPath)
    scalar = pd.read_csv(scalarPath)
    data = shell_data(campaign, campaignPath, hashes)
    power_figure(data, campaign, directory)
    growth_figure(scalar, campaign["parameters"], directory)
    ranks = rank_figure(campaign, directory)
    growth = scalar[scalar.quantity.str.startswith("growth_")]
    # Unused CSV note cells may be missing; strict JSON represents these as null.
    plotted = {"shell_data": data.to_dict(orient="records"), "rank_data": ranks.to_dict(orient="records"),
               "growth_data": growth.astype(object).where(pd.notna(growth), None).to_dict(orient="records")}
    (directory/"plot_data.json").write_text(json.dumps(plotted, indent=2, allow_nan=False)+"\n")
    manifest = {"source_sha256": hashes, "notes": [
        "z49 only in power figure; z200 shares seeds and is not pooled",
        "P0 recovered from initial Lagrangian displacements, not Eulerian density",
        "Error bars are theoretical Gaussian1sigma; lower greyband saved reference6sigma+floor",
        "Mean of modewise ratios is distinct from ratio of raw shell means",
        "Componentwise Nyquist exclusion; highest shells have incomplete angular coverage"],
        "outputs_sha256": {path.name: sha256(path) for path in sorted(directory.iterdir()) if path.is_file()}}
    (directory/"plot_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Created three PNG/SVG figures and numerical plot data in {directory}")


## @cond CLI_DISPATCH
if __name__ == "__main__":
    main()
## @endcond
