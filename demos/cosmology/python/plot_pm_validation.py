#!/usr/bin/env python3
## @file plot_pm_validation.py
# @brief Static scientific plots of saved PM-force and pre-crossing pancake evidence.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
"""Static scientific plots of saved PM-force and pre-crossing pancake evidence.

No simulations, fitted normalizations, or changed gates. Produces PNG/SVG,
the actual plotted numbers, and a source/output SHA256 manifest. A failed
pancake campaign is intentionally supported and its failures remain visible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, ScalarFormatter
import numpy as np
import pandas as pd


## @var Colors
# @brief Named Colors protocol/schema value; the source initializer records its exact contents.
Colors = {"ippl": "#0072B2", "fastpm": "#D55E00", "prediction": "#009E73",
          "raw": "#272E37", "residual": "#CC79A7"}
## @var GridColors
# @brief Named GridColors protocol/schema value; the source initializer records its exact contents.
GridColors = {16: "#999999", 32: "#E69F00", 64: "#0072B2"}
## @var FixtureLabels
# @brief Named FixtureLabels protocol/schema value; the source initializer records its exact contents.
FixtureLabels = {"common_band": "Single common-band mode", "axis": "Axis deformation · N16",
                 "oblique": "Oblique deformation", "jitter_wrapped": "Wrapped jitter",
                 "cluster": "Dense cluster", "axis_fine": "Axis deformation · N32",
                 "common_band_eds": "Common-band mode · Ωm = 1"}


## @brief Compute the streaming SHA256 digest of the file bytes.
# @see cosmology_tools
#
# @param path Input/output filesystem path; freshness and hash requirements are defined by this routine.
# @return Lowercase hexadecimal SHA256 of the exact retained bytes.
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


## @brief Evaluate the unique check helper in the documented module workflow.
# @see cosmology_tools
#
# @param campaign Retained campaign object/report with declared configuration, source hashes and run states.
# @param name Stable artifact/run/check identifier as defined by the caller.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def unique_check(campaign, name):
    matches = [row for row in campaign["checks"] if row["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Missing or duplicate check: {name}")
    return matches[0]


## @brief Evaluate the positive helper in the documented module workflow.
# @see cosmology_tools
#
# @param values Recorded diagnostic values in the metric/schema defined by the caller.
# @param label Stable human-readable curve or check label retained in the report.
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def positive(values, label):
    if not np.isfinite(values).all() or np.any(np.asarray(values) <= 0):
        raise ValueError(f"{label}: log plot needs finite positive data; do not invent a floor")


## @brief Evaluate the force data helper in the documented module workflow.
# @see cosmology_tools
#
# @param campaign Retained campaign object/report with declared configuration, source hashes and run states.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def force_data(campaign):
    rows, zeros = [], []
    for row in campaign["native_operator_differences"]:
        fixture, _, rank = row["name"].split("/")
        scale = row["ippl_force_rms"]
        if scale == 0:
            if row["raw_difference_rms"] != 0 or row["predicted_difference_rms"] != 0:
                raise ValueError("Zero IPPL force has nonzero cross-code difference")
            zeros.append(row["name"])
            continue
        residual = unique_check(campaign, row["name"] + "/predicted_native_difference")
        measured = {"fixture": fixture, "rank": int(rank[:-1]),
                    "raw": row["raw_difference_rms"] / scale,
                    "prediction": row["predicted_difference_rms"] / scale,
                    # Difference of fields from the saved check, NOT difference of RMS magnitudes.
                    "residual": residual["error_rms"] / scale}
        for code in ("ippl", "fastpm"):
            check = unique_check(campaign, f"{fixture}/{code}/{rank}/particle_force")
            measured[code] = check["relative_rms"]
        rows.append(measured)
    data = pd.DataFrame(rows)
    if set(data.fixture) != set(FixtureLabels):
        raise ValueError("Unexpected or missing nonzero frozen fixture")
    for fixture, group in data.groupby("fixture"):
        if sorted(group["rank"]) != [1, 2, 3, 4]:
            raise ValueError(f"Incomplete rank coverage for {fixture}")
    if sorted(zeros) != [f"uniform/cross_code/{r}r" for r in (1, 2, 3, 4)]:
        raise ValueError("Expected four exactly-zero uniform force cases")
    return data, zeros


## @brief Evaluate the save figure helper in the documented module workflow.
# @see cosmology_tools
#
# @param figure Matplotlib figure receiving verified saved evidence.
# @param directory Artifact directory following this module's ownership/freshness contract.
# @param name Stable artifact/run/check identifier as defined by the caller.
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def save_figure(figure, directory, name):
    figure.savefig(directory / f"{name}.png", dpi=210, facecolor="white")
    figure.savefig(directory / f"{name}.svg", facecolor="white")
    plt.close(figure)


## @brief Evaluate the force figure helper in the documented module workflow.
# @see cosmology_tools
#
# @param data Finite numerical or serialized record data in the routine's explicit schema.
# @param directory Artifact directory following this module's ownership/freshness contract.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def force_figure(data, directory):
    aggregate = data.groupby("fixture").max(numeric_only=True).reindex(FixtureLabels)
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True, layout="constrained")
    y = np.arange(len(aggregate))
    for code, marker, offset in (("ippl", "o", -.10), ("fastpm", "s", .10)):
        positive(aggregate[code], code)
        axes[0].scatter(aggregate[code], y + offset, color=Colors[code], marker=marker,
                        s=45, label="IPPL" if code == "ippl" else "Native FastPM", zorder=3)
    for key, marker, size, label in (("raw", "o", 52, "Raw code-to-code difference"),
                                     ("prediction", "o", 115, "Predicted from native operators"),
                                     ("residual", "x", 42, "Observed − predicted field")):
        positive(aggregate[key], key)
        kwargs = {"facecolors": "none", "edgecolors": Colors[key], "linewidths": 1.4} if key == "prediction" else {"color": Colors[key]}
        axes[1].scatter(aggregate[key], y, marker=marker, s=size, label=label, zorder=3, **kwargs)
    axes[0].set_yticks(y, [FixtureLabels[name] for name in aggregate.index])
    axes[0].invert_yaxis()
    minimum_error = aggregate[["ippl", "fastpm"]].to_numpy().min()
    axes[0].set(title="Each code versus its independent native oracle",
                xlabel="Particle-force relative RMS error", xlim=(minimum_error / 2, 2e-7))
    axes[1].set(title="Different native operators are not interchangeable",
                xlabel="Difference RMS / IPPL force RMS", xlim=(8e-10, 1.1))
    for axis in axes:
        axis.set_xscale("log")
        axis.grid(axis="y", visible=False)
        axis.set_ylim(len(aggregate)-.55, -.6)
    axes[0].legend(loc="center", fontsize=9, frameon=False)
    axes[1].legend(loc="upper right", fontsize=8.6, frameon=False)
    for name in ("oblique", "jitter_wrapped"):
        value = aggregate.loc[name, "raw"]
        axes[1].annotate(f"{100 * value:.2f}%", (value, list(aggregate.index).index(name)),
                         xytext=(8, 8), textcoords="offset points", fontsize=10, weight="bold")
    figure.suptitle("Frozen forces: oracle agreement and the Nyquist difference", fontsize=16, weight="bold")
    figure.supxlabel("Maximum over MPI ranks 1–4 for each quantity; saved frozen-force campaign passed. "
                      "No time evolution or native-kernel filtering.\n"
                      "Uniform forces are exactly zero on all ranks; their undefined relative errors are omitted. "
                      "Prediction includes native Nyquist and precision conventions.", fontsize=9)
    save_figure(figure, directory, "frozen_force_comparison")
    return aggregate.reset_index().to_dict(orient="records")


## @brief Evaluate the convergence data helper in the documented module workflow.
# @see cosmology_tools
#
# @param pancake Planar-evolution evidence or state; analytical comparison is valid only before shell crossing.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def convergence_data(pancake):
    spatial, temporal = [], []
    for amplitude in (.5, .8):
        for n in (16, 32, 64):
            name = f"axis_a{int(amplitude * 10)}_n{n}"
            final = pancake["trajectory_metrics"][name]["final"]
            for quantity in ("displacement", "momentum"):
                spatial.append({"amplitude": amplitude, "n": n, "quantity": quantity,
                                "relative_error": final[f"{quantity}_relative_error"]})
    final = pancake["trajectory_metrics"]["axis_a5_n32"]["final"]
    for quantity, reference in (("position", "displacement"), ("momentum", "momentum")):
        scale = final[f"reference_{reference}_rms"] / np.sqrt(3)
        first = unique_check(pancake, f"time {quantity}: nt16/32/64")
        second = unique_check(pancake, f"time {quantity}: nt32/64/128")
        if first["fine_per_component_rms_difference"] != second["coarse_per_component_rms_difference"]:
            raise ValueError("Saved overlapping timestep differences disagree")
        values = [first["coarse_per_component_rms_difference"],
                  first["fine_per_component_rms_difference"], second["fine_per_component_rms_difference"]]
        for steps, value in zip((16, 32, 64), values):
            temporal.append({"steps": steps, "quantity": quantity,
                             "relative_difference": value / scale})
    return pd.DataFrame(spatial), pd.DataFrame(temporal)


## @brief Evaluate the convergence figure helper in the documented module workflow.
# @see cosmology_tools
#
# @param spatial Retained spatial-resolution control results.
# @param temporal Retained timestep-refinement control results.
# @param directory Artifact directory following this module's ownership/freshness contract.
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def convergence_figure(spatial, temporal, directory):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    spatial_orders, temporal_orders = [], []
    for (amplitude, quantity), group in spatial.groupby(["amplitude", "quantity"]):
        values = group.sort_values("n").relative_error.to_numpy()
        positive(values, "spatial errors")
        spatial_orders.extend(np.log2(values[:-1] / values[1:]))
        color = Colors["ippl"] if amplitude == .5 else Colors["fastpm"]
        style, marker = ("-", "o") if quantity == "displacement" else ("--", "s")
        axes[0].plot(group.n, 100 * group.relative_error, color=color, linestyle=style,
                     marker=marker, label=f"A = {amplitude:g} · {quantity}")
    n = np.array([16, 32, 64])
    axes[0].plot(n, 7 * (16 / n)**2, ":", color="#737B84", label=r"$N^{-2}$ slope guide (not a fit)")
    axes[0].set(title="Spatial convergence · 128 timesteps", xlabel="Particles and mesh per dimension, N",
                ylabel="Trajectory RMS error / analytical RMS [%]")
    axes[0].text(.035, .035, f"Measured spatial order: {min(spatial_orders):.2f}–{max(spatial_orders):.2f}\n"
                 "Three-resolution study; inspect local errors too",
                 transform=axes[0].transAxes, fontsize=10, va="bottom",
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
    for quantity, group in temporal.groupby("quantity", sort=False):
        values = group.sort_values("steps").relative_difference.to_numpy()
        positive(values, "timestep differences")
        temporal_orders.extend(np.log2(values[:-1] / values[1:]))
        axes[1].plot(group.steps, 100 * group.relative_difference, marker="o" if quantity == "position" else "s",
                     color=Colors["ippl"] if quantity == "position" else Colors["fastpm"],
                     label="Position" if quantity == "position" else "Momentum")
    baseline = 100 * temporal[temporal.quantity == "position"].relative_difference.iloc[0]
    steps = np.array([16, 32, 64])
    axes[1].plot(steps, baseline * (16 / steps)**2, ":", color="#737B84", label=r"$n_t^{-2}$ slope guide")
    axes[1].set(title="Time convergence · N32, A = 0.5", xlabel=r"Coarse timestep count $n_t$ (compared with $2n_t$)",
                ylabel="Successive-solution difference / analytical RMS [%]")
    axes[1].text(.04, .035, f"Measured temporal order: {min(temporal_orders):.2f}–{max(temporal_orders):.2f}", transform=axes[1].transAxes,
                 fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
    for axis in axes:
        axis.set(xscale="log", yscale="log", xticks=[16, 32, 64])
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.xaxis.set_minor_locator(NullLocator())
        axis.legend(loc="upper right", fontsize=9, frameon=False)
    figure.suptitle("Analytical pancake: separate spatial and temporal convergence", fontsize=16, weight="bold")
    figure.supxlabel("Axis-aligned, one rank; z = 49 → 9; final deformation A = 0.5 or 0.8, before shell crossing.\n"
                      "Time differences use a fixed mesh, not continuum errors. Trajectory gates pass; "
                      "the full campaign retains one failed mass-diagnostic gate.", fontsize=9)
    save_figure(figure, directory, "pancake_convergence")


## @brief Evaluate the profile figure helper in the documented module workflow.
# @see cosmology_tools
#
# @param audit Saved completeness/diagnostic audit; its success does not by itself imply a physical accuracy pass.
# @param directory Artifact directory following this module's ownership/freshness contract.
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def profile_figure(audit, directory):
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.4), sharex=True, layout="constrained")
    for column, amplitude in enumerate((.5, .8)):
        for n in (16, 32, 64):
            profile = audit["local_profiles"][f"axis_a{int(amplitude * 10)}_n{n}"]
            q = np.asarray(profile["lagrangian_q_over_L"])
            actual = np.asarray(profile["numerical_displacement_x"])
            exact = np.asarray(profile["exact_displacement_x"])
            axes[0, column].plot(q, actual-exact, color=GridColors[n], marker="o", markersize=3,
                                 linewidth=1, label=f"N = {n}")
            axes[1, column].plot(profile["lagrangian_interval_midpoint_over_L"], profile["jacobian_error"],
                                 color=GridColors[n], marker="o", markersize=3, linewidth=1, label=f"N = {n}")
        finest = audit["local_profiles"][f"axis_a{int(amplitude * 10)}_n64"]
        axes[0, column].set_title(f"Final deformation A = {amplitude:g}")
        axes[0, column].legend(loc="upper right", fontsize=9, frameon=False)
        axes[1, column].text(.035, .04, f"N64 max |ΔJ| = {finest['jacobian_max_absolute_error']:.3f}",
                             transform=axes[1, column].transAxes, fontsize=10,
                             bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
        axes[1, column].set_xlabel(r"Lagrangian coordinate $q/L$ (interval midpoint for J)")
        for axis in axes[:, column]:
            axis.axhline(0, color="#59616A", linewidth=.6)
            axis.set(xlim=(0, 1), xticks=np.linspace(0, 1, 5))
    axes[0, 0].set_ylabel(r"Displacement error $s_{num}-s_{exact}$ [Mpc/$h$]")
    axes[1, 0].set_ylabel(r"Interval Jacobian error $J_{num}-J_{exact}$")
    figure.suptitle("Local pancake errors: narrower defects, persistent gradient peaks", fontsize=16, weight="bold")
    figure.supxlabel("One transverse row of the axis-aligned solution; z = 9, 128 timesteps. "
                      "J = 1 + Δs/Δq uses the same finite difference for numerical and exact maps.\n"
                      "Connecting lines guide the eye. This is not a CIC density plot; "
                      "local-gradient/density convergence is not established.", fontsize=9)
    save_figure(figure, directory, "pancake_local_errors")


## @brief Evaluate the mass figure helper in the documented module workflow.
# @see cosmology_tools
#
# @param pancake Planar-evolution evidence or state; analytical comparison is valid only before shell crossing.
# @param audit Saved completeness/diagnostic audit; its success does not by itself imply a physical accuracy pass.
# @param pancake_path Saved analytical-planar campaign/diagnostic path.
# @param directory Artifact directory following this module's ownership/freshness contract.
# @param hashes Absolute source/artifact paths mapped to expected SHA256 values; changed bytes invalidate provenance.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def mass_figure(pancake, audit, pancake_path, directory, hashes):
    figure, axis = plt.subplots(figsize=(11.5, 4.7), layout="constrained")
    rows, failed_samples = [], 0
    for amplitude in (.5, .8):
        name = f"axis_a{int(amplitude * 10)}_n64"
        path = pancake_path.parent / name / "output/diagnostics.csv"
        hashes[str(path)] = sha256(path)
        if hashes[str(path)] != audit["input_hashes_before"].get(str(path)):
            raise ValueError("Mass diagnostic no longer matches saved audit")
        data = pd.read_csv(path)
        error = data.mass_error.to_numpy()
        saved = unique_check(pancake, name + ": mass conservation")
        if not np.isclose(error.max(), saved["maximum_error"], rtol=1e-14, atol=0):
            raise ValueError("Mass history disagrees with saved validation")
        axis.plot(data.a, 1e12 * error, linewidth=1.1,
                  color=Colors["ippl"] if amplitude == .5 else Colors["fastpm"], label=f"A = {amplitude:g}")
        failed = error > saved["tolerance"]
        failed_samples += int(failed.sum())
        axis.scatter(data.a[failed], 1e12 * error[failed], color="#A93226", marker="x", s=48, zorder=4)
        for a, step, value in zip(data.a, data.step, error):
            rows.append({"amplitude": amplitude, "step": int(step), "a": float(a), "mass_error": float(value)})
    limit = pancake["tolerances"]["relative_mass"]
    axis.axhline(1e12 * limit, color="#A93226", linestyle="--", linewidth=1.2,
                 label=rf"Unchanged gate: ${limit * 1e12:.1f}\times10^{{-12}}$")
    peak = audit["mass_diagnostics"]["axis_a8_n64"]
    axis.annotate(f"Failed maximum: {peak['maximum_reported_mass_error'] * 1e12:.3f} × 10⁻¹²\n"
                   f"Step {peak['maximum_error_step']}; peak field was not saved",
                  xy=(peak["maximum_error_scale_factor"], 1e12 * peak["maximum_reported_mass_error"]),
                  xytext=(.044, 2.36), fontsize=10, color="#8E2A20",
                  arrowprops={"arrowstyle": "->", "color": "#8E2A20"})
    axis.set(xlabel=r"Scale factor $a$ (z = 49 → 9)", ylabel=r"Reported relative mass error [$10^{-12}$]",
             xlim=(.02, .10), ylim=(0, 2.68))
    axis.legend(loc="upper right", fontsize=9, frameon=False)
    figure.suptitle("Mass diagnostic: a small but retained acceptance failure", fontsize=16, weight="bold")
    count = len(pancake["checks"])
    passed = sum(check["passed"] for check in pancake["checks"])
    figure.supxlabel(f"N64, one rank; crosses mark {failed_samples} samples above the fixed gate. "
                      f"The full pancake campaign passes {passed} of {count} checks, not all checks.\n"
                      "Accurate endpoint redeposition sums recover the expected mass; "
                      "that does not verify the unsaved intermediate maximum. No tolerance was relaxed.", fontsize=9)
    save_figure(figure, directory, "pancake_mass_diagnostic")
    return rows


## @brief Parse the documented command-line interface and execute its selected workflow, retaining nonzero failures.
# @see cosmology_tools
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--pancake", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.frozen.resolve(), args.pancake.resolve(), args.audit.resolve()]
    frozen, pancake, audit = [json.loads(path.read_text()) for path in paths]
    if not frozen.get("complete") or not frozen.get("passed") or pancake.get("quick"):
        raise ValueError("Requires a complete passing frozen campaign and full pancake campaign")
    if Path(audit["campaign"]).resolve() != paths[1].parent:
        raise ValueError("Audit belongs to a different pancake campaign")
    if sha256(paths[1]) != audit["input_hashes_before"].get(str(paths[1])):
        raise ValueError("Pancake results no longer match audit")
    failed = [row for row in pancake["checks"] if not row["passed"]]
    if failed != audit["original_failed_gates"]:
        raise ValueError("Audit/campaign failure lists differ")
    if len(failed) != 1 or failed[0]["name"] != "axis_a8_n64: mass conservation":
        raise ValueError("These figure annotations require the recorded single mass-diagnostic failure")
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Output directory must be new or empty")
    hashes = {str(path): sha256(path) for path in [*paths, Path(__file__).resolve()]}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .20, "grid.linewidth": .6,
                         "axes.titleweight": "semibold", "svg.fonttype": "none"})
    force, zeros = force_data(frozen)
    aggregate = force_figure(force, directory)
    spatial, temporal = convergence_data(pancake)
    convergence_figure(spatial, temporal, directory)
    profile_figure(audit, directory)
    mass = mass_figure(pancake, audit, paths[1], directory, hashes)
    data = {"force_per_rank": force.to_dict(orient="records"), "force_rank_maxima": aggregate,
            "exact_zero_uniform_cases": zeros, "spatial": spatial.to_dict(orient="records"),
            "temporal": temporal.to_dict(orient="records"), "local_profiles": audit["local_profiles"],
            "mass_history": mass, "retained_failed_gates": failed}
    (directory / "plot_data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    if any(sha256(path) != value for path, value in hashes.items()):
        raise RuntimeError("A plot source changed during rendering")
    manifest = {"source_sha256": hashes,
                "output_sha256": {path.name: sha256(path) for path in sorted(directory.iterdir())},
                "notes": ["Saved evidence only; no simulations or changed tolerances",
                          "Force residual is field-difference RMS, never difference of RMS magnitudes",
                          "Uniform relative force errors are undefined, not floored",
                          "Frozen operator qualification passes; full pancake retains mass failure",
                          "No post-crossing, local-density, GPU or exascale qualification"]}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(directory)


## @cond CLI_DISPATCH
if __name__ == "__main__":
    main()
## @endcond
