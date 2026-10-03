#!/usr/bin/env python3
"""Render two static summaries of a completed matched-evolution campaign.

No simulations, fitted normalizations, analytic post-crossing reference, or
zero floors are introduced. Failed qualification checks remain visible. Both
compressed and plain rank snapshots are accepted, with provenance verified.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd


Codes = ("ippl", "fastpm")
Fixtures = ("pancake", "coupled3d")
Meshes = (16, 32, 64)
MeshColors = {16: "#D55E00", 32: "#0072B2", 64: "#009E73"}
CodeColors = {"ippl": "#0072B2", "fastpm": "#D55E00"}
Columns = ["id", "x", "y", "z", "px", "py", "pz"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def direct_metrics(left, right):
    """Exact recorded mode-vector norms; zero signals have undefined ratios."""
    left, right = np.asarray(left), np.asarray(right)
    if (left.ndim != 1 or left.shape != right.shape or not len(left)
            or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError("Expected matching finite nonempty mode vectors")
    power_left = float(np.vdot(left, left).real)
    power_right = float(np.vdot(right, right).real)
    absolute = float(np.linalg.norm(left - right))
    defined = min(power_left, power_right) > 1e-28  # campaign's declared definition
    return {"power_ippl": power_left, "power_fastpm": power_right,
            "absolute_complex_difference": absolute,
            "complex_relative": absolute / (power_left * power_right)**.25 if defined else None,
            "normalization_defined": defined}


def campaign_status(report):
    if report.get("schema") != "ippl-fastpm-evolution-v1" or not report.get("complete"):
        raise ValueError("A complete ippl-fastpm-evolution-v1 results.json is required")
    failures = [check for check in report["checks"] if not check["passed"]]
    if failures != report["failed_checks"] or bool(report["passed"]) != (not failures):
        raise ValueError("Campaign status and recorded failed checks disagree")
    return failures


def selected_run(report, fixture, mesh, code):
    matches = [run for run in report["runs"] if
               (run["fixture"], run["mesh"], run["steps"], run["ranks"], run["code"])
               == (fixture, mesh, 256, 1, code)]
    if len(matches) != 1:
        raise ValueError(f"Need exactly one completed {fixture}, NM={mesh}, nt=256, r1, {code} run")
    return matches[0]


def snapshot(run, checkpoint, count, input_hashes):
    directory = Path(run["output"])
    expected = {f"particles_checkpoint{checkpoint:04d}_rank{rank}.csv" for rank in range(run["ranks"])}
    paths = sorted(directory.glob(f"particles_checkpoint{checkpoint:04d}_rank*.csv*"))
    if len(paths) != run["ranks"] or {p.name.removesuffix(".gz") for p in paths} != expected:
        raise ValueError(f"Wrong rank snapshot file set: {directory}, checkpoint {checkpoint}")
    archived = {Path(record["path"]).name: record for record in run["snapshots"]}
    frames = []
    for path in paths:
        digest = sha256(path)
        if path.suffix == ".gz":
            record = archived.get(path.name)
            if not record or digest != record["sha256"]:
                raise ValueError(f"Compressed snapshot hash mismatch: {path}")
            with gzip.open(path, "rb") as stream:
                recovered_hash = hashlib.sha256(stream.read()).hexdigest()
            if recovered_hash != record["csv_sha256"]:
                raise ValueError(f"Uncompressed snapshot hash mismatch: {path}")
        else:
            record = archived.get(path.name + ".gz")
            if not record or digest != record["csv_sha256"]:
                raise ValueError(f"Plain snapshot has no matching archived content hash: {path}")
        input_hashes[str(path.resolve())] = digest
        frame = pd.read_csv(path, dtype={"id": np.uint64}, float_precision="round_trip")
        if list(frame.columns) != Columns:
            raise ValueError(f"Unexpected snapshot columns: {path}")
        frames.append(frame)
    nonempty = [frame for frame in frames if len(frame)]
    if not nonempty:
        raise ValueError("Empty particle snapshot")
    frame = pd.concat(nonempty, ignore_index=True).sort_values("id").reset_index(drop=True)
    if (len(frame) != count or not np.array_equal(frame.id, np.arange(count))
            or not np.isfinite(frame[Columns].to_numpy(dtype=float)).all()):
        raise ValueError("Snapshot must contain complete aligned IDs and finite phase space")
    return frame


def extract_data(report, input_hashes):
    failures = campaign_status(report)
    parameters = report["parameters"]
    if parameters["checkpoints"] != 8:
        raise ValueError("This fixed summary requires eight checkpoints")
    failed_names = {check["name"] for check in failures}
    data = {"schema": "ippl-fastpm-evolution-plot-v1", "parameters": parameters,
            "campaign_passed": report["passed"], "failed_checks": failures,
            "limitations": report["limitations"], "phase_portraits": [], "spectral_series": [],
            "axis_policy": "Linear y axes; exact zeros retained; undefined normalized residuals remain null",
            "resolved_power_definition": "sum |delta_hat|^2 over recorded unique +/- mode pairs; no window or shot-noise correction",
            "complex_residual_definition": "norm(delta_IPPL-delta_FastPM)/(P_IPPL*P_FastPM)^(1/4)",
            "scope": "Synthetic fixtures; no analytic post-shell-crossing truth; not a continuum or exascale qualification"}
    comparisons = {(row["case"], row["checkpoint"]): row for row in report["comparisons"]}
    for fixture in Fixtures:
        pair_count = len(report["fixtures"][fixture]["resolved_modes"])
        for mesh in Meshes:
            runs = {code: selected_run(report, fixture, mesh, code) for code in Codes}
            coefficients = {code: {row["checkpoint"]: row for row in run["density_modes"]}
                            for code, run in runs.items()}
            series = {"fixture": fixture, "mesh": mesh, "steps": 256, "ranks": 1,
                      "mode_pairs": pair_count,
                      "complex_budget": report["limits"]["resolved_complex_relative"][str(mesh)], "points": []}
            for checkpoint in range(9):
                mode_rows = [coefficients[code][checkpoint] for code in Codes]
                a = mode_rows[0]["a"]
                if not np.isclose(a, mode_rows[1]["a"], rtol=2e-13, atol=0):
                    raise ValueError("Cross-code mode checkpoints do not align")
                vectors = [np.asarray(row["real"]) + 1j*np.asarray(row["imag"]) for row in mode_rows]
                if any(len(vector) != pair_count for vector in vectors):
                    raise ValueError("Wrong number of resolved Fourier coefficients")
                metrics = direct_metrics(*vectors)
                case = f"{fixture}_m{mesh}_t256_r1"
                recorded = comparisons[(case, checkpoint)]["resolved_density"]
                for key, original in (("power_ippl", "power_left"), ("power_fastpm", "power_right"),
                                      ("complex_relative", "complex_relative")):
                    if metrics[key] is None or recorded[original] is None:
                        if metrics[key] != recorded[original]:
                            raise ValueError("Undefined mode normalization disagrees with report")
                    elif not np.isclose(metrics[key], recorded[original], rtol=2e-12, atol=1e-28):
                        raise ValueError("Recomputed mode metric disagrees with report")
                series["points"].append({"checkpoint": checkpoint, "a": a, **metrics,
                    "power_check_failed": f"{case}/{checkpoint}/cross/resolved_power" in failed_names,
                    "complex_check_failed": f"{case}/{checkpoint}/cross/resolved_complex" in failed_names})
            data["spectral_series"].append(series)

    box, n = parameters["box_size"], parameters["particle_grid"]
    modes = report["fixtures"]["pancake"]["modes"]
    if len(modes) != 1 or modes[0][0] != [1, 0, 0]:
        raise ValueError("Phase portrait recentering requires the recorded single x mode")
    center = (-modes[0][2] * box / (2*np.pi)) % box
    data["phase_center"] = center
    data["phase_slice"] = f"Fixed IDs 0..{n-1}: one transverse Lagrangian row, x-fast ID order"
    for checkpoint in (0, 4, 8):
        panel = {"checkpoint": checkpoint, "codes": {}}
        for code in Codes:
            run = selected_run(report, "pancake", 32, code)
            frame = snapshot(run, checkpoint, n**3, input_hashes)
            row = frame.iloc[:n]
            if not np.array_equal(row.id, np.arange(n)):
                raise ValueError("Phase portrait particle IDs do not align")
            centered_x = row.x.to_numpy() - center
            centered_x -= box * np.rint(centered_x / box)
            observation = next(item for item in run["observations"] if item["checkpoint"] == checkpoint)
            panel["a"] = observation["a"]
            panel["codes"][code] = {"id": row.id.tolist(), "x_centered": centered_x.tolist(),
                "px": row.px.tolist(), "minimum_sampled_jacobian": observation["minimum_sampled_jacobian"],
                "transverse_row_displacement_spread": observation["transverse_row_displacement_spread"]}
        data["phase_portraits"].append(panel)
    return data


def status_text(data):
    if data["failed_checks"]:
        names = {check["name"] for check in data["failed_checks"]}
        if names == {"coupled3d/ippl/8/time/finest_momentum", "coupled3d/fastpm/8/time/finest_momentum"}:
            checks = {check["name"].split("/")[1]: check for check in data["failed_checks"]}
            return ("Original campaign: 128→256-step 3D momentum differences "
                    f"{checks['ippl']['value']:.4g} / {checks['fastpm']['value']:.4g} "
                    f"> {checks['ippl']['limit']:.4g} (2 failed gates).")
        return f"Campaign: {len(data['failed_checks'])} failed checks retained. No overall qualification claimed."
    return "Recorded campaign checks passed within the stated synthetic, local-CPU scope."


def format_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=.18, linewidth=.6)
    axis.set_axisbelow(True)
    axis.tick_params(direction="out", length=3)


def phase_figure(data):
    box, n = data["parameters"]["box_size"], data["parameters"]["particle_grid"]
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 4.3))
    for axis, panel in zip(axes, data["phase_portraits"]):
        for code in Codes:
            values = panel["codes"][code]
            x, p = np.asarray(values["x_centered"]), np.asarray(values["px"])
            # Join only adjacent Lagrangian IDs; never bridge a periodic edge.
            breaks = np.flatnonzero(np.abs(np.diff(x)) > box/2) + 1
            for indices in np.split(np.arange(len(x)), breaks):
                axis.plot(x[indices], p[indices], color=CodeColors[code],
                          linewidth=1.35, linestyle="-" if code == "ippl" else "--", alpha=.8)
            axis.plot(x, p, linestyle="none", marker="o", markersize=3.3 if code == "ippl" else 5.0,
                      markerfacecolor=CodeColors[code] if code == "ippl" else "none",
                      markeredgewidth=.9, color=CodeColors[code], label=code.upper() if code == "ippl" else "FastPM (plain PM)")
        jacobian = panel["codes"]["ippl"]["minimum_sampled_jacobian"]
        axis.set_title(f"Checkpoint {panel['checkpoint']} · a = {panel['a']:.4f}\nSampled min J = {jacobian:.3f}", fontsize=10)
        axis.set_xlim(-box/2, box/2)
        axis.set_xlabel(r"Periodic $x-x_c$ [Mpc/$h$]")
        axis.set_ylabel(r"Canonical $p_x$ [Mpc/$h$]")
        format_axis(axis)
    axes[0].legend(frameon=False, fontsize=8.5, loc="best")
    figure.suptitle(rf"Synthetic planar collapse · $N_p={n}^3$, $N_m=32$, 256 steps, 1 MPI rank", fontsize=13, y=.98)
    figure.text(.07, .085, rf"$p=a^2\,dx/d(H_0t)$. Same {n} particle IDs in every panel; lines follow IDs, not an interpolated distribution.", fontsize=9)
    figure.text(.07, .043, "Actual particle states; no analytic post-shell-crossing reference. " + status_text(data),
                fontsize=8.5, color="#9C2F23" if data["failed_checks"] else "#333333")
    figure.subplots_adjust(left=.075, right=.985, top=.77, bottom=.25, wspace=.34)
    return figure


def spectra_figure(data):
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 8.1), sharex=True)
    for row_index, fixture in enumerate(Fixtures):
        power_axis, residual_axis = axes[row_index]
        series_group = [series for series in data["spectral_series"] if series["fixture"] == fixture]
        finite_residuals = [point["complex_relative"] for series in series_group for point in series["points"]
                            if point["complex_relative"] is not None]
        maximum_residual = max(finite_residuals, default=0)
        residual_upper = 1.15 * maximum_residual if maximum_residual > 0 else 1
        offscale_budgets = []
        for series in series_group:
            color = MeshColors[series["mesh"]]
            points = pd.DataFrame(series["points"])
            a = points.a.to_numpy()
            for code in Codes:
                power_axis.plot(a, points["power_"+code], color=color, linewidth=1.7,
                                linestyle="-" if code == "ippl" else "--",
                                marker="o" if code == "ippl" else "s", markersize=3.2,
                                markerfacecolor=color if code == "ippl" else "white", markeredgewidth=.8)
            residual = np.array([np.nan if point["complex_relative"] is None else point["complex_relative"]
                                 for point in series["points"]])
            residual_axis.plot(a, residual, color=color, marker="o", markersize=3.5, linewidth=1.7)
            if series["complex_budget"] <= residual_upper:
                residual_axis.axhline(series["complex_budget"], color=color, linestyle=":", linewidth=1, alpha=.75)
            else:
                offscale_budgets.append(f"NM{series['mesh']}: {100*series['complex_budget']:g}%")
            failed_power = points.power_check_failed.to_numpy()
            failed_complex = points.complex_check_failed.to_numpy()
            power_axis.plot(a[failed_power], points.power_ippl.to_numpy()[failed_power], "x", color="#CC3311", markersize=8, markeredgewidth=1.5)
            residual_axis.plot(a[failed_complex], residual[failed_complex], "x", color="#CC3311", markersize=8, markeredgewidth=1.5)
            if np.isnan(residual).any():
                residual_axis.text(.03, .93-.075*Meshes.index(series["mesh"]),
                                   f"NM{series['mesh']}: {np.isnan(residual).sum()} undefined normalization(s)",
                                   transform=residual_axis.transAxes, va="top", fontsize=8, color=color)
        fixture_label = "Planar pancake" if fixture == "pancake" else "Coupled 3D modes"
        power_axis.set_title(f"{fixture_label} · {series_group[0]['mode_pairs']} unique mode pairs", fontsize=10.5)
        residual_title = f"{fixture_label} · cross-code residual"
        if offscale_budgets:
            residual_title += "\nBudgets above range: " + "; ".join(offscale_budgets)
        residual_axis.set_title(residual_title, fontsize=9.5, pad=10)
        power_axis.set_ylabel(r"Resolved power sum $\sum |\hat\delta|^2$")
        residual_axis.set_ylabel(r"$\|\hat\delta_I-\hat\delta_F\|_2/(P_I P_F)^{1/4}$")
        power_axis.set_ylim(bottom=0)
        # Linear scales include actual zeros; no replacement by a numerical floor.
        residual_axis.set_ylim(-.035 * residual_upper, residual_upper)
        for axis in (power_axis, residual_axis):
            axis.set_xscale("log")
            axis.set_xticks([.02, .05, .1, .2])
            axis.xaxis.set_major_formatter(ScalarFormatter())
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
            format_axis(axis)
    for axis in axes[-1]:
        axis.set_xlabel("Scale factor a")
    handles = [Line2D([], [], color=MeshColors[mesh], linewidth=2, label=rf"$N_m={mesh}$") for mesh in Meshes]
    handles += [Line2D([], [], color="#333333", linestyle="-", label="IPPL power"),
                Line2D([], [], color="#333333", linestyle="--", label="FastPM power"),
                Line2D([], [], color="#CC3311", marker="x", linestyle="none", label="Failed shown check")]
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(.5, .087))
    figure.suptitle("Direct particle-density modes · fixed particles, 256 steps, 1 MPI rank", fontsize=13, y=.98)
    figure.text(.09, .064, "Common band: 0 < |n| ≤ 4 (planar nₓ=1…4), unique ± pairs. No window/shot-noise correction. Linear y axes retain zeros.", fontsize=8.2)
    figure.text(.09, .037, "Synthetic ICs; no analytic post-crossing truth. " + status_text(data),
                fontsize=8.2, color="#9C2F23" if data["failed_checks"] else "#333333")
    figure.subplots_adjust(left=.105, right=.98, top=.84, bottom=.22, hspace=.48, wspace=.30)
    return figure


def run(args):
    source = args.results.resolve()
    initial_hash = sha256(source)
    report = json.loads(source.read_text())
    input_hashes = {str(source): initial_hash, str(Path(__file__).resolve()): sha256(__file__)}
    data = extract_data(report, input_hashes)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Figure output directory must be new or empty")
    data_path = output / "plotted-data.json"
    data_path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    outputs = [data_path]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 10, "axes.titlesize": 11,
                         "svg.hashsalt": "ippl-evolution-summary-v1", "savefig.facecolor": "white"}):
        for name, factory in (("pancake-phase-space", phase_figure), ("resolved-density-evolution", spectra_figure)):
            figure = factory(data)
            for extension in ("png", "svg"):
                path = output / f"{name}.{extension}"
                metadata = {"Software": "IPPL plot_evolution.py"} if extension == "png" else {"Date": None, "Creator": "IPPL plot_evolution.py"}
                figure.savefig(path, dpi=args.dpi, metadata=metadata)
                outputs.append(path)
            plt.close(figure)
    if any(sha256(path) != digest for path, digest in input_hashes.items()):
        raise RuntimeError("Plot input changed during rendering")
    manifest = {"schema": "ippl-evolution-figures-v1", "inputs_sha256": input_hashes,
                "outputs_sha256": {str(path): sha256(path) for path in outputs},
                "dpi": args.dpi, "numpy": np.__version__, "pandas": pd.__version__,
                "matplotlib": matplotlib.__version__, "campaign_passed": data["campaign_passed"],
                "failed_checks_count": len(data["failed_checks"]), "simulations_run": False}
    (output / "sha256-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Two figures rendered; {len(data['failed_checks'])} failed campaign checks retained: {output}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()
    if args.dpi < 72 or args.dpi > 600:
        parser.error("--dpi must be between 72 and 600")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
