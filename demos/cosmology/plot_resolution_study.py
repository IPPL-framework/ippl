#!/usr/bin/env python3
"""Plot recorded resolution-study evidence; never open snapshots or run models.

Only a fully analyzed stage is plotted, even when another stage is incomplete.
Failures and qualification prefixes are carried unchanged into the plot data.
Direct particle Fourier sums cover 0<|n|<=4. Higher Gaussian bands are explicitly
FFT characterization, not additional qualified measurements or truth errors.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


Codes = ("ippl", "fastpm")
Colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#666666")
Edges = (.5, 1.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5)
ShellLabels = ("0<|n|<1.5", "1.5≤|n|<2.5", "2.5≤|n|≤4")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def modes_for(fixture, cutoff=4):
    if fixture == "pancake":
        return np.asarray([[n, 0, 0] for n in range(1, 5)])
    if fixture not in ("coupled3d", "gaussian"):
        raise ValueError("Unknown fixture mode convention")
    return np.asarray([m for m in itertools.product(range(-cutoff, cutoff+1), repeat=3)
                       if 0 < sum(v*v for v in m) <= cutoff**2
                       and next(v for v in m if v) > 0])


def decode(record, count):
    real, imag = np.asarray(record["real"]), np.asarray(record["imag"])
    if real.shape != (count,) or imag.shape != (count,):
        raise ValueError("Wrong recorded Fourier vector length")
    result = real + 1j*imag
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite recorded Fourier coefficient")
    return result


def metrics(left, right):
    power_left, power_right = float(np.vdot(left, left).real), float(np.vdot(right, right).real)
    absolute = float(np.linalg.norm(left-right))
    defined = min(power_left, power_right) > 1e-28
    norm = np.sqrt(power_left*power_right)
    return {"power_left": power_left, "power_right": power_right,
            "power_ratio": power_left/power_right if defined else None,
            "complex_relative": absolute/np.sqrt(norm) if defined else None,
            "correlation": float(np.vdot(right, left).real/norm) if defined else None,
            "absolute_difference": absolute, "normalization_defined": defined}


def _case_name(case):
    shift = "shift" if case["shifted"] else "base"
    return (f"{case['fixture']}_p{case['particles']}_m{case['mesh']}_t{case['steps']}"
            f"_r{case['ranks']}_z{case['redshift']}_{shift}")


def _required_evidence(planned, stage):
    """Exact comparison/time coverage for the fixed, predeclared stage matrix."""
    comparisons, temporal = set(), set()
    def case(fixture, particles, mesh, steps, **changes):
        result = dict(stage=stage, fixture=fixture, particles=particles, mesh=mesh,
                      steps=steps, ranks=1, redshift=49, shifted=False)
        result.update(changes)
        return result
    def add(category, left, lcode, right, rcode, checkpoints=range(9)):
        comparisons.update((category, _case_name(left)+"_"+lcode, _case_name(right)+"_"+rcode, point)
                           for point in checkpoints)
    for item in planned:
        add("cross", item, "ippl", item, "fastpm")
        add("cross_phase", item, "ippl", item, "fastpm")
        if item["ranks"] > 1:
            for code in Codes:
                add("rank", item, code, dict(item, ranks=1), code)
    fixtures = ("pancake", "coupled3d") if stage == "spatial" else ("gaussian",)
    steps = 1024 if stage == "spatial" else 2048
    for fixture in fixtures:
        for code in Codes:
            for fixed in (32, 64):
                add("particle_resolution", case(fixture, 32, fixed, steps), code,
                    case(fixture, 64, fixed, steps), code)
                add("mesh_resolution", case(fixture, fixed, 32, steps), code,
                    case(fixture, fixed, 64, steps), code)
                if stage == "spatial":
                    shifted, base = case(fixture, 64, fixed, steps, shifted=True), case(fixture, 64, fixed, steps)
                    add("translation", shifted, code, base, code)
                    add("translation_phase", shifted, code, base, code)
            for checkpoint in (4, 8):
                temporal.add((fixture, code, 49, checkpoint, (steps//2, steps, steps*2)))
            if stage == "gaussian":
                add("start_redshift", case(fixture, 64, 64, 4096), code,
                    case(fixture, 64, 64, 4096, redshift=99), code, checkpoints=(8,))
                for checkpoint in (4, 8):
                    temporal.add((fixture, code, 99, checkpoint, (2048, 4096)))
    return comparisons, temporal


def complete_stages(report):
    """Trust no completion flag without the planned runs and derived evidence."""
    if report.get("schema") != "ippl-resolution-study-v1" or report["configuration"].get("smoke"):
        raise ValueError("Require a non-smoke ippl-resolution-study-v1 report")
    if (report["parameters"]["checkpoints"] != 8 or report["parameters"]["qualified_max_mode"] != 4
            or tuple(report["budgets"]["shell_edges"]) != Edges):
        raise ValueError("Unsupported checkpoint or direct-mode/shell convention")
    failures = [row for row in report["checks"] if not row["passed"]]
    if failures != report["failed_checks"]:
        raise ValueError("Recorded failed-check list disagrees with checks")
    if report.get("complete") and bool(report["passed"]) != (not failures):
        raise ValueError("Completed campaign status disagrees with checks")
    runs = {run["name"]: run for run in report["runs"]}
    if len(runs) != len(report["runs"]):
        raise ValueError("Duplicate recorded run")
    complete, omitted = [], {}
    for stage in ("spatial", "gaussian"):
        planned = [case for case in report["planned_cases"] if case["stage"] == stage]
        if not planned:
            continue
        expected = {_case_name(case) + "_" + code for case in planned for code in Codes}
        if not expected.issubset(runs):
            omitted[stage] = "Not every planned run is recorded"
            continue
        fixtures = ("pancake", "coupled3d") if stage == "spatial" else ("gaussian",)
        if (stage not in report.get("completed_stages", [])
                or any(f not in report["qualification"] for f in fixtures)
                or any(not any(row["fixture"] == f and row["checkpoint"] == 8
                               for row in report["time_refinement"]) for f in fixtures)):
            omitted[stage] = "Stage comparisons, time evidence or qualification are not complete"
            continue
        expected_comparisons, expected_time = _required_evidence(planned, stage)
        actual_comparisons = [(row["category"], row["left"], row["right"], row["checkpoint"])
                              for row in report["comparisons"] if row["left"] in expected]
        actual_time = [(row["fixture"], row["code"], row["redshift"], row["checkpoint"], tuple(row["steps"]))
                       for row in report["time_refinement"] if row["stage"] == stage]
        if (set(actual_comparisons) != expected_comparisons
                or len(actual_comparisons) != len(expected_comparisons)
                or set(actual_time) != expected_time or len(actual_time) != len(expected_time)):
            omitted[stage] = "Required comparison or temporal evidence is missing, duplicated or unexpected"
            continue
        for name in expected:
            rows = runs[name]["density"]
            if [row["checkpoint"] for row in rows] != list(range(9)):
                raise ValueError(f"Incomplete recorded density epochs: {name}")
            a = np.asarray([row["a"] for row in rows])
            if not np.isfinite(a).all() or not np.all(np.diff(a) > 0):
                raise ValueError("Invalid recorded scale factors")
        complete.append(stage)
    if not complete:
        raise ValueError("No complete analyzed stage is available: " + str(omitted))
    return complete, omitted, runs


def _verified_shell_rows(report, runs, stages):
    """Recompute every displayed shell metric from the recorded direct vectors."""
    rows = []
    eligible = {name for name, run in runs.items() if run["stage"] in stages}
    for comparison in report["comparisons"]:
        if "shells" not in comparison:
            continue
        if comparison["left"] not in eligible:
            continue
        left, right = runs[comparison["left"]], runs[comparison["right"]]
        if left["stage"] not in stages:
            continue
        if left["fixture"] != right["fixture"]:
            raise ValueError("Cannot compare different fixture modes")
        checkpoint = comparison["checkpoint"]
        ldata, rdata = left["density"][checkpoint], right["density"][checkpoint]
        if not np.isclose(ldata["a"], rdata["a"], rtol=2e-13, atol=0):
            raise ValueError("Comparison epochs do not align")
        modes = modes_for(left["fixture"])
        lcoef, rcoef = decode(ldata["direct"], len(modes)), decode(rdata["direct"], len(modes))
        radius = np.linalg.norm(modes, axis=1)
        if [row["shell"] for row in comparison["shells"]] != [0, 1, 2]:
            raise ValueError("Expected the three declared direct-low-mode shells")
        for saved in comparison["shells"]:
            index = saved["shell"]
            mask = (radius >= Edges[index]) & (radius < Edges[index+1])
            actual = metrics(lcoef[mask], rcoef[mask])
            for key, value in actual.items():
                recorded = saved["metrics"][key]
                if value is None or isinstance(value, bool):
                    if value != recorded:
                        raise ValueError("Recorded shell signal definition disagrees")
                elif not np.isclose(value, recorded, rtol=2e-12, atol=1e-28):
                    raise ValueError("Recorded shell metric disagrees with Fourier coefficients")
            rows.append({"stage": left["stage"], "fixture": left["fixture"],
                         "category": comparison["category"], "code": left["code"],
                         "left": left["name"], "right": right["name"],
                         "particles": left["particles"], "mesh": left["mesh"],
                         "checkpoint": checkpoint, "a": ldata["a"], "shell": index,
                         "passed": saved["passed"], "metrics": actual})
    return rows


def _envelope(rows, fixture, category):
    """Worst sampled-epoch/code residual, never a mean or an error vs truth."""
    selected = [row for row in rows if row["fixture"] == fixture and row["category"] == category]
    groups = {}
    for row in selected:
        fixed = row["particles"] if category == "mesh_resolution" else row["mesh"]
        key = fixed if category != "start_redshift" else row["code"]
        groups.setdefault(key, []).append(row)
    curves = []
    for key, samples in sorted(groups.items(), key=lambda item: str(item[0])):
        points = []
        for shell in range(3):
            data = [r for r in samples if r["shell"] == shell]
            defined = [r for r in data if r["metrics"]["normalization_defined"]]
            points.append({"shell": shell, "comparisons": len(data),
                "undefined_comparisons": len(data)-len(defined),
                "power_fraction_max": max((abs(r["metrics"]["power_ratio"]-1) for r in defined), default=None),
                "complex_relative_max": max((r["metrics"]["complex_relative"] for r in defined), default=None),
                "correlation_minimum": min((r["metrics"]["correlation"] for r in defined), default=None),
                "failed_comparisons": sum(not r["passed"] for r in data)})
        curves.append({"category": category, "fixed": key, "points": points})
    return curves


def _time_series(report, fixture):
    series = []
    for row in report["time_refinement"]:
        if row["fixture"] != fixture or row["checkpoint"] != 8:
            continue
        steps = row["steps"]
        differences = row["phase_space_differences"]
        if len(differences) != len(steps)-1 or not np.all(np.diff(steps) > 0):
            raise ValueError("Invalid recorded temporal differences")
        values = [entry["momentum_relative"] for entry in differences]
        if any(value is None or not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("Undefined/nonfinite temporal momentum measurement")
        series.append({"code": row["code"], "redshift": row["redshift"], "steps": steps,
                       "finer_steps": steps[1:], "momentum_relative": values,
                       "orders": row["orders"]})
    return series


def _gaussian_spectra(report, runs):
    """Use exact direct powers below n=4; FFT powers above, without fitting."""
    box = report["parameters"]["box_size"]
    direct_modes = modes_for("gaussian")
    full_modes = modes_for("gaussian", report["parameters"]["gaussian_cutoff"])
    series = []
    for particles, mesh in itertools.product((32, 64), repeat=2):
        pair = {}
        assignments = {}
        for code in Codes:
            matches = [run for run in runs.values() if run["fixture"] == "gaussian"
                and (run["particles"], run["mesh"], run["steps"], run["ranks"], run["redshift"], run["shifted"], run["code"])
                == (particles, mesh, 2048, 1, 49, False, code)]
            if len(matches) != 1:
                raise ValueError("Gaussian spectra require the complete z49/2048-step resolution cross")
            final = matches[0]["density"][8]
            if not np.isclose(final["a"], 1., rtol=2e-13, atol=0):
                raise ValueError("Final Gaussian spectrum is not at a=1")
            pair[code] = (decode(final["direct"], len(direct_modes)), decode(final["fft"], len(full_modes)))
            assignments[code] = final.get("diagnostics", {}).get("assignment", "not recorded")
        points = []
        for index, (lower, upper) in enumerate(zip(Edges[:-1], Edges[1:])):
            mode_set = direct_modes if index < 3 else full_modes
            radius = np.linalg.norm(mode_set, axis=1)
            selected = (radius >= lower) & (radius < upper)
            vectors = [pair[code][0 if index < 3 else 1][selected] for code in Codes]
            if not selected.any():
                continue
            comparison = metrics(*vectors)
            points.append({"shell": index, "k_mean": float(radius[selected].mean()*2*np.pi/box),
                "pairs": int(selected.sum()), "measurement": "direct" if index < 3 else "FFT characterization",
                "power_ippl": box**3*comparison["power_left"]/int(selected.sum()),
                "power_fastpm": box**3*comparison["power_right"]/int(selected.sum()),
                "comparison": comparison})
        series.append({"particles": particles, "mesh": mesh, "steps": 2048, "redshift_initial": 49,
                       "a": 1., "fft_assignment": assignments, "points": points})
    return series


def extract_data(report):
    stages, omitted, runs = complete_stages(report)
    verified = _verified_shell_rows(report, runs, stages)
    data = {"schema": "ippl-resolution-plot-v1", "completed_stages": stages,
        "omitted_stages": omitted, "campaign_complete": report["complete"], "campaign_passed": report["passed"],
        "parameters": report["parameters"], "budgets": report["budgets"],
        "failed_checks": report["failed_checks"], "qualification": report["qualification"],
        "limitations": report["limitations"], "spatial": {}, "gaussian": {},
        "envelope_definition": "Maximum over recorded scale factors and both codes; undefined comparisons remain counted, not silently passed",
        "spectrum_definition": "P(k)=L^3 mean(abs(delta_hat)^2); unique +/- pairs; no shot-noise subtraction or fitted normalization",
        "scope": "Robustness of local finite-resolution calculations; differences are not errors against truth",
        "high_band_policy": "Direct |n|<=4 only can qualify; higher FFT bands are characterization regardless of visual agreement",
        "axis_policy": "Residual plots retain exact zeros on linear axes; undefined points are gaps; no residual floors"}
    for fixture in ("pancake", "coupled3d", "gaussian"):
        stage = "gaussian" if fixture == "gaussian" else "spatial"
        if stage not in stages:
            continue
        for category in ("particle_resolution", "mesh_resolution", "translation"):
            if (stage == "spatial" or category != "translation") and not any(
                    row["fixture"] == fixture and row["category"] == category for row in verified):
                raise ValueError("Missing required spatial/translation comparison evidence")
        controls = {"resolution": _envelope(verified, fixture, "particle_resolution")
                                 + _envelope(verified, fixture, "mesh_resolution"),
                    "time": _time_series(report, fixture)}
        if stage == "spatial":
            controls["translation"] = _envelope(verified, fixture, "translation")
            data["spatial"][fixture] = controls
        else:
            controls["start_redshift"] = _envelope(verified, fixture, "start_redshift")
            if not controls["start_redshift"]:
                raise ValueError("Missing Gaussian starting-redshift comparison")
            controls["spectra"] = _gaussian_spectra(report, runs)
            data["gaussian"] = controls
    return data


def _axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=.18, linewidth=.6)
    axis.set_axisbelow(True)
    axis.tick_params(direction="out", length=3)


def _shell_axis(axis):
    axis.set_xticks(range(3), ShellLabels, fontsize=7.3)
    axis.set_xlim(-.15, 2.15)
    axis.set_xlabel("Exact direct density-mode shell", fontsize=8)
    _axis(axis)


def _values(curve, key):
    return [np.nan if point[key] is None else 100*point[key] for point in curve["points"]]


def _resolution_panel(axis, curves, budget):
    for index, curve in enumerate(curves):
        particle = curve["category"] == "particle_resolution"
        label = (f"NP32→64, NM{curve['fixed']}" if particle else f"NM32→64, NP{curve['fixed']}")
        axis.plot(range(3), _values(curve, "power_fraction_max"), marker="o", markersize=3.5,
                  color=Colors[index], linestyle="-" if particle else "--", label=label)
    axis.axhline(100*budget, color="#555555", lw=.8, linestyle=":", label="Power budget")
    axis.set_ylabel("Maximum |power ratio − 1| [%]")
    axis.set_ylim(bottom=0)
    _shell_axis(axis)
    axis.legend(fontsize=6.8, frameon=False, loc="best")


def _translation_panel(axis, curves, budgets):
    for index, curve in enumerate(curves):
        for key, style, text in (("power_fraction_max", "-", "power"), ("complex_relative_max", "--", "complex")):
            axis.plot(range(3), _values(curve, key), marker="o", markersize=3.2, linestyle=style,
                      color=Colors[index], label=f"NM{curve['fixed']} {text}")
    for key, style in (("power", "-"), ("complex", "--")):
        axis.axhline(100*budgets[key], color="#777777", linestyle=style, lw=.8,
                     label=f"{key.capitalize()} budget")
    axis.set_ylabel("Maximum dephased difference [%]")
    axis.set_ylim(bottom=0)
    _shell_axis(axis)
    axis.legend(fontsize=6.8, frameon=False, loc="best")


def _time_panel(axis, curves, limit, gaussian=False):
    for curve in curves:
        code = curve["code"]
        label = "IPPL" if code == "ippl" else "FastPM plain PM"
        if gaussian:
            label += f", zi={curve['redshift']}"
        axis.plot(curve["finer_steps"], np.asarray(curve["momentum_relative"])*100,
                  marker="s" if gaussian and curve["redshift"] == 99 else "o",
                  markersize=5.2 if code == "fastpm" else 3.8, color=Colors[Codes.index(code)],
                  markerfacecolor="none" if code == "fastpm" else Colors[Codes.index(code)],
                  linestyle="--" if code == "fastpm" else "-", label=label)
    axis.axhline(100*limit, color="#555555", linestyle=":", lw=.9, label="Finest-pair budget")
    ticks = sorted({step for curve in curves for step in curve["finer_steps"]})
    axis.set_xticks(ticks)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Finer step count of each successive pair")
    axis.set_ylabel("Final momentum difference [%]")
    axis.legend(fontsize=7, frameon=False)
    _axis(axis)


def status_text(data, stage):
    failures = [row for row in data["failed_checks"] if row.get("stage") == stage]
    fixtures = ("pancake", "coupled3d") if stage == "spatial" else ("gaussian",)
    prefixes = ", ".join(f"{fixture}: {data['qualification'][fixture]['contiguous_shell_count']}/3 low shells"
                         for fixture in fixtures)
    return f"Recorded {stage} stage: {len(failures)} failed checks retained. Prefixes — {prefixes}."


def spatial_figure(data):
    fig, axes = plt.subplots(2, 3, figsize=(13., 8.))
    for row, fixture in enumerate(("pancake", "coupled3d")):
        controls = data["spatial"][fixture]
        _resolution_panel(axes[row, 0], controls["resolution"], data["budgets"]["spatial"]["power"])
        _translation_panel(axes[row, 1], controls["translation"], data["budgets"]["translation"])
        _time_panel(axes[row, 2], controls["time"], data["budgets"]["inherited"]["time_finest_momentum_relative"])
        for col, title in enumerate(("Independent NP / NM controls", "NP64 translation, phase removed", "Final a=0.2 · timestep differences")):
            axes[row, col].set_title(f"{fixture} · {title}", fontsize=9.2)
    fig.suptitle("Spatial robustness controls · direct particle density modes only", fontsize=14, y=.975)
    fig.text(.055, .035, status_text(data, "spatial"), fontsize=8.5, color="#8B1A1A")
    fig.text(.055, .012, "Translation Δ=(0.37,0.23,0.41)L/64. Control maxima span saved epochs and both codes; exact |n|≤4 only. Differences are not truth errors.", fontsize=8)
    fig.tight_layout(rect=(.015, .065, .99, .945), h_pad=2.5, w_pad=2.)
    return fig


def gaussian_figure(data):
    fig, axes = plt.subplots(2, 3, figsize=(13., 8.2))
    controls = data["gaussian"]
    cutoff = 4*2*np.pi/data["parameters"]["box_size"]
    positive_power_count, nonpositive_power_count = 0, 0
    for index, series in enumerate(controls["spectra"]):
        points = series["points"]
        k = np.asarray([p["k_mean"] for p in points])
        label = f"NP{series['particles']}/NM{series['mesh']}"
        for code, linestyle in (("ippl", "-"), ("fastpm", "--")):
            raw_values = np.asarray([p["power_"+code] for p in points], dtype=float)
            valid = np.isfinite(raw_values) & (raw_values > 0)
            positive_power_count += int(valid.sum())
            nonpositive_power_count += int((~valid).sum())
            # A zero or invalid logarithmic value is a gap, never replaced by
            # a positive floor. The exact recorded value remains in plot_data.
            values = np.where(valid, raw_values, np.nan)
            axes[0, 0].plot(k, values, color=Colors[index], linestyle=linestyle, linewidth=1.1,
                           label=label if code == "ippl" else None)
            axes[0, 0].scatter(k[:3], values[:3], color=Colors[index], s=13)
            axes[0, 0].scatter(k[3:], values[3:], edgecolor=Colors[index], facecolor="none", s=17)
        for axis, key in ((axes[0, 1], "power_ratio"), (axes[0, 2], "complex_relative")):
            values = [np.nan if p["comparison"][key] is None else
                      100*(abs(p["comparison"][key]-1) if key == "power_ratio" else p["comparison"][key]) for p in points]
            axis.plot(k, values, color=Colors[index], lw=1.1, label=label)
            axis.scatter(k[:3], values[:3], color=Colors[index], s=13)
            axis.scatter(k[3:], values[3:], edgecolor=Colors[index], facecolor="none", s=17)
    if positive_power_count:
        axes[0, 0].set_yscale("log")
    else:
        axes[0, 0].text(.5, .5, "No positive shell power\nlog scale undefined", transform=axes[0, 0].transAxes,
                        ha="center", va="center", fontsize=9, color="#8B1A1A")
    if nonpositive_power_count:
        axes[0, 0].text(.98, .04, f"{nonpositive_power_count} nonpositive/undefined powers\nshown as gaps; no floor",
                        transform=axes[0, 0].transAxes, ha="right", va="bottom", fontsize=7, color="#8B1A1A")
    axes[0, 0].set_ylabel(r"$P(k)$ [(Mpc/$h$)$^3$]")
    axes[0, 0].set_title("a=1, zi=49, 2048 steps\nIPPL solid · native plain PM dashed", fontsize=9.3)
    axes[0, 1].set_ylabel("|P(IPPL) / P(FastPM) − 1| [%]")
    axes[0, 1].set_title("Paired-code power difference", fontsize=10)
    axes[0, 2].set_ylabel("Complex density-mode difference [%]")
    correlations = [p["comparison"]["correlation"] for s in controls["spectra"] for p in s["points"]
                    if p["measurement"] == "direct" and p["comparison"]["correlation"] is not None]
    min_r = f"{min(correlations):.7f}" if correlations else "undefined"
    axes[0, 2].set_title(f"Final complex residual · min direct r={min_r}", fontsize=9.3)
    for axis in axes[0]:
        right = axis.get_xlim()[1]
        axis.axvspan(cutoff, right, color="#eeeeee", zorder=-5)
        axis.axvline(cutoff, color="#888888", lw=.7)
        axis.text(.98, .98, "FFT characterization\nbeyond |n|=4", transform=axis.transAxes,
                  ha="right", va="top", fontsize=7.2, color="#555555")
        axis.set_xlabel(r"Shell-mean $k$ [$h$/Mpc]")
        _axis(axis)
    for axis, key in ((axes[0, 1], "power"), (axes[0, 2], "complex")):
        handles = []
        for mesh, style in ((32, ":"), (64, "--")):
            budget = data["budgets"]["cross"][str(mesh)][key]
            handles.append(axis.hlines(100*budget, 0, cutoff, color="#777777", linestyle=style, lw=.8,
                                       label=f"NM{mesh} direct-band budget"))
        axis.set_ylim(bottom=0)
        axis.legend(handles=handles, fontsize=6.6, frameon=False, loc="upper left")
    axes[0, 0].legend(fontsize=7.1, frameon=False, loc="lower left")
    _resolution_panel(axes[1, 0], controls["resolution"], data["budgets"]["spatial"]["power"])
    axes[1, 0].set_title("Spatial controls · epoch/code maxima", fontsize=10)
    for index, curve in enumerate(controls["start_redshift"]):
        code = curve["fixed"]
        code_label = "IPPL" if code == "ippl" else "FastPM plain PM"
        for key, style, label in (("power_fraction_max", "-", "power"), ("complex_relative_max", "--", "complex")):
            axes[1, 1].plot(range(3), _values(curve, key), color=Colors[Codes.index(code)], linestyle=style,
                           marker="o", markersize=3.3, label=f"{code_label} {label}")
    for key, style in (("power", "-"), ("complex", "--")):
        axes[1, 1].axhline(100*data["budgets"]["start_redshift"][key], color="#777777", linestyle=style, lw=.8)
    axes[1, 1].set_ylabel("zi=49 vs zi=99 difference [%]")
    axes[1, 1].set_title("Starting-redshift sensitivity\nNP=NM64, 4096 steps, same a=1", fontsize=9.5)
    axes[1, 1].set_ylim(bottom=0)
    _shell_axis(axes[1, 1])
    axes[1, 1].legend(fontsize=6.8, frameon=False)
    _time_panel(axes[1, 2], controls["time"], data["budgets"]["inherited"]["time_finest_momentum_relative"], gaussian=True)
    axes[1, 2].set_title("Finest-mesh timestep differences", fontsize=10)
    fig.suptitle("Common-phase band-limited Gaussian ΛCDM · one realization, not a precision-cosmology claim", fontsize=12.5, y=.977)
    fig.text(.055, .037, status_text(data, "gaussian"), fontsize=8.5, color="#8B1A1A")
    fig.text(.055, .014, "IC band |n|≤12. Filled: exact direct |n|≤4; open/shaded: FFT characterization only. No shot-noise subtraction. Differences are not truth errors.", fontsize=8)
    fig.tight_layout(rect=(.015, .075, .99, .948), h_pad=2.6, w_pad=2.)
    return fig


def render(report_path, output_dir):
    report_path, output_dir = Path(report_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    script_digest = sha256(__file__)
    raw = report_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    data = extract_data(json.loads(raw))
    data["input_report"] = {"path": str(report_path), "sha256": digest}
    output_dir.mkdir(parents=True, exist_ok=False)
    # Preserve the bytes read once. A running campaign may atomically append
    # another stage while its already-complete stage is being rendered.
    snapshot = output_dir / "input-report.json.gz"
    snapshot.write_bytes(gzip.compress(raw, mtime=0))
    data["input_report"]["snapshot"] = str(snapshot)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5,
                         "svg.fonttype": "none", "svg.hashsalt": "ippl-resolution-study-v1",
                         "savefig.facecolor": "white"})
    paths = [snapshot]
    for stage, plotter in (("spatial", spatial_figure), ("gaussian", gaussian_figure)):
        if stage not in data["completed_stages"]:
            continue
        figure = plotter(data)
        for extension in ("png", "svg"):
            path = output_dir / f"{stage}-controls.{extension}"
            figure.savefig(path, dpi=240, metadata={"Date": None} if extension == "svg" else None)
            paths.append(path)
        plt.close(figure)
    plotted = output_dir / "plot_data.json"
    plotted.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    paths.append(plotted)
    if sha256(__file__) != script_digest:
        raise ValueError("Plot script changed during rendering; output is not a completed artifact")
    manifest = {"schema": "ippl-resolution-plot-manifest-v1", "input_report": data["input_report"],
                "script": {"path": str(Path(__file__).resolve()), "sha256": script_digest},
                "source_report_unchanged_during_plot": sha256(report_path) == digest,
                "input_snapshot_contract": "gzip stores the exact input bytes read once; decompressed SHA256 equals input_report.sha256",
                "completed_stages": data["completed_stages"], "omitted_stages": data["omitted_stages"],
                "campaign_complete": data["campaign_complete"], "campaign_passed": data["campaign_passed"],
                "failed_check_count": len(data["failed_checks"]),
                "software": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
                "outputs": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in paths]}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.report, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
