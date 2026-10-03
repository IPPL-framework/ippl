#!/usr/bin/env python3
"""Opt-in 128/256/512-step follow-up of the completed coupled3d evolution case.

Only NP=NM=32, MPI=1 is extended, using the exact parent CSV and executables.
The parent campaign and its failures remain unchanged. Success here means
successive timestep differences satisfy the original budgets, not errors
against truth, a replacement full-campaign pass, or a new mesh/MPI qualification.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

import validate_evolution as original


Codes = ("ippl", "fastpm")
Steps = (128, 256, 512)
Checkpoints = (4, 8)
SourceNames = ("validate_evolution.py", "validate_linear.py", "CosmologySimulation.h",
               "CosmologyPhysics.h", "CosmologyConfig.h", "tests/CompareCosmologyEvolution.cpp",
               "reference/FastPMEvolution.c", "reference/build_fastpm_evolution.sh")


def canonical(value):
    """Compare JSON objects whose integer dictionary keys were serialized."""
    return json.loads(json.dumps(value, allow_nan=False))


def verify_hashes(hashes):
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Missing provenance hashes")
    for path, expected in hashes.items():
        if not Path(path).is_absolute() or original.sha256(path) != expected:
            raise ValueError(f"Provenance hash mismatch: {path}")


def verify_archive(record):
    path = Path(record["path"])
    verify_hashes({str(path): record["sha256"]})
    with gzip.open(path, "rb") as stream:
        content = stream.read()
    if (hashlib.sha256(content).hexdigest() != record["csv_sha256"]
            or len(content) != record["csv_bytes"]
            or path.stat().st_size != record["compressed_bytes"]):
        raise ValueError(f"Archived snapshot content mismatch: {path}")


def saved_spectrum(run, checkpoint, parameters):
    entries = [entry for entry in run["density_modes"] if entry["checkpoint"] == checkpoint]
    if len(entries) != 1:
        raise ValueError("Missing or duplicate parent Fourier checkpoint")
    entry = entries[0]
    real, imag = np.asarray(entry["real"], dtype=float), np.asarray(entry["imag"], dtype=float)
    expected_a = parameters["a_initial"] * math.exp(checkpoint * math.log(
        parameters["a_final"] / parameters["a_initial"]) / parameters["checkpoints"])
    if (real.shape != (len(original.resolved_modes("coupled3d")),) or imag.shape != real.shape
            or not np.isfinite(real).all() or not np.isfinite(imag).all()
            or not math.isfinite(entry["a"])
            or abs(entry["a"] / expected_a - 1) > original.Limits["schedule_relative"]):
        raise ValueError("Invalid parent Fourier coefficients or epoch")
    return real + 1j * imag


def load_input(path, parameters):
    frame = pd.read_csv(path, dtype={"id": np.uint64}, float_precision="round_trip")
    if list(frame.columns) != original.Columns + ["mass"]:
        raise ValueError("Parent input has incorrect columns")
    frame = frame.sort_values("id").reset_index(drop=True)
    original.validate_snapshot(frame)
    momentum = frame[["px", "py", "pz"]].to_numpy()
    positions = frame[["x", "y", "z"]].to_numpy()
    if (len(frame) != parameters["particle_grid"]**3 or not (frame.mass == 1).all()
            or not ((positions >= 0) & (positions < parameters["box_size"])).all()
            or not np.array_equal(momentum, momentum.astype(np.float32).astype(np.float64))):
        raise ValueError("Parent input violates count, unit-mass, periodic or shared-momentum contract")
    return frame


def inspect_parent(path):
    """Read-only preflight. No fixture generation, output creation, or MPI launch."""
    path = Path(path).resolve()
    parent_sha = original.sha256(path)
    parent = json.loads(path.read_text())
    if (parent.get("schema") != "ippl-fastpm-evolution-v1" or parent.get("complete") is not True
            or parent.get("quick") is not False or parent.get("passed") is not False
            or parent.get("parameters") != original.Parameters
            or parent.get("limits") != canonical(original.Limits)):
        raise ValueError("Parent is not the completed full campaign with unchanged parameters and limits")
    checks = parent["checks"]
    failures = [check for check in checks if check["passed"] is False]
    expected_failures = {f"coupled3d/{code}/8/time/finest_momentum" for code in Codes}
    if (not checks or len({check["name"] for check in checks}) != len(checks)
            or any(type(check["passed"]) is not bool for check in checks)
            or failures != parent["failed_checks"] or len(failures) != 2
            or {check["name"] for check in failures} != expected_failures):
        raise ValueError("Parent failure set differs from the two recorded finest-momentum failures")
    expected_runs = {case.name + "_" + code for case in original.cases(False) for code in Codes}
    if (len(parent["runs"]) != len(expected_runs)
            or {run["name"] for run in parent["runs"]} != expected_runs):
        raise ValueError("Parent full-campaign run coverage is incomplete")
    hashes = parent["hashes_before"]
    if hashes != parent["hashes_after"]:
        raise ValueError("Original campaign recorded changed provenance")
    verify_hashes(hashes)
    source = Path(original.__file__).resolve().parent
    required = {str(source / name) for name in SourceNames}
    fixtures = parent["fixtures"]
    for kind in ("pancake", "coupled3d"):
        fixture = fixtures[kind]
        fixture_path = Path(fixture["path"])
        if (fixture_path.parent != path.parent or hashes.get(str(fixture_path)) != fixture["sha256"]
                or fixture["particle_grid"] != original.Parameters["particle_grid"]):
            raise ValueError("Parent fixture provenance mismatch")
        required.add(str(fixture_path))
    if fixtures["coupled3d"]["resolved_modes"] != original.resolved_modes("coupled3d").tolist():
        raise ValueError("Parent resolved-mode ordering mismatch")
    runs, spectra, artifacts, executables = {}, {}, {}, {}
    for code in Codes:
        for steps in Steps[:-1]:
            case = original.Case("coupled3d", 32, steps)
            run = next(run for run in parent["runs"] if run["name"] == case.name + "_" + code)
            if any(run[key] != value for key, value in {**case.__dict__, "code": code}.items()):
                raise ValueError("Parent run identity mismatch")
            output = Path(run["output"])
            command = run["command"]
            expected_tail = [str(original.Parameters["particle_grid"]), "32",
                             *(str(original.Parameters[key]) for key in
                               ("box_size", "omega_m", "a_initial", "a_final")), str(steps), "8",
                             fixtures["coupled3d"]["path"], str(output)]
            if (len(command) < 11 or command[-10:] != expected_tail
                    or output.parent != path.parent or output.name != run["name"]
                    or run["input_sha256"] != fixtures["coupled3d"]["sha256"]):
                raise ValueError("Parent run input, output or executable arguments mismatch")
            executable = Path(command[-11])
            if code in executables and executables[code] != executable:
                raise ValueError("Parent uses inconsistent executables")
            executables[code] = executable
            required.add(str(executable))
            runs[(case, code)] = run
            for checkpoint in Checkpoints:
                expected_file = output / f"particles_checkpoint{checkpoint:04d}_rank0.csv.gz"
                records = [item for item in run["snapshots"] if item["path"] == str(expected_file)]
                if len(records) != 1:
                    raise ValueError("Missing or duplicate parent snapshot provenance")
                verify_archive(records[0])
                artifacts[str(expected_file)] = records[0]["sha256"]
                original.read_snapshot(str(output), checkpoint, 1, original.Parameters["particle_grid"])
                spectra[(case, code, checkpoint)] = saved_spectrum(run, checkpoint, original.Parameters)
    manifests = [Path(key) for key in hashes if Path(key).name == "build-manifest.txt"]
    if len(manifests) != 1:
        raise ValueError("Parent must retain exactly one native build manifest")
    required.add(str(manifests[0]))
    if set(hashes) != required:
        raise ValueError("Parent source, executable, manifest or input hash coverage mismatch")
    fixture = load_input(fixtures["coupled3d"]["path"], original.Parameters)
    verify_hashes({str(path): parent_sha})
    return {"path": path, "sha256": parent_sha, "report": parent, "runs": runs,
            "spectra": spectra, "artifacts": artifacts, "executables": executables,
            "manifest": manifests[0], "fixture": fixture}


def assess_refinement(code, checkpoint, snapshots, spectra, parameters):
    """The original gates applied to successive 128/256/512 differences."""
    if code not in Codes or checkpoint not in Checkpoints or len(snapshots) != 3 or len(spectra) != 3:
        raise ValueError("Refinement needs the specified code, checkpoint and all three resolutions")
    differences = [original.phase_space_metrics(a, b, parameters["box_size"], 32)
                   for a, b in zip(snapshots[:-1], snapshots[1:])]
    density = [original.density_comparison(a, b) for a, b in zip(spectra[:-1], spectra[1:])]
    label = f"coupled3d/{code}/{checkpoint}/time"
    row = {"name": label, "steps": list(Steps), "phase_space_differences": differences,
           "density_differences": density, "orders": {},
           "interpretation": "Successive timestep differences, not errors against truth"}
    checks = []
    lower, upper = (original.Limits["time_precross_ratio_range"] if checkpoint == 4
                    else (original.Limits["time_postcross_ratio_minimum"], None))
    for quantity in ("position_cells", "momentum_relative", "complex_relative"):
        values = [entry[quantity] for entry in (density if quantity == "complex_relative" else differences)]
        floor = original.Limits["time_precision_" + quantity]
        result = original.refinement(*values, floor, lower, upper)
        row["orders"][quantity] = result
        checks.append({"name": label + "/" + quantity, **result,
                       "coarse_difference": values[0], "fine_difference": values[1], "precision_floor": floor})
    if checkpoint == 8:
        for suffix, quantity in (("position", "position_cells"), ("momentum", "momentum_relative")):
            value, limit = differences[-1][quantity], original.Limits["time_finest_" + quantity]
            checks.append({"name": label + "/finest_" + suffix,
                           "passed": bool(value is not None and np.isfinite(value) and value <= limit),
                           "value": value, "limit": limit})
    return row, checks


class RefinementCampaign(original.Campaign):
    def __init__(self, args, parent):
        for code in Codes:
            requested = getattr(args, code + "_exe")
            expected = parent["executables"][code]
            if requested is not None and requested.resolve() != expected:
                raise ValueError(f"{code} executable must be the exact recorded parent executable")
            setattr(args, code + "_exe", expected)
        if args.fastpm_manifest is not None and args.fastpm_manifest.resolve() != parent["manifest"]:
            raise ValueError("Native manifest must be the exact recorded parent manifest")
        args.fastpm_manifest, args.quick = parent["manifest"], False
        if args.output_dir is None:
            args.output_dir = Path(tempfile.mkdtemp(prefix="evolution-refinement-", dir=args.ippl_exe.parent))
        if args.output_dir.resolve().is_relative_to(parent["path"].parent):
            raise ValueError("Follow-up output must be outside the preserved parent campaign")
        super().__init__(args)
        self.parent = parent
        self.hashes.update(parent["report"]["hashes_before"])
        self.hashes.update(parent["artifacts"])
        self.hashes[str(parent["path"])] = parent["sha256"]
        self.hashes[str(Path(__file__).resolve())] = original.sha256(__file__)
        verify_hashes(self.hashes)
        self.report.update({"schema": "ippl-fastpm-evolution-refinement-v1", "target_steps": list(Steps),
            "parent": {"path": str(parent["path"]), "sha256": parent["sha256"],
                       "complete": parent["report"]["complete"], "passed": parent["report"]["passed"],
                       "failed_checks": deepcopy(parent["report"]["failed_checks"]),
                       "failures_superseded": False},
            "scope": "Only coupled3d NP32 NM32 MPI1 timestep refinement; no new mesh/MPI qualification",
            "qualification": "Pending; parent full campaign remains failed",
            "expected_new_runs": 2, "reused_runs": [], "mesh_refinement": [],
            "accepted_baseline": "Original full-campaign failures are retained, not waived or replaced"})
        self.report["limitations"].extend(["Finest-pair differences are not errors against a truth solution",
                                           "Only two new single-rank runs; no MPI or mesh study is repeated"])
        self.fixtures["coupled3d"] = parent["fixture"]
        self.report["fixtures"]["coupled3d"] = deepcopy(parent["report"]["fixtures"]["coupled3d"])
        self.spectra.update(parent["spectra"])
        for key, run in parent["runs"].items():
            self.outputs[key] = Path(run["output"])
            self.report["reused_runs"].append({field: deepcopy(run[field]) for field in
                ("name", "fixture", "mesh", "steps", "ranks", "code", "output", "input_sha256")})
        self.report["reused_snapshot_hashes"] = parent["artifacts"]
        self.save()

    def run(self):
        print(f"Evidence: {self.root}", flush=True)
        case = original.Case("coupled3d", 32, 512)
        verify_hashes(self.hashes)
        for code in Codes:
            self.run_case(case, code)
        self.compare_codes(case)
        group = [original.Case("coupled3d", 32, steps) for steps in Steps]
        for code in Codes:
            for checkpoint in Checkpoints:
                snapshots = [original.read_snapshot(str(self.outputs[(item, code)]), checkpoint, 1, 32)
                             for item in group]
                spectra = [self.spectra[(item, code, checkpoint)] for item in group]
                row, checks = assess_refinement(code, checkpoint, snapshots, spectra, self.parameters)
                self.report["time_refinement"].append(row)
                self.report["checks"].extend(checks)
        self.report["hashes_after"] = {path: original.sha256(path) for path in self.hashes}
        self.check("parent, reused artifacts, sources, inputs and executables unchanged",
                   self.report["hashes_after"] == self.hashes)
        self.report["complete"] = (len(self.report["runs"]) == 2 and len(self.report["comparisons"]) == 9
                                   and len(self.report["time_refinement"]) == 4)
        self.report["passed"] = bool(self.report["complete"] and self.report["checks"]
                                      and all(check["passed"] for check in self.report["checks"]))
        self.report["qualification"] = ("Targeted 128/256/512 timestep differences satisfy the unchanged budgets"
            if self.report["passed"] else "Targeted refinement did not satisfy all unchanged budgets")
        self.report["qualification"] += "; original full campaign remains failed"
        self.report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        self.save()
        print(f"2 new runs, {len(self.report['checks'])} checks, {len(self.report['failed_checks'])} failed: "
              f"{self.root / 'results.json'}", flush=True)
        print(self.report["qualification"], flush=True)
        return 0 if self.report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True, help="Original full campaign results.json")
    parser.add_argument("--ippl-exe", type=Path, help="Optional; must match hashed parent executable")
    parser.add_argument("--fastpm-exe", type=Path, help="Optional; must match hashed parent executable")
    parser.add_argument("--fastpm-manifest", type=Path, help="Optional; must match hashed parent manifest")
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--numproc-flag", default="-n")
    parser.add_argument("--mpi-arg", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, help="New empty directory outside the parent campaign")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--check-only", action="store_true", help="Read-only provenance/artifact preflight; no simulations")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    campaign = None
    try:
        parent = inspect_parent(args.parent)
        if args.check_only:
            print(f"Verified parent {parent['sha256']}; all {len(parent['report']['hashes_before'])} provenance "
                  f"hashes, {len(parent['artifacts'])} reused archives and exact shared input; no simulations")
            return 0
        campaign = RefinementCampaign(args, parent)
        return campaign.run()
    except Exception as error:
        if campaign is not None:
            campaign.report.update({"complete": False, "passed": False, "execution_error": str(error)})
            campaign.save()
        print(f"Refinement failed: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
