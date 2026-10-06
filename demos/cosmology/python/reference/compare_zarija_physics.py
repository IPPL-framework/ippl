#!/usr/bin/env python3
## @file compare_zarija_physics.py
# @brief Build and run public-API probes against unmodified Zarija sources.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
"""Build and run public-API probes against unmodified Zarija sources.

Requires a configured IPPL build for Kokkos include paths and MPI compiler.
All compilation and reference side effects remain below --output-dir.
The supplied table is tested both as-is and in a separate input-only variant
whose CDM and baryon columns are divided by their weighted first-row value.
That variant removes the legacy first-row normalization defect without
changing any original source or the physical normalized transfer function.
A failed gated as-is comparison remains a failure in results.json; declared
DC/first-interval differences are reported separately rather than gated.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


## @brief Compute the source/artifact checksum used by the reference qualification.
# @see cosmology_tools
#
# @param path Input/output filesystem path; freshness and hash requirements are defined by this routine.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


## @brief Read and verify cache.
# @see cosmology_tools
#
# @param path Input/output filesystem path; freshness and hash requirements are defined by this routine.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def read_cache(path):
    values = {}
    for line in path.read_text().splitlines():
        if not line.startswith(("#", "//")) and ":" in line and "=" in line:
            name, value = line.split("=", 1)
            values[name.split(":", 1)[0]] = value
    return values


## @brief Parse the documented command-line interface and execute its selected workflow, retaining nonzero failures.
# @see cosmology_tools
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--ippl-build", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        help="New results directory; default: unique zarija-physics-* in current directory")
    parser.add_argument("--table", type=Path)
    parser.add_argument("--np", type=int, default=32)
    parser.add_argument("--box-size", type=float, default=168.75)
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--omega-m", type=float, default=0.31)
    parser.add_argument("--omega-bar", type=float, default=0.0487)
    parser.add_argument("--hubble", type=float, default=0.675)
    parser.add_argument("--sigma8", type=float, default=0.82)
    parser.add_argument("--ns", type=float, default=0.965)
    arguments = parser.parse_args()
    reference = arguments.reference_source.resolve()
    build = arguments.ippl_build.resolve()
    output = (arguments.output_dir.resolve() if arguments.output_dir is not None
              else Path(tempfile.mkdtemp(prefix="zarija-physics-", dir=Path.cwd())))
    output.mkdir(parents=True, exist_ok=True)
    resultPath = output / "results.json"
    if resultPath.exists():
        parser.error("Use a new --output-dir to preserve earlier comparison results")
    result = {"passed": False, "stage": "build", "cases": []}
    resultPath.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Comparison results: {resultPath}", flush=True)
    table = (arguments.table or reference / "cmb.tf").resolve()
    cache = read_cache(build / "CMakeCache.txt")
    sourceDirectory = Path(__file__).resolve().parents[2] / "reference"
    referenceFiles = [reference / name for name in (
        "Cosmology.cpp", "Cosmology.h", "MT_Random.cpp", "MT_Random.h",
        "DataBase.h", "InputParser.h", "TypesAndDefs.h")]
    inputHashes = {str(path): checksum(path) for path in referenceFiles + [table]}
    executable = output / "CompareZarijaPhysics"
    command = [
        cache["MPI_CXX_COMPILER"], "-std=c++20", "-O2", "-DDOUBLE_REAL", "-DUSENAMESPACE",
        "-I" + str(reference),
        "-I" + str(Path(cache["Kokkos_SOURCE_DIR"]) / "core/src"),
        "-I" + cache["Kokkos_BINARY_DIR"],
        str(sourceDirectory / "CompareZarijaPhysics.cpp"),
        str(reference / "Cosmology.cpp"), str(reference / "MT_Random.cpp"),
        "-o", str(executable),
    ]
    environment = os.environ.copy()
    environment["OMPI_CXX"] = cache["CMAKE_CXX_COMPILER"]
    print("Building public-API probe against original sources", flush=True)
    result["compile_command"] = command
    result["source_sha256_before"] = inputHashes
    try:
        with (output / "build.log").open("w") as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
        result["source_sha256_after"] = {str(path): checksum(path) for path in referenceFiles + [table]}
        resultPath.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Probe build failed; see {output / 'build.log'}", file=sys.stderr)
        return 2

    parameters = {
        "np": arguments.np, "nt": 1, "box_size": arguments.box_size, "seed": arguments.seed,
        "z_in": 49.0, "z_fi": 24.0, "hubble": arguments.hubble,
        "Omega_m": arguments.omega_m, "Omega_bar": arguments.omega_bar,
        "Sigma_8": arguments.sigma8, "n_s": arguments.ns, "ic_mode": "gaussian",
    }
    rows = [[float(value) for value in line.split()] for line in table.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    firstNormalization = (rows[0][1] * (arguments.omega_m-arguments.omega_bar)
                          + rows[0][2] * arguments.omega_bar) / arguments.omega_m
    normalizedTable = output / "normalized-input-table.tf"
    with normalizedTable.open("w") as stream:
        for row in rows:
            normalizedRow = row.copy()
            normalizedRow[1] /= firstNormalization
            normalizedRow[2] /= firstNormalization
            stream.write(" ".join(f"{value:.17g}" for value in normalizedRow) + "\n")
    result = {
        "passed": False, "stage": "comparison",
        "parameters": parameters, "compile_command": command,
        "executable_sha256": checksum(executable),
        "compiler_environment": {"OMPI_CXX": environment["OMPI_CXX"]},
        "source_sha256_before": inputHashes,
        "ippl_header_sha256": {str(path): checksum(path) for path in (
            sourceDirectory.parent / "CosmologyConfig.h",
            sourceDirectory.parent / "CosmologyPhysics.h")},
        "probe_sha256": {str(path): checksum(path) for path in (
            sourceDirectory / "CompareZarijaPhysics.cpp", Path(__file__).resolve())},
        "normalized_table_sha256": checksum(normalizedTable),
        "first_table_weighted_normalization": firstNormalization,
        "tolerance_basis": {
            "transfer_relative": "2e-12, identical analytic function/interpolation away from first row",
            "power_relative": "5e-4, five times original midpoint integrator EPS=1e-4",
            "growth_relative": "2e-5, original RK EPS=1e-6 plus decaying-mode contamination",
            "finite_start_growth_relative_z200": (2.0/3.0) * (201.0/100001.0)**2.5,
        },
        "cases": [],
    }
    for label, flag, transferPath in (
        ("bbks", 4, table), ("raw-table", 0, table), ("normalized-table", 0, normalizedTable),
    ):
        caseDirectory = output / label
        caseDirectory.mkdir()
        caseParameters = {**parameters, "TFFlag": flag, "transfer_file": str(transferPath)}
        inputPath = caseDirectory / "input.par"
        inputPath.write_text("".join(f"{key}={value}\n" for key, value in caseParameters.items()))
        csvPath = caseDirectory / "comparison.csv"
        probeCommand = [str(executable), str(inputPath), str(csvPath)]
        start = time.monotonic()
        returnCode = None
        caseError = None
        try:
            with (caseDirectory / "run.log").open("w") as log:
                completed = subprocess.run(probeCommand, cwd=caseDirectory, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=300)
            returnCode = completed.returncode
        except (OSError, subprocess.SubprocessError) as error:
            caseError = str(error)
        comparisons = []
        try:
            if csvPath.exists():
                with csvPath.open() as stream:
                    comparisons = list(csv.DictReader(stream))
            maximumK2 = 3 * (arguments.np // 2)**2
            expectedKeys = {
                (quantity, -1, 0.0) for quantity in
                ("sigma8_raw", "power_normalization", "sigma8_at_ippl_normalization")
            }
            expectedKeys.update(
                (quantity, k2, 0.0) for quantity in ("transfer_grid", "power_grid")
                for k2 in range(1, maximumK2+1)
            )
            expectedKeys.update(
                (quantity, -1, z) for quantity in ("growth_D", "growth_Ddot", "growth_f")
                for z in (0.0, 9.0, 49.0, 200.0)
            )
            if flag == 0:
                expectedKeys.add(("transfer_second_table_row", -1, 0.0))
            actualKeys = [
                (row["quantity"], int(row["k2"]),
                 float(row["argument"]) if row["quantity"].startswith("growth_") else 0.0)
                for row in comparisons if row["gated"] == "1"
            ]
            comparisonsComplete = (set(actualKeys) == expectedKeys
                                   and len(actualKeys) == len(expectedKeys))
        except (OSError, ValueError, KeyError, csv.Error) as error:
            caseError = str(error)
            comparisons = []
            comparisonsComplete = False
        failures = [row for row in comparisons if row["gated"] == "1" and row["passed"] == "0"]
        firstInterval = next((row for row in comparisons
                              if row["quantity"] == "first_interval_variance_estimate"), None)
        rawSigma = next((row for row in comparisons if row["quantity"] == "sigma8_raw"), None)
        firstIntervalDiagnosis = None
        if firstInterval and rawSigma:
            firstIntervalDiagnosis = {
                "original_first_interval_variance": float(firstInterval["zarija"]),
                "ippl_first_interval_variance_approximation": float(firstInterval["ippl"]),
                "original_interval_fraction_of_reported_raw_variance": (
                    float(firstInterval["zarija"]) / float(rawSigma["zarija"])**2),
                "top_hat_approximation_relative_bound": (8*float(firstInterval["argument"]))**2 / 5,
                "note": "A large first-interval fraction absent from Sigma_r indicates that the legacy quadrature does not resolve its own table discontinuity.",
            }
        result["cases"].append({
            "name": label, "parameters": caseParameters, "command": probeCommand,
            "return_code": returnCode, "elapsed_seconds": time.monotonic()-start,
            "error": caseError, "comparisons_complete": comparisonsComplete,
            "comparison_rows": len(comparisons),
            "gated_comparisons": sum(row["gated"] == "1" for row in comparisons),
            "reported_not_gated": sum(row["gated"] == "0" for row in comparisons),
            "comparison_csv": str(csvPath), "failed_gates": failures,
            "first_interval_diagnosis": firstIntervalDiagnosis,
            "maximum_relative_error_by_quantity": {
                quantity: max(float(row["relative_error"]) for row in comparisons
                              if row["quantity"] == quantity)
                for quantity in sorted({row["quantity"] for row in comparisons})
            },
        })
        print(f"{label}: {len(comparisons)} comparisons, {len(failures)} failed gates, complete={comparisonsComplete}, "
              f"return code {returnCode}", flush=True)
        resultPath.write_text(json.dumps(result, indent=2) + "\n")
    result["source_sha256_after"] = {str(path): checksum(path) for path in referenceFiles + [table]}
    result["reference_unchanged"] = result["source_sha256_after"] == inputHashes
    result["passed"] = (result["reference_unchanged"]
                        and len(result["cases"]) == 3
                        and all(case["return_code"] == 0 and case["comparisons_complete"]
                                and not case["failed_gates"] and case["error"] is None
                                for case in result["cases"]))
    result["stage"] = "complete"
    resultPath.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Comparison results: {resultPath}", flush=True)
    return 0 if result["passed"] else 1


## @cond CLI_DISPATCH
if __name__ == "__main__":
    sys.exit(main())
## @endcond
