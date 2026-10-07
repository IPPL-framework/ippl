#!/usr/bin/env python3
## @file gpu_mpiexec.py
# @brief Strict, evidence-producing MPI launcher for the single-node Merlin campaign.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
"""Strict, evidence-producing MPI launcher for the single-node Merlin campaign.

Usage: gpu_mpiexec.py --config /absolute/config.json -n R TARGET [untouched args]
The two prefix flags may be reversed; --config=/absolute/config.json also works.
No other MPI options are accepted. Config requires absolute paths for
mpiexec, python, gpu_helper, evidence_dir; gpu_executables and cpu_executables
are lists of absolute paths. sha256 maps every executable/helper to
its digest (additional pinned provenance files, including this script, are OK).
allocation_evidence is the job-preflight CSV path: four headerless rows of
GPU name, UUID, PCI bus ID, MIG mode. It is hashed at launch, not at configure.

GPU_BINDING validates observed rank/PCI distinctness, not GPU model or MIG
mode. Full A100 / MIG-disabled allocation evidence belongs to the job preflight;
explicit MIG tokens are rejected here. A successful launch is NOT a scientific
pass. No process group/session is created: the outer validator's timeout owns
the wrapper, mpiexec, and its children together.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid


## @var GpuNames
# @brief Named GpuNames protocol/schema value; the source initializer records its exact contents.
GpuNames = {"Cosmology", "CompareCosmologyForce", "CompareCosmologyEvolution"}
## @var CpuNames
# @brief Named CpuNames protocol/schema value; the source initializer records its exact contents.
CpuNames = {"FastPMForce", "FastPMEvolution"}
## @var BindingPrefix
# @brief Named BindingPrefix protocol/schema value; the source initializer records its exact contents.
BindingPrefix = b"GPU_BINDING "
## @var BlasEnvironment
# @brief Named BlasEnvironment protocol/schema value; the source initializer records its exact contents.
BlasEnvironment = {"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


## @brief Rejected launch configuration or incomplete binding evidence.
# @see cosmology_tools
class LaunchError(RuntimeError):
    """Rejected launch configuration or incomplete binding evidence."""


## @brief Compute the streaming SHA256 digest of the file bytes.
# @see cosmology_tools
#
# @param path Input/output filesystem path; freshness and hash requirements are defined by this routine.
# @return Lowercase hexadecimal SHA256 of the exact retained bytes.
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


## @brief Consume only launcher prefix options; never parse target arguments.
# @see cosmology_tools
#
# @param arguments Parsed command-line options; see main/--help and the module workflow contract.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def parse_arguments(arguments):
    """Consume only launcher prefix options; never parse target arguments."""
    arguments = list(arguments)
    configPath, ranks, offset = None, None, 0
    while offset < len(arguments):
        word = arguments[offset]
        if word == "--config" or word.startswith("--config="):
            if configPath is not None:
                raise LaunchError("Duplicate --config")
            if word == "--config":
                offset += 1
                if offset == len(arguments):
                    raise LaunchError("Missing --config value")
                configPath = arguments[offset]
            else:
                configPath = word.partition("=")[2]
        elif word == "-n":
            if ranks is not None:
                raise LaunchError("Duplicate -n")
            offset += 1
            if offset == len(arguments) or re.fullmatch(r"[1-4]", arguments[offset]) is None:
                raise LaunchError("Require -n with ranks 1, 2, 3, or 4")
            ranks = int(arguments[offset])
        elif word.startswith("-"):
            raise LaunchError(f"Unsupported launcher option: {word}")
        else:
            break
        offset += 1
    if not configPath or not Path(configPath).is_absolute() or ranks is None or offset == len(arguments):
        raise LaunchError("Require absolute --config, -n R, and target executable")
    if not Path(arguments[offset]).is_absolute():
        raise LaunchError("Target executable must be absolute")
    return Path(configPath).resolve(), ranks, arguments[offset:]


## @brief Evaluate the real path helper in the documented module workflow.
# @see cosmology_tools
#
# @param value Measured or serialized scalar in the declared metric/schema; no normalization is inferred.
# @param file Source/executable/file path whose retained bytes are hashed or validated.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def _real_path(value, *, file=True):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise LaunchError("Configuration paths must be absolute strings")
    path = Path(value)
    if file and not path.is_file():
        raise LaunchError(f"Require existing file: {value}")
    if not file and path.exists() and not path.is_dir():
        raise LaunchError("Evidence path is not a directory")
    return path


## @brief Reject mismatches between retained bytes and their recorded provenance hashes.
# @see cosmology_tools
#
# @param hashes Absolute source/artifact paths mapped to expected SHA256 values; changed bytes invalidate provenance.
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def verify_hashes(hashes):
    for path, expected in hashes.items():
        if sha256(path) != expected:
            raise LaunchError(f"SHA256 mismatch: {path}")


## @brief Load and verify config.
# @see cosmology_tools
#
# @param path Input/output filesystem path; freshness and hash requirements are defined by this routine.
# @param targetArguments Untouched argument vector passed to the selected executable; no extra Kokkos options are invented.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def load_config(path, targetArguments):
    try:
        configBytes = path.read_bytes()
        config = json.loads(configBytes)
        for key in ("mpiexec", "python", "gpu_helper"):
            _real_path(config[key])
        _real_path(config["evidence_dir"], file=False)
        _real_path(config["allocation_evidence"])
        allowed = {}
        for kind, names in (("gpu", GpuNames), ("cpu", CpuNames)):
            entries = config[kind + "_executables"]
            if not isinstance(entries, list) or len(set(entries)) != len(entries):
                raise LaunchError("Executable allowlists must contain unique paths")
            for value in entries:
                executable = _real_path(value)
                identity = str(executable.resolve())
                if executable.name not in names or identity in allowed:
                    raise LaunchError(f"Unsupported or overlapping target allowlist: {value}")
                allowed[identity] = kind
        suppliedHashes = config["sha256"]
        if not isinstance(suppliedHashes, dict) or not suppliedHashes:
            raise LaunchError("Nonempty SHA256 map required")
        hashes = {}
        for value, digest in suppliedHashes.items():
            identity = str(_real_path(value).resolve())
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise LaunchError("SHA256 digests must contain 64 lowercase hexadecimal characters")
            if identity in hashes and hashes[identity] != digest:
                raise LaunchError("Conflicting SHA256 values for path aliases")
            hashes[identity] = digest
        config["sha256"] = hashes
        required = set(allowed) | {str(Path(config[key]).resolve()) for key in ("mpiexec", "python", "gpu_helper")}
        if not required.issubset(hashes):
            raise LaunchError("SHA256 coverage must include launch tools, helper, and every allowed target")
        target = str(Path(targetArguments[0]).resolve(strict=True))
        if target not in allowed:
            raise LaunchError("Target is not allowlisted")
        for executable in (config["mpiexec"], config["python"], target):
            if not os.access(executable, os.X_OK):
                raise LaunchError(f"File is not executable: {executable}")
        verify_hashes(hashes)
        return config, target, allowed[target], hashlib.sha256(configBytes).hexdigest()
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise LaunchError(f"Invalid launcher configuration: {error}") from error


## @brief Require the recorded Slurm CPU cap, even for a one-rank launch.
# @see cosmology_tools
#
# @param environment Explicit subprocess environment; it does not by itself qualify an execution backend.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def allocation(environment):
    """Require the recorded Slurm CPU cap, even for a one-rank launch."""
    job = environment.get("SLURM_JOB_ID", "")
    if re.fullmatch(r"[0-9]+", job) is None:
        raise LaunchError("An active Slurm job allocation is required")
    nodeKeys = [key for key in ("SLURM_JOB_NUM_NODES", "SLURM_NNODES") if key in environment]
    if not nodeKeys or any(environment[key] != "1" for key in nodeKeys):
        raise LaunchError("Require exactly one allocated Slurm node")
    expected = {"SLURM_CPUS_ON_NODE": "4", "SLURM_NTASKS": "4", "SLURM_CPUS_PER_TASK": "1"}
    if any(environment.get(key) != value for key, value in expected.items()):
        raise LaunchError("Require Slurm CPUs_ON_NODE=4, NTASKS=4, CPUS_PER_TASK=1")
    return {key: environment[key] for key in ("SLURM_JOB_ID", *nodeKeys, *expected)}


## @brief Build command.
# @see cosmology_tools
#
# @param config Explicit validated launcher/campaign configuration; machine identity and hashes are checked separately.
# @param ranks Positive MPI rank count; all expected snapshot shards must exist.
# @param kind Fixture selector from the module's declared supported cases.
# @param targetArguments Untouched argument vector passed to the selected executable; no extra Kokkos options are invented.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def build_command(config, ranks, kind, targetArguments):
    command = [config["mpiexec"], "--bind-to", "none", "--map-by", "slot", "-n", str(ranks)]
    if kind == "gpu":
        command += [config["python"], "-B", config["gpu_helper"]]
    return command + list(targetArguments)


## @brief Evaluate the pci identity helper in the documented module workflow.
# @see cosmology_tools
#
# @param value Measured or serialized scalar in the declared metric/schema; no normalization is inferred.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def pci_identity(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", value) is None:
        raise LaunchError("Invalid physical PCI identity")
    domain, bus, tail = value.split(":")
    device, function = tail.split(".")
    return tuple(int(part, 16) for part in (domain, bus, device, function))


## @brief Parse only the recorded allocation query, never enumerate GPUs here.
# @see cosmology_tools
#
# @param path Input/output filesystem path; freshness and hash requirements are defined by this routine.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def allocation_evidence(path):
    """Parse only the recorded allocation query, never enumerate GPUs here."""
    raw = Path(path).read_bytes()
    try:
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8")), skipinitialspace=True))
    except (UnicodeError, csv.Error) as error:
        raise LaunchError(f"Invalid allocated-GPU CSV: {error}") from error
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise LaunchError("Allocation evidence must contain exactly four headerless GPU rows")
    devices, identifiers, physical = [], set(), set()
    for row in rows:
        name, identifier, pci, mig = (entry.strip() for entry in row)
        if (re.search(r"\bA100\b", name) is None or mig != "Disabled"
                or re.fullmatch(r"GPU-[0-9a-fA-F-]+", identifier) is None):
            raise LaunchError("Require four A100 devices with GPU UUIDs and MIG Disabled")
        identities = pci_identity(pci)
        identifiers.add(identifier.lower())
        physical.add(identities)
        devices.append({"name": name, "uuid": identifier, "pci": pci, "mig_mode": mig})
    if len(identifiers) != 4 or len(physical) != 4:
        raise LaunchError("Allocation evidence contains duplicate UUIDs or physical PCI devices")
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "devices": devices}


## @brief Enforce N_ranks*N_host_threads<=4 within the fixed four-CPU allocation.
# Two host threads are permitted for one or two GPU ranks. This changes only
# host scheduling: device binding, kernels, and scientific tolerances are unchanged.
# @see cosmology_tools
#
# @param environment Explicit subprocess environment; it does not by itself qualify an execution backend.
# @param ranks Positive MPI rank count; all expected snapshot shards must exist.
# @param kind Fixture selector from the module's declared supported cases.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def thread_environment(environment, ranks, kind):
    requested = environment.get("OMP_NUM_THREADS", "1")
    if requested not in ("1", "2"):
        raise LaunchError("Only OMP_NUM_THREADS=1 or 2 is permitted")
    threads = int(requested) if kind == "gpu" else 1
    if ranks*threads > 4:
        raise LaunchError("Total rank times host-thread CPU use must not exceed four")
    return {**BlasEnvironment, "OMP_NUM_THREADS": str(threads)}


## @brief Validate bindings.
# @see cosmology_tools
#
# @param records Retained evidence records under the calling validator's schema and ordering.
# @param ranks Positive MPI rank count; all expected snapshot shards must exist.
# @param kind Fixture selector from the module's declared supported cases.
# @param config Explicit validated launcher/campaign configuration; machine identity and hashes are checked separately.
# @param target Selected executable or runtime binding target as defined by this launcher/test.
# @param requestedTarget Requested executable path resolved under the strict launcher configuration.
# @param allocationRecord Verified scheduler/device allocation evidence; it must match the current launch.
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def validate_bindings(records, ranks, kind, config, target, requestedTarget, allocationRecord):
    if kind == "cpu":
        if records:
            raise LaunchError("CPU reference unexpectedly emitted GPU binding records")
        return
    if len(records) != ranks:
        raise LaunchError(f"Expected {ranks} GPU binding records, received {len(records)}")
    hosts, globalRanks, localRanks, physical, allocations = set(), set(), set(), set(), set()
    allocatedPci = {pci_identity(device["pci"]) for device in allocationRecord["devices"]}
    for row in records:
        if not isinstance(row, dict):
            raise LaunchError("GPU binding must be a JSON object")
        for key in ("rank", "local_rank", "local_size", "world_size", "runtime_device_count", "visible_device_ordinal"):
            if type(row.get(key)) is not int:
                raise LaunchError(f"GPU binding {key} must be an integer")
        if (row.get("schema") != "ippl-gpu-binding-v1" or row["local_size"] != ranks
                or row["world_size"] != ranks or row["runtime_device_count"] != 1
                or row["visible_device_ordinal"] != 0 or row.get("executable") != target
                or row.get("requested_executable") != requestedTarget
                or row.get("helper_path") != str(Path(config["gpu_helper"]).resolve())
                or row.get("helper_sha256") != config["sha256"][str(Path(config["gpu_helper"]).resolve())]):
            raise LaunchError("GPU binding rank/device/artifact contract mismatch")
        host, pci, tokens = row.get("host"), row.get("pci"), row.get("allocated_tokens")
        if not isinstance(host, str) or not host.strip():
            raise LaunchError("GPU binding lacks a host")
        identity = pci_identity(pci)
        if identity not in allocatedPci:
            raise LaunchError("Observed GPU PCI device is outside the recorded Slurm allocation")
        if (not isinstance(tokens, list) or len(tokens) < ranks or not all(isinstance(token, str) and token for token in tokens)
                or len(set(tokens)) != len(tokens) or any(token.upper().startswith("MIG-") for token in tokens)
                or not 0 <= row["local_rank"] < ranks
                or row.get("visible_token") != tokens[row["local_rank"]]):
            raise LaunchError("Invalid GPU allocation selection or unsupported MIG allocation")
        physical.add(identity)
        hosts.add(host)
        globalRanks.add(row["rank"])
        localRanks.add(row["local_rank"])
        allocations.add(tuple(tokens))
    if (globalRanks != set(range(ranks)) or localRanks != set(range(ranks))
            or len(hosts) != 1 or len(physical) != ranks or len(allocations) != 1):
        raise LaunchError("Require complete single-host ranks and distinct physical PCI devices")


## @brief Evaluate the save manifest helper in the documented module workflow.
# @see cosmology_tools
#
# @param path Input/output filesystem path; freshness and hash requirements are defined by this routine.
# @param record Structured retained evidence record following this module's declared schema.
# @param create Whether this phase is authorized to create a new artifact/evidence destination.
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def _save_manifest(path, record, *, create=False):
    # All filenames are uniquely claimed before launch; only this launch's
    # manifest is replaced after its terminal state has been determined.
    temporary = path if create else path.with_suffix(".json.partial")
    with temporary.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if not create:
        os.replace(temporary, path)


## @brief Evaluate the tee helper in the documented module workflow.
# @see cosmology_tools
#
# @param raw Unparsed runtime/configuration input to be validated before use.
# @param log Path retaining combined subprocess output and failure evidence.
# @return None; success is expressed by the recorded artifact/check or by returning without an exception.
def _tee(raw, log):
    log.write(raw)
    log.flush()
    if hasattr(sys.stdout, "buffer"):
        sys.stdout.buffer.write(raw)
    else:  # StringIO in tests; the evidence log always retains original bytes.
        sys.stdout.write(raw.decode("utf-8", errors="backslashreplace"))
    sys.stdout.flush()


## @brief Launch the documented module workflow.
# @see cosmology_tools
#
# @param configPath Absolute strict-launcher configuration JSON path.
# @param ranks Positive MPI rank count; all expected snapshot shards must exist.
# @param targetArguments Untouched argument vector passed to the selected executable; no extra Kokkos options are invented.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def launch(configPath, ranks, targetArguments):
    config, target, kind, configDigest = load_config(configPath, targetArguments)
    slurm = allocation(os.environ)
    allocated = allocation_evidence(config["allocation_evidence"])
    threads = thread_environment(os.environ, ranks, kind)
    command = build_command(config, ranks, kind, targetArguments)
    root = Path(config["evidence_dir"])
    root.mkdir(parents=True, exist_ok=True)
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{os.getpid()}-{uuid.uuid4().hex}"
    logPath, manifestPath = root / (identifier + ".log"), root / (identifier + ".json")
    pinned = dict(config["sha256"])
    pinned[str(configPath)] = configDigest
    pinned[allocated["path"]] = allocated["sha256"]
    verify_hashes(pinned)
    record = {"schema": "ippl-merlin-mpi-launch-v1", "status": "running", "return_code": None,
              "wrapper_return_code": None, "command": command, "requested_target_arguments": list(targetArguments),
              "target": target, "target_kind": kind, "ranks": ranks, "slurm": slurm,
              "thread_environment": threads, "hashes": pinned, "bindings": [], "binding_errors": [],
              "allocation_evidence": allocated,
              "log": str(logPath), "started_utc": datetime.now(timezone.utc).isoformat(),
              "scientific_acceptance": "not assessed", "gpu_model_and_mig_mode": "from pinned allocation-preflight CSV, not inferred from PCI"}
    process = None
    with logPath.open("xb") as log:
        _save_manifest(manifestPath, record, create=True)
        try:
            # Deliberately inherit the wrapper's process group. StudyStorage
            # starts the outer group and owns group-wide timeout termination.
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=dict(os.environ, **threads), start_new_session=False)
            for raw in iter(process.stdout.readline, b""):
                _tee(raw, log)
                if raw.startswith(BindingPrefix):
                    try:
                        record["bindings"].append(json.loads(raw[len(BindingPrefix):]))
                    except (ValueError, UnicodeDecodeError) as error:
                        record["binding_errors"].append(str(error))
            record["return_code"] = process.wait()
            verify_hashes(pinned)
            if record["return_code"] != 0:
                record["status"] = "execution_failed"
                record["wrapper_return_code"] = record["return_code"] if record["return_code"] > 0 else 128-record["return_code"]
            else:
                if record["binding_errors"]:
                    raise LaunchError("Malformed GPU_BINDING JSON in merged MPI output")
                validate_bindings(record["bindings"], ranks, kind, config, target, targetArguments[0], allocated)
                record.update(status="launch_complete", wrapper_return_code=0)
        except (Exception, KeyboardInterrupt) as error:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process is not None:
                record["return_code"] = process.returncode
            record.update(status="launcher_rejected", wrapper_return_code=2, error=str(error))
            _tee(("GPU_LAUNCH_ERROR " + str(error) + "\n").encode(), log)
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            record["log_sha256"] = sha256(logPath)
            _save_manifest(manifestPath, record)
    return record["wrapper_return_code"]


## @brief Parse the documented command-line interface and execute its selected workflow, retaining nonzero failures.
# @see cosmology_tools
#
# @param arguments Parsed command-line options; see main/--help and the module workflow contract.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def main(arguments=None):
    arguments = sys.argv[1:] if arguments is None else arguments
    if list(arguments) == ["--help"]:
        print(__doc__)
        return 0
    try:
        return launch(*parse_arguments(arguments))
    except (LaunchError, OSError) as error:
        print(f"GPU_LAUNCH_ERROR {error}", file=sys.stderr, flush=True)
        return 2


## @cond CLI_DISPATCH
if __name__ == "__main__":
    raise SystemExit(main())
## @endcond
