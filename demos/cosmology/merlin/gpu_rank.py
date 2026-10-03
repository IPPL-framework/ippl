#!/usr/bin/env python3
"""Bind one OpenMPI local rank to one scheduler-visible GPU, then exec.

Usage: python -B gpu_rank.py TARGET [TARGET_ARGUMENTS ...]

Each rank must inherit the full node allocation in CUDA_VISIBLE_DEVICES.
This wrapper does not discover GPUs, alter device ordering, or append Kokkos
arguments. An explicit IPPL_CUDART_LIBRARY may select the allocated job's CUDA
runtime; otherwise the runtime is located through the normal library loader.
Importing the module, asking for help, and the unit tests never query a GPU.

One GPU_BINDING JSON line is flushed before exec. The launcher must aggregate
these records and require distinct PCI IDs per host for full physical GPUs.
MIG instances can share a parent PCI ID: their unchanged tokens are recorded,
but this wrapper does not claim physical-GPU distinctness for MIG instances.
"""
from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys


class BindingError(RuntimeError):
    """Invalid allocation, rank information, or selected CUDA runtime device."""


@dataclass(frozen=True)
class Binding:
    rank: int
    localRank: int
    localSize: int
    visibleToken: str
    allocatedTokens: tuple[str, ...]
    worldSize: int | None


def _integer(environment, name, minimum=0):
    value = environment.get(name)
    if value is None or re.fullmatch(r"[0-9]+", value) is None or int(value) < minimum:
        raise BindingError(f"{name} must be a decimal integer >= {minimum}")
    return int(value)


def binding_plan(environment):
    """Pure allocation/rank validation: no environment mutation or CUDA calls."""
    rank = _integer(environment, "OMPI_COMM_WORLD_RANK")
    localRank = _integer(environment, "OMPI_COMM_WORLD_LOCAL_RANK")
    localSize = _integer(environment, "OMPI_COMM_WORLD_LOCAL_SIZE", 1)
    worldSize = (_integer(environment, "OMPI_COMM_WORLD_SIZE", 1)
                 if "OMPI_COMM_WORLD_SIZE" in environment else None)
    if localRank >= localSize or (worldSize is not None and (rank >= worldSize or localSize > worldSize)):
        raise BindingError("Inconsistent OpenMPI rank or size")
    allocated = environment.get("CUDA_VISIBLE_DEVICES")
    if allocated is None:
        raise BindingError("CUDA_VISIBLE_DEVICES must explicitly identify the scheduler allocation")
    tokens = tuple(token.strip() for token in allocated.split(","))
    # UUID prefixes and both CUDA MIG token forms are deliberately preserved.
    # The runtime, after masking, must confirm that the selected token is usable.
    if not tokens or any(re.fullmatch(r"(?:[0-9]+|GPU-[0-9A-Fa-f-]+|MIG-[0-9A-Fa-f-]+|MIG-GPU-[0-9A-Fa-f-]+/[0-9]+/[0-9]+)",
                                     token) is None for token in tokens):
        raise BindingError("CUDA_VISIBLE_DEVICES contains an empty, disabled, or unsupported GPU token")
    identities = [str(int(token)) if token.isdecimal() else token.lower() for token in tokens]
    if len(set(identities)) != len(identities):
        raise BindingError("CUDA_VISIBLE_DEVICES contains duplicate allocation tokens")
    if len(tokens) < localSize:
        raise BindingError(f"Insufficient allocated GPU tokens: {len(tokens)} for {localSize} local ranks; "
                           "each rank must inherit the complete node allocation")
    return Binding(rank, localRank, localSize, tokens[localRank], tokens, worldSize)


def load_runtime():
    """Called only after the visible-device mask is narrowed for an actual run."""
    explicit = os.environ.get("IPPL_CUDART_LIBRARY")
    if explicit is not None:
        if not explicit:
            raise BindingError("IPPL_CUDART_LIBRARY is empty")
        candidates = [explicit]
    else:
        found = ctypes.util.find_library("cudart")
        candidates = list(dict.fromkeys(name for name in (found, "libcudart.so", "libcudart.so.12", "libcudart.so.13") if name))
    failures = []
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError as error:
            failures.append(f"{candidate}: {error}")
    raise BindingError("Cannot load CUDA runtime; set IPPL_CUDART_LIBRARY to the job's libcudart: " + "; ".join(failures))


def probe_cuda(runtime=None):
    """Inspect only visible device zero; never enumerate the unmasked devices."""
    runtime = load_runtime() if runtime is None else runtime
    try:
        countFunction, pciFunction = runtime.cudaGetDeviceCount, runtime.cudaDeviceGetPCIBusId
    except AttributeError as error:
        raise BindingError("Selected library lacks the required CUDA runtime API") from error
    countFunction.argtypes, countFunction.restype = [ctypes.POINTER(ctypes.c_int)], ctypes.c_int
    pciFunction.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int]
    pciFunction.restype = ctypes.c_int
    count = ctypes.c_int()
    status = countFunction(ctypes.byref(count))
    if status != 0 or count.value != 1:
        raise BindingError(f"Expected exactly one usable CUDA device after masking; status={status}, count={count.value}")
    pci = ctypes.create_string_buffer(64)
    status = pciFunction(pci, len(pci), 0)
    if status != 0:
        raise BindingError(f"Cannot query visible device zero PCI identity; CUDA status={status}")
    try:
        pciId = pci.value.decode("ascii").lower()
    except UnicodeDecodeError as error:
        raise BindingError("CUDA returned a non-ASCII PCI identity") from error
    if re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", pciId) is None:
        raise BindingError(f"CUDA returned an invalid PCI identity: {pciId!r}")
    return {"pci": pciId, "runtime_device_count": count.value,
            "cudart_library": str(getattr(runtime, "_name", "injected runtime"))}


def run(targetArguments):
    if not targetArguments or not targetArguments[0]:
        raise BindingError("Supply a target executable and its untouched arguments")
    plan = binding_plan(os.environ)
    executable = shutil.which(targetArguments[0])
    if executable is None:
        raise BindingError(f"Target executable is not available: {targetArguments[0]}")
    executable = str(Path(executable).resolve())
    # CUDA has not been loaded before this assignment. Keep the scheduler's
    # original token, including UUID/MIG syntax and existing device ordering.
    os.environ["CUDA_VISIBLE_DEVICES"] = plan.visibleToken
    probe = probe_cuda()
    record = {"schema": "ippl-gpu-binding-v1", "host": socket.gethostname(),
              "rank": plan.rank, "local_rank": plan.localRank, "local_size": plan.localSize,
              "world_size": plan.worldSize, "visible_token": plan.visibleToken,
              "allocated_tokens": list(plan.allocatedTokens), "visible_device_ordinal": 0,
              "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
              "executable": executable, "requested_executable": targetArguments[0],
              "helper_path": str(Path(__file__).resolve()),
              "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **probe}
    # One write keeps the record and newline together even under unbuffered
    # Python, where separate print() writes could interleave between ranks.
    sys.stdout.write("GPU_BINDING " + json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
    sys.stdout.flush()
    # exec replaces the wrapper: target args, PID, signal ownership and MPI
    # rank environment remain intact; only CUDA_VISIBLE_DEVICES is narrowed.
    os.execvpe(targetArguments[0], targetArguments, dict(os.environ))


def main(arguments=None):
    arguments = sys.argv[1:] if arguments is None else list(arguments)
    if arguments == ["--help"]:
        print(__doc__)
        return 0
    try:
        run(arguments)
    except (BindingError, OSError) as error:
        print(f"GPU_BINDING_ERROR {error}", file=sys.stderr, flush=True)
        return 2
    return 0  # Reachable only when exec is mocked by a unit test.


if __name__ == "__main__":
    raise SystemExit(main())
