## @file runtime_metadata.py
# @brief Strict CPU/GPU execution metadata, independent of numerical acceptance gates.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
"""Strict CPU/GPU execution metadata, independent of numerical acceptance gates.

Historical IPPL ``threads`` means DefaultExecutionSpace.concurrency(). On a
GPU that is not a host thread count. Native FastPM and old CPU IPPL outputs
retain their strict legacy rank/thread contract. New IPPL metadata must state
both concurrency values and a consistent execution/memory-space pair.
"""
from __future__ import annotations

from collections.abc import Mapping
import re


## @var CpuSpaces
# @brief Named CpuSpaces protocol/schema value; the source initializer records its exact contents.
CpuSpaces = {"Serial": {"Host"}, "OpenMP": {"Host"}, "Threads": {"Host"}}
# Kokkos name() values, not C++ type spellings. Only explicitly listed,
# backend-compatible device or managed memory spaces are accepted.
## @var GpuSpaces
# @brief Named GpuSpaces protocol/schema value; the source initializer records its exact contents.
GpuSpaces = {"Cuda": {"Cuda", "CudaUVM"}, "HIP": {"HIP", "HIPManaged"},
             "SYCL": {"SYCLDeviceUSM", "SYCLSharedUSM"}}
## @var ExplicitFields
# @brief Named ExplicitFields protocol/schema value; the source initializer records its exact contents.
ExplicitFields = {"execution_concurrency", "host_threads", "execution_space", "memory_space"}


## @brief Evaluate the positive integer helper in the documented module workflow.
# @see cosmology_tools
#
# @param value Measured or serialized scalar in the declared metric/schema; no normalization is inferred.
# @param name Stable artifact/run/check identifier as defined by the caller.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def _positive_integer(value, name):
    if isinstance(value, bool) or not re.fullmatch(r"[1-9][0-9]*", str(value)):
        raise ValueError(f"Execution metadata {name} must be a positive integer")
    return int(value)


## @brief Validate requested rank/host configuration, returning normalized evidence.
# @see cosmology_tools
#
# @param metadata Retained named unit/model/execution fields associated with the source artifact.
# @param ranks Positive MPI rank count; all expected snapshot shards must exist.
# @param host_threads Expected host execution concurrency; distinct from CUDA device concurrency.
# @param code Solver identifier (IPPL, native FastPM or GADGET) selected by the protocol.
# @param expected_execution_space Expected recorded Kokkos execution-space name for the audited backend.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def validate_runtime_metadata(metadata, ranks, host_threads=1, *, code="ippl",
                              expected_execution_space=None):
    """Validate requested rank/host configuration, returning normalized evidence.

    ``expected_execution_space='Cuda'`` additionally rules out CPU fallback.
    GPU execution concurrency is checked for positivity and internal consistency,
    never equated with OMP_NUM_THREADS. This checks configuration, not GPU scaling.
    """
    if not isinstance(metadata, Mapping):
        raise ValueError("Execution metadata must be a mapping")
    if code not in ("ippl", "fastpm"):
        raise ValueError("Unknown execution metadata producer")
    requestedRanks = _positive_integer(ranks, "requested ranks")
    requestedThreads = _positive_integer(host_threads, "requested host_threads")
    actualRanks = _positive_integer(metadata.get("ranks"), "ranks")
    legacyThreads = _positive_integer(metadata.get("threads"), "threads")
    if actualRanks != requestedRanks:
        raise ValueError("Execution metadata MPI rank count differs from request")
    executionSpace = metadata.get("execution_space")
    memorySpace = metadata.get("memory_space")
    explicit = bool({"execution_concurrency", "host_threads"}.intersection(metadata))
    if explicit and not ExplicitFields.issubset(metadata):
        raise ValueError("Incomplete explicit execution metadata")
    if any(key in metadata and (not isinstance(metadata[key], str) or not metadata[key])
           for key in ("execution_space", "memory_space")):
        raise ValueError("Execution and memory spaces must be nonempty strings")
    if bool(executionSpace) != bool(memorySpace):
        raise ValueError("Execution and memory spaces must both be present")
    isGpu = executionSpace in GpuSpaces
    if executionSpace is not None:
        allowed = GpuSpaces if isGpu else CpuSpaces
        if executionSpace not in allowed or memorySpace not in allowed[executionSpace]:
            raise ValueError("Unsupported or inconsistent execution/memory-space pair")
    if expected_execution_space is not None and executionSpace != expected_execution_space:
        raise ValueError("Execution backend differs from the requested backend")
    if code == "fastpm" and (isGpu or explicit):
        raise ValueError("Native reference must retain its CPU legacy metadata contract")
    if isGpu and not explicit:
        raise ValueError("GPU output requires explicit execution_concurrency and host_threads")
    if explicit:
        concurrency = _positive_integer(metadata["execution_concurrency"], "execution_concurrency")
        actualHostThreads = _positive_integer(metadata["host_threads"], "host_threads")
        if legacyThreads != concurrency:
            raise ValueError("Legacy threads and execution_concurrency disagree")
        if not isGpu and concurrency != actualHostThreads:
            raise ValueError("CPU execution concurrency and host_threads disagree")
    else:
        concurrency = actualHostThreads = legacyThreads
    if executionSpace == "Serial" and actualHostThreads != 1:
        raise ValueError("Serial execution must have one host thread")
    if actualHostThreads != requestedThreads:
        raise ValueError("Actual host thread count differs from request")
    return {"contract": "explicit" if explicit else "legacy_cpu",
            "ranks": actualRanks, "host_threads": actualHostThreads,
            "execution_concurrency": concurrency, "execution_space": executionSpace,
            "memory_space": memorySpace, "gpu_execution": isGpu,
            "scope": "MPI/backend/host-thread configuration, not GPU thread count or scaling"}
