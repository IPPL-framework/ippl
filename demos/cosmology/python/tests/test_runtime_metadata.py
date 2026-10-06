#!/usr/bin/env python3
## @file test_runtime_metadata.py
# @brief CPU/GPU execution metadata regressions; no device or simulation required.
# @ingroup cosmology_python
# @see cosmology_tools cosmology_validation
# Tests assert declared invariants using isolated fixtures; no scientific tolerances are relaxed by documentation.
"""CPU/GPU execution metadata regressions; no device or simulation required."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_metadata import validate_runtime_metadata


## @brief Evaluate the explicit helper in the documented module workflow.
# @see cosmology_tools
#
# @param backend Declared execution backend/space; labels alone do not qualify hardware correctness.
# @param memory Recorded Kokkos memory-space name; it must be consistent with the execution backend.
# @param ranks Positive MPI rank count; all expected snapshot shards must exist.
# @param host Recorded host execution concurrency/identity used by the fixture.
# @param concurrency Recorded execution-space concurrency, distinct from host thread count on CUDA.
# @return The computed value or retained diagnostic record described by this routine; physical units follow the module contract.
def explicit(backend="OpenMP", memory="Host", *, ranks=4, host=1, concurrency=1):
    return {"ranks": str(ranks), "threads": str(concurrency),
            "execution_concurrency": str(concurrency), "host_threads": str(host),
            "execution_space": backend, "memory_space": memory}


## @brief Regression suite for RuntimeMetadata.
# @see cosmology_tools
class RuntimeMetadataTests(unittest.TestCase):
    ## @brief Verify explicit cpu configuration.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_explicit_cpu_configuration(self):
        for threads in (1, 2):
            result = validate_runtime_metadata(explicit(host=threads, concurrency=threads), 4, threads)
            self.assertEqual(result["contract"], "explicit")
            self.assertEqual(result["host_threads"], threads)
            self.assertEqual(result["execution_concurrency"], threads)
            self.assertFalse(result["gpu_execution"])
        self.assertFalse(validate_runtime_metadata(explicit("Serial"), 4)["gpu_execution"])

    ## @brief Verify gpu concurrency is not omp thread count.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_gpu_concurrency_is_not_omp_thread_count(self):
        for host in (1, 2):
            meta = explicit("Cuda", "Cuda", host=host, concurrency=221184)
            result = validate_runtime_metadata(meta, 4, host, expected_execution_space="Cuda")
            self.assertEqual(result["execution_concurrency"], 221184)
            self.assertEqual(result["host_threads"], host)
            self.assertTrue(result["gpu_execution"])
            self.assertEqual(meta["threads"], "221184")
            self.assertIn("not GPU thread count or scaling", result["scope"])
        with self.assertRaises(ValueError):
            validate_runtime_metadata(explicit("Cuda", "Cuda", host=1, concurrency=221184), 4, 2)

    ## @brief Verify named gpu memory spaces are backend specific.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_named_gpu_memory_spaces_are_backend_specific(self):
        for backend, memory in (("Cuda", "Cuda"), ("Cuda", "CudaUVM"),
                                ("HIP", "HIP"), ("HIP", "HIPManaged"),
                                ("SYCL", "SYCLDeviceUSM"), ("SYCL", "SYCLSharedUSM")):
            with self.subTest(backend=backend, memory=memory):
                self.assertTrue(validate_runtime_metadata(
                    explicit(backend, memory, concurrency=1024), 4)["gpu_execution"])

    ## @brief Verify legacy cpu and native keep strict host gate.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_legacy_cpu_and_native_keep_strict_host_gate(self):
        for code in ("ippl", "fastpm"):
            for spaces in ({}, {"execution_space": "OpenMP", "memory_space": "Host"}):
                metadata = {"ranks": "3", "threads": "1", **spaces}
                result = validate_runtime_metadata(metadata, 3, code=code)
                self.assertEqual(result["contract"], "legacy_cpu")
                self.assertEqual(result["host_threads"], 1)
                with self.assertRaises(ValueError):
                    validate_runtime_metadata(metadata, 3, 2, code=code)
        self.assertEqual(validate_runtime_metadata({"ranks": "1", "threads": "2"}, 1, 2)["host_threads"], 2)

    ## @brief Verify gpu cannot use legacy or partial metadata.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_gpu_cannot_use_legacy_or_partial_metadata(self):
        metadata = explicit("Cuda", "Cuda", concurrency=221184)
        for missing in ("execution_concurrency", "host_threads", "execution_space", "memory_space"):
            incomplete = dict(metadata)
            del incomplete[missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                validate_runtime_metadata(incomplete, 4)
        legacy = {"ranks": "4", "threads": "1", "execution_space": "Cuda", "memory_space": "Cuda"}
        with self.assertRaises(ValueError):
            validate_runtime_metadata(legacy, 4)

    ## @brief Verify new cpu metadata cannot fall back if partial.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_new_cpu_metadata_cannot_fall_back_if_partial(self):
        metadata = explicit()
        for missing in ("execution_concurrency", "host_threads", "execution_space", "memory_space"):
            incomplete = dict(metadata)
            del incomplete[missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                validate_runtime_metadata(incomplete, 4)

    ## @brief Verify counts must be positive integers and consistent.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_counts_must_be_positive_integers_and_consistent(self):
        for key in ("ranks", "threads", "execution_concurrency", "host_threads"):
            for bad in (None, "", "0", "-1", "1.0", "NaN", "2e0", True, 1.5):
                with self.subTest(key=key, value=bad), self.assertRaises(ValueError):
                    validate_runtime_metadata(dict(explicit(), **{key: bad}), 4)
        for update in ({"threads": "2"}, {"execution_concurrency": "2"},
                       {"threads": "2", "execution_concurrency": "2"}, {"ranks": "3"}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_runtime_metadata(dict(explicit(), **update), 4)
        with self.assertRaises(ValueError):
            validate_runtime_metadata(explicit("Serial", host=2, concurrency=2), 4, 2)

    ## @brief Verify incompatible unknown or empty spaces reject.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_incompatible_unknown_or_empty_spaces_reject(self):
        for backend, memory in (("Cuda", "Host"), ("OpenMP", "Cuda"), ("Cuda", "HIP"),
                                ("CUDA", "Cuda"), ("Cuda", "CudaSpace"), ("Unknown", "Host"),
                                (None, None), ("", ""), ("OpenMP", None), (None, "Host"),
                                ([], []), (1, 2)):
            with self.subTest(backend=backend, memory=memory), self.assertRaises(ValueError):
                validate_runtime_metadata(explicit(backend, memory), 4)
        for spaces in ({"execution_space": "OpenMP"}, {"memory_space": "Host"},
                       {"execution_space": "", "memory_space": ""}):
            with self.subTest(spaces=spaces), self.assertRaises(ValueError):
                validate_runtime_metadata({"ranks": "4", "threads": "1", **spaces}, 4)

    ## @brief Verify requested gpu backend rejects cpu fallback.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_requested_gpu_backend_rejects_cpu_fallback(self):
        for metadata in (explicit(), {"ranks": "4", "threads": "1"},
                         explicit("HIP", "HIP", concurrency=1024)):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                validate_runtime_metadata(metadata, 4, expected_execution_space="Cuda")

    ## @brief Verify native reference cannot reinterpret threads as gpu.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_native_reference_cannot_reinterpret_threads_as_gpu(self):
        for metadata in (explicit(), explicit("Cuda", "Cuda", concurrency=1024),
                         {"ranks": "4", "threads": "1024"}):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                validate_runtime_metadata(metadata, 4, code="fastpm")

    ## @brief Verify malformed metadata and requests reject.
    # @see cosmology_tools
    # @return None; success is expressed by the recorded artifact/check or by returning without an exception.
    def test_malformed_metadata_and_requests_reject(self):
        for metadata in (None, [], "ranks=4", {}, {"ranks": "4"}, {"threads": "1"}):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                validate_runtime_metadata(metadata, 4)
        for ranks, threads in ((0, 1), (4, 0), (True, 1), (4, None)):
            with self.subTest(ranks=ranks, threads=threads), self.assertRaises(ValueError):
                validate_runtime_metadata(explicit(), ranks, threads)
        with self.assertRaises(ValueError):
            validate_runtime_metadata(explicit(), 4, code="unrecognized")


## @cond CLI_DISPATCH
if __name__ == "__main__":
    unittest.main()
## @endcond
