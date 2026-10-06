"""GPU binding tests with fake CUDA functions and exec; no GPU is queried."""
from contextlib import redirect_stderr, redirect_stdout
import ctypes
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch


HelperPath = Path(__file__).resolve().parents[1] / "merlin" / "gpu_rank.py"
Spec = importlib.util.spec_from_file_location("cosmology_merlin_gpu_rank", HelperPath)
binding = importlib.util.module_from_spec(Spec)
sys.modules[Spec.name] = binding
Spec.loader.exec_module(binding)


def environment(**changes):
    return {"OMPI_COMM_WORLD_RANK": "6", "OMPI_COMM_WORLD_LOCAL_RANK": "2",
            "OMPI_COMM_WORLD_LOCAL_SIZE": "4", "OMPI_COMM_WORLD_SIZE": "8",
            "CUDA_VISIBLE_DEVICES": "3,1,7,5", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", **changes}


def runtime(count=1, countStatus=0, pciStatus=0, pci=b"0000:AF:00.0"):
    def countFunction(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = count
        return countStatus
    def pciFunction(buffer, length, ordinal):
        if ordinal != 0:
            raise AssertionError("Only masked device zero may be queried")
        buffer.value = pci
        return pciStatus
    library = Mock()
    library._name = "/allocated-job/libcudart.so"
    library.cudaGetDeviceCount = Mock(side_effect=countFunction)
    library.cudaDeviceGetPCIBusId = Mock(side_effect=pciFunction)
    return library


class BindingPlanTests(unittest.TestCase):
    def test_numeric_tokens_use_scheduler_order_not_global_rank(self):
        original = environment()
        plan = binding.binding_plan(original)
        self.assertEqual(plan.visibleToken, "7")
        self.assertEqual(plan.rank, 6)
        self.assertEqual(plan.localRank, 2)
        self.assertEqual(plan.allocatedTokens, ("3", "1", "7", "5"))
        self.assertEqual(original["CUDA_VISIBLE_DEVICES"], "3,1,7,5")

    def test_uuid_and_mig_tokens_are_preserved(self):
        tokens = ("GPU-a123-b456", "GPU-CDEF-7890", "MIG-GPU-a123-b456/1/0", "MIG-1234-abcd")
        for localRank, token in enumerate(tokens):
            with self.subTest(token=token):
                plan = binding.binding_plan(environment(CUDA_VISIBLE_DEVICES=",".join(tokens),
                                                        OMPI_COMM_WORLD_LOCAL_RANK=str(localRank)))
                self.assertEqual(plan.visibleToken, token)

    def test_every_local_rank_fails_before_runtime_if_allocation_is_too_small(self):
        for localRank in range(4):
            with self.subTest(localRank=localRank):
                with self.assertRaisesRegex(binding.BindingError, "Insufficient"):
                    binding.binding_plan(environment(CUDA_VISIBLE_DEVICES="0,1", OMPI_COMM_WORLD_LOCAL_RANK=str(localRank)))
        with self.assertRaisesRegex(binding.BindingError, "Insufficient"):
            binding.binding_plan(environment(CUDA_VISIBLE_DEVICES="GPU-a123"))

    def test_missing_or_invalid_rank_information_rejected(self):
        for name in ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_SIZE"):
            missing = environment()
            del missing[name]
            with self.subTest(name=name), self.assertRaises(binding.BindingError):
                binding.binding_plan(missing)
            for value in ("", "-1", "1.5", "x", " 2"):
                with self.subTest(name=name, value=value), self.assertRaises(binding.BindingError):
                    binding.binding_plan(environment(**{name: value}))
        for changes in ({"OMPI_COMM_WORLD_LOCAL_RANK": "4"}, {"OMPI_COMM_WORLD_LOCAL_SIZE": "0"},
                        {"OMPI_COMM_WORLD_RANK": "8"}, {"OMPI_COMM_WORLD_SIZE": "2"}):
            with self.assertRaises(binding.BindingError):
                binding.binding_plan(environment(**changes))

    def test_disabled_empty_malformed_duplicate_devices_rejected(self):
        for tokens in ("", "-1", "NoDevFiles", "0,,1,2", "0,1,2,", "0,1,1,2", "0,00,1,2",
                       "GPU-abcd,GPU-ABCD,1,2", "all", "0;1;2;3", "GPU-x,1,2,3"):
            with self.subTest(tokens=tokens), self.assertRaises(binding.BindingError):
                binding.binding_plan(environment(CUDA_VISIBLE_DEVICES=tokens))
        missing = environment()
        del missing["CUDA_VISIBLE_DEVICES"]
        with self.assertRaises(binding.BindingError):
            binding.binding_plan(missing)

    def test_extra_allocated_devices_and_single_rank_are_valid(self):
        plan = binding.binding_plan(environment(OMPI_COMM_WORLD_LOCAL_RANK="0", OMPI_COMM_WORLD_LOCAL_SIZE="1"))
        self.assertEqual(plan.visibleToken, "3")
        plan = binding.binding_plan(environment(OMPI_COMM_WORLD_LOCAL_RANK="0", OMPI_COMM_WORLD_LOCAL_SIZE="1",
                                                CUDA_VISIBLE_DEVICES="MIG-GPU-dead-beef/2/1"))
        self.assertEqual(plan.visibleToken, "MIG-GPU-dead-beef/2/1")


class RuntimeAndExecTests(unittest.TestCase):
    def test_probe_queries_only_masked_zero_and_reports_pci(self):
        library = runtime()
        result = binding.probe_cuda(library)
        self.assertEqual(result["pci"], "0000:af:00.0")
        self.assertEqual(result["runtime_device_count"], 1)
        self.assertEqual(result["cudart_library"], "/allocated-job/libcudart.so")
        library.cudaGetDeviceCount.assert_called_once()
        self.assertEqual(library.cudaDeviceGetPCIBusId.call_args.args[2], 0)
        self.assertEqual(library.cudaGetDeviceCount.restype, ctypes.c_int)

    def test_runtime_errors_zero_or_multiple_devices_and_bad_pci_fail_closed(self):
        for options in ({"count": 0}, {"count": 2}, {"countStatus": 100}, {"pciStatus": 1},
                        {"pci": b""}, {"pci": b"garbage"}, {"pci": b"\xff"}):
            with self.subTest(options=options), self.assertRaises(binding.BindingError):
                binding.probe_cuda(runtime(**options))

    def test_explicit_runtime_override_uses_no_discovery_or_fallback(self):
        sentinel = object()
        with (patch.dict(binding.os.environ, {"IPPL_CUDART_LIBRARY": "/job/cuda/lib64/libcudart.so"}, clear=True),
              patch.object(binding.ctypes.util, "find_library", side_effect=AssertionError("No discovery")),
              patch.object(binding.ctypes, "CDLL", return_value=sentinel) as loader):
            self.assertIs(binding.load_runtime(), sentinel)
            loader.assert_called_once_with("/job/cuda/lib64/libcudart.so")
        with (patch.dict(binding.os.environ, {"IPPL_CUDART_LIBRARY": "/missing/runtime"}, clear=True),
              patch.object(binding.ctypes, "CDLL", side_effect=OSError("missing")) as loader):
            with self.assertRaises(binding.BindingError):
                binding.load_runtime()
            self.assertEqual(loader.call_count, 1)

    def test_binding_happens_before_cuda_and_exec_keeps_exact_arguments(self):
        arguments = ["target", "32", "32", "168.75", "--kokkos-device-id=0", "path with spaces", "literal$argument"]
        class RecordingStream(io.StringIO):
            def __init__(self):
                super().__init__()
                self.writes, self.flushed = [], False
            def write(self, value):
                self.writes.append(value)
                return super().write(value)
            def flush(self):
                self.flushed = True
                return super().flush()
        output = RecordingStream()
        def load():
            self.assertEqual(binding.os.environ["CUDA_VISIBLE_DEVICES"], "7")
            self.assertEqual(binding.os.environ["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
            return runtime()
        with (patch.dict(binding.os.environ, environment(EXTRA_SETTING="preserved"), clear=True),
              patch.object(binding.shutil, "which", return_value="/allocated/target"),
              patch.object(binding.socket, "gethostname", return_value="gpu-host"),
              patch.object(binding, "load_runtime", side_effect=load),
              patch.object(binding.os, "execvpe") as execute, redirect_stdout(output)):
            def before_exec(*unused):
                self.assertTrue(output.flushed)
                self.assertEqual(len(output.writes), 1)
                self.assertTrue(output.writes[0].endswith("\n"))
            execute.side_effect = before_exec
            self.assertEqual(binding.main(arguments), 0)
            execute.assert_called_once()
            target, passedArguments, passedEnvironment = execute.call_args.args
            self.assertEqual(target, "target")
            self.assertEqual(passedArguments, arguments)
            self.assertEqual(passedEnvironment, environment(EXTRA_SETTING="preserved", CUDA_VISIBLE_DEVICES="7"))
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("GPU_BINDING "))
        record = json.loads(lines[0].removeprefix("GPU_BINDING "))
        self.assertEqual(record["rank"], 6)
        self.assertEqual(record["local_rank"], 2)
        self.assertEqual(record["visible_token"], "7")
        self.assertEqual(record["pci"], "0000:af:00.0")
        self.assertEqual(record["executable"], "/allocated/target")
        self.assertEqual(len(record["helper_sha256"]), 64)

    def test_failure_never_executes_or_emits_success_binding(self):
        for currentEnvironment, library in ((environment(CUDA_VISIBLE_DEVICES="0"), runtime()),
                                            (environment(), runtime(count=2))):
            stdout, stderr = io.StringIO(), io.StringIO()
            with (patch.dict(binding.os.environ, currentEnvironment, clear=True),
                  patch.object(binding.shutil, "which", return_value="/allocated/target"),
                  patch.object(binding, "load_runtime", return_value=library),
                  patch.object(binding.os, "execvpe") as execute,
                  redirect_stdout(stdout), redirect_stderr(stderr)):
                self.assertEqual(binding.main(["target"]), 2)
                execute.assert_not_called()
            self.assertEqual(stdout.getvalue(), "")
            self.assertTrue(stderr.getvalue().startswith("GPU_BINDING_ERROR "))

    def test_help_missing_target_and_missing_rank_never_load_cuda(self):
        with (patch.dict(binding.os.environ, {}, clear=True),
              patch.object(binding, "load_runtime", side_effect=AssertionError("No GPU query")),
              patch.object(binding.os, "execvpe", side_effect=AssertionError("No exec")),
              redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
            self.assertEqual(binding.main(["--help"]), 0)
            self.assertEqual(binding.main([]), 2)
            self.assertEqual(binding.main(["target"]), 2)

    def test_missing_executable_never_loads_cuda(self):
        with (patch.dict(binding.os.environ, environment(), clear=True),
              patch.object(binding.shutil, "which", return_value=None),
              patch.object(binding, "load_runtime", side_effect=AssertionError("No CUDA query")),
              patch.object(binding.os, "execvpe", side_effect=AssertionError("No exec")),
              redirect_stderr(io.StringIO())):
            self.assertEqual(binding.main(["missing-target"]), 2)

    def test_runtime_loader_uses_only_mocked_normal_loader_candidates(self):
        sentinel = object()
        with (patch.dict(binding.os.environ, {}, clear=True),
              patch.object(binding.ctypes.util, "find_library", return_value="libcudart.so.12") as find,
              patch.object(binding.ctypes, "CDLL", side_effect=[OSError("not found"), sentinel]) as loader):
            self.assertIs(binding.load_runtime(), sentinel)
            find.assert_called_once_with("cudart")
            self.assertEqual([call.args[0] for call in loader.call_args_list], ["libcudart.so.12", "libcudart.so"])


if __name__ == "__main__":
    unittest.main()
