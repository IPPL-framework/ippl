# 05: Add CI jobs and bencher.dev submission

**What to build:** GitLab CI jobs on Eiger OpenMP, Daint GH200 CUDA, and MI300 ROCm that build with `IPPL_ENABLE_BENCHMARK=ON`, run `LandauDampingBench`, and submit both Google Benchmark JSON and BMF JSON to bencher.dev. Jobs remain non-blocking.

**Blocked by:** 01, 04

**Status:** resolved

- [x] One benchmark job exists per platform under `ci/cscs/`.
- [x] Each job runs `bencher run --adapter cpp_google` and `bencher run --adapter json --file <bmf>`.
- [x] Jobs are gated by `IPPL_RUN_BENCHMARKS` and do not fail the pipeline on alerts.
- [x] Bencher testbed names match the spec (`eiger-openmp-1node`, `daint-gh200-1node`, `mi300-rocm-1node`).
