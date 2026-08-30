# Continuous Benchmarking for IPPL/OpalX

Status: ready-for-agent

## Problem Statement

IPPL/OpalX has no systematic, automated way to detect performance regressions in its solver and particle kernels before code merges. Kernel runtimes are currently observed ad-hoc through `IpplTimings` console output and CSV dumps. Without a continuous benchmarking pipeline, regressions are only noticed late, are hard to attribute to a specific change, and require manual reproduction across the three CSCS target platforms (Eiger OpenMP, Daint GH200 CUDA, MI300 ROCm).

## Solution

Introduce an optional `IPPL_ENABLE_BENCHMARK` CMake path that fetches Google Benchmark and builds dedicated benchmark executables alongside existing demos/tests. The first benchmark will instrument `LandauDampingManager` via the existing `IpplTimings` framework, emit both Google Benchmark JSON and a BMF JSON file, and submit both to bencher.dev from per-platform GitLab CI jobs. The pipeline starts non-blocking and flips to blocking once thresholds are trustworthy.

## User Stories

1. As an IPPL developer, I want benchmark executables to live next to the demos they instrument, so that I can reuse the existing manager classes without modifying the demo binary.
2. As a CI maintainer, I want a single CMake option (`IPPL_ENABLE_BENCHMARK`) to enable the benchmarking build, so that benchmark jobs are opt-in and do not affect normal builds.
3. As a performance engineer, I want Google Benchmark fetched automatically via CMake `FetchContent`, so that no manual dependency installation is required on any platform.
4. As a developer adding a new benchmark, I want an `add_ippl_benchmark()` CMake helper that accepts `REPORT_TIMERS` and `IGNORE_TIMERS`, so that the set of `IpplTimings` phases submitted to bencher.dev is version-controlled and visible next to the target.
5. As a benchmark author, I want the helper to generate a compile-time timer whitelist header, so that the binary only emits the timers I care about with no runtime file I/O or parsing.
6. As a CI runner, I want the benchmark executable to write Google Benchmark JSON and a BMF JSON file, so that bencher.dev can ingest headline latency plus per-phase breakdowns.
7. As a platform maintainer, I want one GitLab CI job per CSCS platform (OpenMP, CUDA, ROCm), so that regressions are caught on every target architecture.
8. As a project lead, I want the CI jobs to be non-blocking initially, so that we can seed baselines and tune statistical thresholds before making them merge-blocking.
9. As a reviewer, I want benchmark results to appear on PRs via bencher.dev, so that performance impact is visible alongside code changes.
10. As a developer debugging a regression, I want the benchmark to still print `IpplTimings` output and CSV dumps, so that I can investigate locally with familiar tooling.
11. As a maintainer of a non-CSCS fork, I want the benchmarking code to be self-contained in IPPL and not require bencher.dev credentials to build or run, so that the feature is portable.
12. As a future benchmark author, I want the first benchmark (`LandauDampingBench`) to be a clear template, so that adding `PenningTrapBench` or `BumponTailInstabilityBench` is mostly copy-paste.
13. As a CI operator, I want only rank 0 to emit JSON output, so that multi-rank runs do not produce corrupt or duplicate benchmark artifacts.
14. As a performance engineer, I want benchmark iterations to call `IpplTimings::resetAllTimers()` before the measured loop, so that cumulative timer state does not leak across Google Benchmark iterations.
15. As a project lead, I want the bencher.dev project name and testbed names documented in the spec, so that CI configuration is reproducible.

## Implementation Decisions

- **CMake option**: Add `option(IPPL_ENABLE_BENCHMARK "Enable Google Benchmark-based benchmarks" OFF)` in the root `CMakeLists.txt`.
- **Dependency fetching**: Google Benchmark v1.9.4 is fetched via `FetchContent_Declare` in `cmake/Dependencies.cmake`, mirroring the existing GTest fetch. Benchmark tests and installation are disabled.
- **Benchmark helper**: A new CMake module `cmake/AddIpplBenchmark.cmake` defines `add_ippl_benchmark(<name> SOURCES ... [REPORT_TIMERS ...] [IGNORE_TIMERS ...])`.
  - If neither `REPORT_TIMERS` nor `IGNORE_TIMERS` is given, all `IpplTimings` phases are reportable.
  - `REPORT_TIMERS` is a whitelist; `IGNORE_TIMERS` is a blacklist; supplying both is an error.
  - The helper generates a header `<name>_timers.h` in the build tree containing a `constexpr std::array<std::string_view, N> reportedTimers`.
- **BMF writer**: A small header-only utility (e.g. `src/Utility/BenchmarkMetrics.h`) collects timer statistics from `IpplTimings` and writes BMF JSON. It filters emitted timers against the generated whitelist.
- **First benchmark**: `demos/alpine/LandauDampingBench.cpp` builds a `LandauDampingManager<>` in `SetUp()`, runs `pre_run()` once as warmup, resets timers, then benchmarks `manager->advance()` per Google Benchmark iteration. `TearDown()` prints timers and writes BMF JSON.
- **JSON emission**: Only rank 0 uses the real `benchmark::BenchmarkReporter`; other ranks use `benchmark::NullReporter`.
- **CI integration**: Extend `ci/cscs/` with a benchmark stage/job per platform. Jobs run only when `IPPL_RUN_BENCHMARKS` is `"true"` and execute two `bencher run` submissions:
  1. `--adapter cpp_google` against the Google Benchmark JSON.
  2. `--adapter json --file timings.bmf.json` for per-phase metrics.
  - The variable defaults to `"false"` in `ci/cscs/common.yml` so normal build/test pipelines are unaffected.
  - Set it to `"true"` via the GitLab pipeline trigger variables or the CSCS CI admin console to enable benchmarks.
- **Bencher project**: Target project is `ippl/OpalX` (name TBD with team); testbeds are `eiger-openmp-1node`, `daint-gh200-1node`, `mi300-rocm-1node`.
- **CI blocking policy**: Jobs set `IPPL_BENCH_ERROR_ON_ALERT=false` (or equivalent non-blocking configuration) until baselines are seeded and thresholds tuned.
- **Warmup semantics**: `SetUp()` calls `pre_run()` once and `IpplTimings::resetAllTimers()` so that initialization and solver warmup are not included in the measured `advance()` loop.
- **Benchmark arguments**: Problem size defaults are configurable via Google Benchmark `->Args(...)` or CLI args; the CI default is left as a spec decision (initially `16 16 16` with `10000000` particles and a small number of steps to keep iteration time reasonable).

## Testing Decisions

- **Seam 1 — CMake helper**: Verify that `add_ippl_benchmark(LandauDampingBench ... REPORT_TIMERS solve pushVelocity ...)` configures successfully and that the generated `<name>_timers.h` contains exactly the requested timer names. This is tested by a configure-and-build check, not a runtime unit test.
- **Seam 2 — BMF writer**: If the BMF writer is extracted into a small pure function (string list + timer statistics → JSON string), it can be covered by a lightweight unit test that asserts correct JSON shape and whitelist filtering. Otherwise it is tested indirectly via the benchmark executable.
- **Seam 3 — First benchmark executable**: Run `LandauDampingBench` locally (single-rank and multi-rank) and assert that:
  - It exits 0.
  - Google Benchmark JSON is produced and contains the expected benchmark name.
  - BMF JSON is produced and contains only whitelisted timer names.
  - Only rank 0 writes JSON artifacts.
- **Seam 4 — CI integration**: A dry-run on one platform (e.g. OpenMP) confirms that `bencher run` is invoked with the correct adapters and that the job is non-blocking.
- **Regression policy**: No runtime assertions on performance numbers are added to the test suite; bencher.dev owns regression detection.

## Out of Scope

- Modifying existing demo/test binaries to act as benchmarks (the plan explicitly creates separate benchmark executables).
- Large-scale (>1 node) benchmarks for this first effort; the testbeds remain single-node.
- OpalX-specific benchmark integration or regression-test coupling; this effort is scoped to IPPL demos only.
- Bencher.dev dashboard configuration beyond project/testbed naming; the skill assumes the project exists or will be created separately.
- Runtime JSON config for the timer whitelist; the generated-header approach is chosen for simplicity.
- Changing `IpplTimings` internals; only the existing public interface is consumed.

## Further Notes

- The generated timer whitelist header should be placed in the build tree next to the benchmark binary (e.g. `${CMAKE_CURRENT_BINARY_DIR}/<name>_timers.h`) and included privately by the benchmark source.
- Keep benchmark binaries out of the default `IPPL_ENABLE_TESTS` path so that `ctest` does not try to run Google Benchmark executables as Catch/GTest-style tests.
- The LandauDamping integration test in `demos/alpine/CMakeLists.txt` already passes arguments `"16" "16" "16" "10000000" "25" "FFT" "0.01" "LeapFrog" "--overallocate" "2.0" "--info" "10"`; these are a reasonable starting point for the benchmark, though step count and overallocate factor may be tuned for stable iteration time.
- When `IPPL_ENABLE_BENCHMARK` is OFF, no Google Benchmark code or headers should be compiled or linked.
