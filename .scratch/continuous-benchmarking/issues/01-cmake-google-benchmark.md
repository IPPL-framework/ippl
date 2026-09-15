# 01: Add `IPPL_ENABLE_BENCHMARK` CMake option and fetch Google Benchmark

**What to build:** A CMake-controlled path that, when enabled, fetches Google Benchmark v1.9.4 and makes it available to targets. A minimal smoke benchmark executable proves the whole path works: configure with `IPPL_ENABLE_BENCHMARK=ON`, build, run, and see Google Benchmark JSON output.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [ ] `option(IPPL_ENABLE_BENCHMARK ... OFF)` exists in root `CMakeLists.txt`.
- [ ] Google Benchmark v1.9.4 is fetched via `FetchContent` when the option is ON.
- [ ] A minimal `BenchmarkSmoke` executable builds and runs under `ctest`/manually.
- [ ] When the option is OFF, no benchmark code is compiled or linked.
