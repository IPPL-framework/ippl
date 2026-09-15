# 04: Implement `LandauDampingBench`

**What to build:** The first real benchmark executable for the Alpine module, instrumenting `LandauDampingManager`. It uses `add_ippl_benchmark()` and the BMF writer, runs `pre_run()` once as warmup, then benchmarks `advance()` per Google Benchmark iteration.

**Blocked by:** 03

**Status:** resolved

- [ ] `LandauDampingBench` builds with `IPPL_ENABLE_BENCHMARK=ON`.
- [ ] Running it produces Google Benchmark JSON and BMF JSON.
- [ ] Only rank 0 emits JSON artifacts.
- [ ] `IpplTimings::resetAllTimers()` is called before the measured loop.
- [ ] The benchmark still prints `IpplTimings` output for local debugging.
