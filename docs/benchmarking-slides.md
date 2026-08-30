---
marp: true
theme: default
paginate: true
title: Continuous Benchmarking for IPPL/OpalX
---

<style>
section {
  font-size: 22px;
}
section h1 {
  font-size: 40px;
}
section h2 {
  font-size: 30px;
}
section table {
  font-size: 18px;
}
section code {
  font-size: 18px;
}
</style>

# Continuous Benchmarking for IPPL/OpalX

**Catch performance regressions before they merge**

Run small, representative benchmarks on every PR across the three CSCS
platforms, submit results to bencher.dev, and alert when a kernel regresses
beyond a statistical threshold.

*Status: implemented on `continuous-benchmarking`; data is flowing from Eiger,
Daint GH200, and Beverin MI300.*

---

## What we built

1. ✅ **Google Benchmark** integrated via CMake `FetchContent`
   (`IPPL_ENABLE_BENCHMARK`, off by default).
2. ✅ **`add_ippl_benchmark()`** helper with `REPORT_TIMERS`/`IGNORE_TIMERS`
   and a compile-time timer whitelist header.
3. ✅ **BMF JSON writer** bridging `IpplTimings` to bencher.dev.
4. ✅ **LandauDampingBench** as the first benchmark.
5. ✅ Three GitLab CI benchmark jobs, gated by `IPPL_RUN_BENCHMARKS=true`.
6. ✅ Results submitted to **bencher.dev** project `ippl-opalx`.
7. ✅ **Non-blocking at first** — collecting data before enabling alerts.

---

## How it fits together

```
  PR opened ─► GitLab CI builds IPPL/OpalX (+Google Benchmark)
                    │
                    ▼
        ctest runs unit/integration tests
                    │
                    ▼
        IPPL_RUN_BENCHMARKS=true?
                    │ yes
                    ▼
        srun LandauDampingBench (all ranks)
                    │
                    ▼
        bencher run --adapter cpp_google   (headline latency)
        bencher run --adapter json --file  (per-phase breakdown)
                    │
                    ▼
        bencher.dev  ─►  plots over time, per testbed & branch
                        ─►  alert on PR if regression > threshold
```

- Benchmark jobs run after the 4-rank release tests.
- Only rank 0 uploads to bencher to avoid duplicate submissions.

---

## First target: LandauDamping

- Already instrumented with `IpplTimings`:
  - `solve` (field solve)
  - `pushVelocity` / `pushPosition` (particle push)
  - `update` (particle redistribution)
  - `loadBalance` (ORB repartition)
  - `particlesCreation`, `dumpData`, `initialize`, `solveWarmup`, `total`
- Google Benchmark measures **per-iteration wall time** natively
  (mean / median / stddev / min / max).
- Existing `IpplTimings` measurements are bridged into bencher so
  **each kernel becomes its own plot** over time.

---

## What you'll see on bencher.dev

Project: `ippl-opalx`

- **Headline plot** (one per testbed):
  `LandauDampingFixture/BM_Advance` — mean latency per time step.
- **Per-kernel plots** (one per IpplTimings phase × testbed):
  - `solve`, `pushVelocity`, `pushPosition`, `update`, `loadBalance`
- Testbeds:
  - `eiger-openmp-1node`
  - `daint-gh200-1node`
  - `mi300-rocm-1node`
- PR branches compared against `main` via a statistical threshold
  (Student's t-test, tunable upper boundary).

---

## Example: real data from Daint GH200

Data is already flowing. Each point is one CI run of `LandauDampingBench` on a
single GH200 node (4 ranks).

- Headline: ~5–20 s per full `advance()` step for 16³ grid, 100k particles, 5
  steps.
- Per-phase breakdown comes from `IpplTimings` via the BMF JSON adapter.

![h:360px Example bencher.dev plot](bencher-plot-1.png)

---

## ❓ Discussion: what should we benchmark?

**Applications**
- **Demos:** LandauDamping, PenningTrap, BumponTailInstability,
  UniformPlasma, electrostaticPIF, cosmology, FEL, collisions.
- **Solvers:** FFT, CG, FEM, multigrid, Maxwell.
- **Low-level:** scatter/gather, particle comms, interpolation, RNG.

**Kernels inside LandauDamping**

| Phase | What it does | Regression risk |
|---|---|---|
| `solve` | Field solve (FFT / CG / FEM) | ★★★ algorithm |
| `pushVelocity` / `pushPosition` | Particle push kernels | ★★ memory bandwidth |
| `update` | Particle redistribution (MPI) | ★★★ communication |
| `loadBalance` | ORB repartition | ★★ algorithm |
| `par2grid` / `grid2par` | Scatter/gather — *not yet timed* | ★★★ bandwidth |

👉 *Which applications and kernels should we track first?*
👉 *Are scatter/gather the missing pieces?*

---

## ❓ Discussion: everything, or a curated subset?

- **Everything:** complete coverage, but long CI runs and noisy data.
- **Curated subset:** fast, clean signal, but blind spots remain.

**Proposed selection criteria:**
1. Exercises a primary solver + its communication pattern.
2. Has a stable, representative problem size.
3. Runs in < 5 minutes in CI.
4. Covers the backends we ship (OpenMP, CUDA, ROCm).

👉 *Do these criteria match your priorities?*
👉 *What balance of coverage vs CI cost is right for IPPL/OpalX?*

---

## Current configuration & open questions

**Current CI problem size:**
- 16³ grid, 100k particles, 5 steps
- `--benchmark_min_time=1.0s`
- Single-node, 4 ranks per platform

👉 *Is this size representative, or should we increase it for better signal?*

**Gating policy:**
- Jobs are enabled by `IPPL_RUN_BENCHMARKS=true` in the CSCS CI admin console.
- Non-blocking while we collect baseline data (a few weeks).
- Flip to `--error-on-alert` once thresholds are tuned.

👉 *When should regressions start blocking PRs?*
👉 *Who decides thresholds — kernel owners or the team collectively?*

---

## Next steps

1. ✅ **Implemented:** CMake option, helper, BMF writer, `LandauDampingBench`,
   three CI jobs, bencher upload.
2. **Seed baselines** on `main` for all three testbeds.
3. **Collect data** for ~2 weeks, non-blocking.
4. **Tune thresholds** per kernel & testbed in the bencher.dev UI.
5. **Flip to blocking** on PRs once thresholds are trustworthy.
6. **Expand** to more applications/kernels (PenningTrap,
   BumponTailInstability, solvers, etc.).

*Note: large-scale benchmarks (>1 node) consume significant node-hours and
should be limited to `main`/nightly/release cycles unless a dedicated budget
is available.*

👉 *Are we aligned? Any blockers or concerns?*
