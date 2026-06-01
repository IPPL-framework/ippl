# PR 501 Split Map

Source PR: https://github.com/IPPL-framework/ippl/pull/501

Local mapping branch: `pr-501-map`

Compared against: `origin/master` at `da43fd66a18848b65dcd82af90001f5b64a798ab`

Overall size:

- 41 commits
- 149 files changed
- 23,231 insertions
- 2,205 deletions

## Subsystem Size

| Subsystem | Files | Additions | Deletions |
| --- | ---: | ---: | ---: |
| Interpolation / Autotune | 33 | 8,138 | 20 |
| FFT / NUFFT | 24 | 6,571 | 814 |
| Particle | 21 | 3,841 | 757 |
| Alpine / PIF examples | 12 | 1,923 | 37 |
| Utility | 11 | 1,424 | 45 |
| Communicate | 14 | 626 | 156 |
| Solvers | 6 | 394 | 253 |
| Field / Halo / Layout | 11 | 116 | 87 |
| Build / CMake | 4 | 117 | 12 |
| Types | 2 | 35 | 4 |
| FEM | 3 | 12 | 13 |
| Other | 8 | 34 | 7 |

The review load is dominated by three areas:

- Interpolation / Autotune: new scatter/gather dispatch, binning, tiling, tuning cache.
- FFT / NUFFT: backend/transform refactor plus native NUFFT and pruned transforms.
- Particle: new spatial update path, send/recv serialization hooks, sorting/buffer reuse.

## Commit Shape

The branch starts with well-labeled subsystem commits:

| Commit | Area | Notes |
| --- | --- | --- |
| `7cff674a` | Build / dependencies | Adds `IPPL_ENABLE_FINUFFT`, `IPPL_ENABLE_CUFFTMP`, autotune preset plumbing. |
| `13950c4f` | FFT / NUFFT | Large FFT backend/transform split plus native NUFFT engine. |
| `dbbe59a4` | Interpolation | Scatter/gather redesign, binning, tiling, autotune runtime cache. |
| `2878b90b` | Particle / halo | Rewrites `ParticleSpatialLayout`, adds sort buffers, threads `nghost` through field layout/halo. |
| `9507a796` | Utility | Timing overhaul, `BufferView`, `Tuning`, `ParallelDispatch`, type utilities. |
| `f2820064` | Communicate | Serialization hooks, shared buffer handler. |
| `63d6ccf2` | Integration glue | Field/solver/type updates for new FFT and particle layout APIs. |
| `99d21ba7` | Alpine PIF | Adds ElectrostaticPIF examples. |
| `f0560f76` | Unit tests | Adds NUFFT, interpolation, binning, particle update tests. |
| `927cac7b` | Integration tests | Updates particle benchmark and adds scaling benchmark. |

After that, the branch has many fixup/regression commits. Several are descriptive and likely fold into the earlier subsystem commits; some are separable:

| Commit | Area | Split relevance |
| --- | --- | --- |
| `40438d17` | PCG / preconditioner | Good candidate for an independent PR: hoists per-iteration allocations out of solver loop. |
| `be5f794c` | PCG / PoissonCG | Follow-up to the PCG allocation PR: pass `Field` by reference through `OperatorF`. |
| `95762403` | PIF examples | Consolidates pruned variants into parameterized source. Belongs with PIF examples. |
| `0e7bc58d`, `6c1c48ad` | FFT / NUFFT | FINUFFT guards for CUDA + 2D unit-test build. Belong with NUFFT/FINUFFT PR. |
| `cec1ca61` | Cleanup | Doxygen, formatting, size-type consistency. Should be folded into relevant split PRs or kept as a final cleanup PR. |

## Dependency Observations

### NUFFT Depends On Interpolation

`src/FFT/NUFFT/NativeNUFFT.h` includes and uses:

- `Interpolation/Scatter/ScatterConfig.h`
- `Interpolation/Gather/GatherConfig.h`
- scatter/gather kernels through the new interpolation dispatch

That means native NUFFT cannot be cleanly reviewed before the higher-order scatter/gather layer.

### PIF Depends On NUFFT

The Alpine PIF examples use:

- `ippl::FFT<ippl::NUFFTransform, Field_t>`
- `scatterPIFNUFFT`
- `gatherPIFNUFFT`

So PIF examples should come after the NUFFT transform API.

### Particle Update And Communicate Are Coupled

The particle update rewrite uses:

- `ParticleAttrib::serialize` / `deserialize`
- shared `BufferHandler`
- direct archive serialization
- reusable send/recv buffers

This suggests communication and particle update may be one PR unless a clean lower-level communication PR is extracted first.

### `nghost` / Halo Changes Cross Particle And Field

`ParticleSpatialLayout` changes also thread `nghost` through:

- `FieldLayout`
- `HaloCells`
- field layout APIs

This is small in line count but high in integration risk. It should be explicitly called out in whichever PR contains particle update.

### PCG Allocation Fix Looks Separately Reviewable

The late commits:

- `40438d17`
- `be5f794c`

only touch:

- `src/LinearSolvers/PCG.h`
- `src/LinearSolvers/Preconditioner.h`
- `src/PoissonSolvers/PoissonCG.h`

This can likely become an independent performance/regression PR, separate from PIF.

## Candidate Split

### PR 1: PCG Allocation / Solver Performance

Scope:

- `src/LinearSolvers/PCG.h`
- `src/LinearSolvers/Preconditioner.h`
- `src/PoissonSolvers/PoissonCG.h`

Candidate commits:

- `40438d17`
- `be5f794c`

Reason:

- Smallest coherent split.
- Directly addresses the reported PCG slowdown / repeated `cudaMalloc` concern.
- Does not conceptually depend on PIF, NUFFT, or interpolation.

Risk:

- Need to verify solver convergence and exact iteration behavior.

### PR 2: Communication And Particle Update Infrastructure

Scope:

- `src/Communicate/*`
- `src/Particle/ParticleSpatialLayout*`
- `src/Particle/ParticleAttrib*`
- `src/Particle/ParticleBase*`
- `src/Particle/ParticleSort.h`
- `src/Particle/SortBuffer.h`
- `src/FieldLayout/FieldLayout*`
- `src/Field/HaloCells*`
- particle update tests

Candidate commits:

- `2878b90b`
- `f2820064`
- relevant parts of `63d6ccf2`
- relevant parts of `f0560f76`
- relevant fixup commits after `728af370`

Reason:

- Captures the PIC scaling improvement that is not PIF-specific.
- Reviewable independently from FFT/NUFFT algorithms.

Risk:

- This is still a substantial behavioral change.
- Needs multi-rank CPU/GPU particle update and `ParticleSendRecv` coverage.

### PR 3: Higher-Order Scatter/Gather And Autotune

Status: extracted on local branch `pr501-hosg` from `pr501-communication-particle-update`.

Current validation:

- Serial Debug build: `Binning`, `KernelGatherScatterTest`, and `TestCurrentDeposition` pass through ctest.

Scope:

- `src/Interpolation/*`
- `cmake/AutoTunePresets.cmake`
- `cmake/IpplAutoTunePresets.h.in`
- `cmake/auto_tune/*`
- interpolation tests

Candidate commits:

- `7cff674a` only for autotune preset plumbing, not FINUFFT/CUFFTMP if separable.
- `dbbe59a4`
- interpolation portions of `f0560f76`
- `dacdcd9a` if routing legacy CIC through the new framework is included.

Reason:

- This is the algorithmic layer native NUFFT requires.
- Can be validated without PIF examples.

Risk:

- `dacdcd9a` changes existing CIC scatter/gather behavior by routing legacy paths through the new framework. That may be better as a second PR after the new interpolation layer lands.

### PR 4: FFT Backend / Transform Refactor

Scope:

- `src/FFT/Backend/*`
- `src/FFT/Transform/CC.h`
- `src/FFT/Transform/RC.h`
- `src/FFT/Transform/Trig.h`
- `src/FFT/Transform/PrunedCC.h`
- `src/FFT/Transform/PrunedRC.h`
- `src/FFT/Transform/Common.h`
- `src/FFT/Traits.h`
- existing FFT test updates

Candidate commits:

- FFT refactor parts of `13950c4f`
- FFT portions of `f0560f76`
- FFT portions of later build fixes

Reason:

- Gives reviewers the backend/transform API without also reviewing NUFFT and PIF.

Risk:

- Current commit `13950c4f` mixes backend refactor with native NUFFT files. This split likely requires path-based extraction or commit surgery.

Status: extracted on local branch `pr501-fft` from `pr501-hosg`.

Included:

- FFT backend facade headers under `src/FFT/Backend/`
- transform specializations for CC, RC, pruned CC/RC, and trigonometric transforms
- shared transform helpers in `src/FFT/Transform/Common.h`
- public FFT traits/tags in `src/FFT/Traits.h`
- `src/FFT/FFT.h` converted to the aggregate include for the split transform layer
- removed stale monolithic `src/FFT/FFT.hpp`

Intentionally left out for PR 5:

- `src/FFT/NUFFT/*`
- `src/FFT/Transform/NUFFT.*`
- NUFFT-specific unit tests
- FINUFFT/cuFINUFFT dependency and configure plumbing beyond harmless dormant tags already present in `Traits.h`

Validation:

- Serial Debug configure with FFT/unit tests enabled:
  `cmake -S . -B build-pr501-fft-debug -DCMAKE_BUILD_TYPE=Debug -DIPPL_PLATFORMS=SERIAL -DIPPL_ENABLE_FFT=ON -DIPPL_ENABLE_UNIT_TESTS=ON`
- Build:
  `cmake --build build-pr501-fft-debug --target ippl FFT -j 8`
- Test:
  `ctest --test-dir build-pr501-fft-debug -R '^FFT$' --output-on-failure`
- Result: `1/1` FFT tests passed, `0.81 sec`.

### PR 5: Native NUFFT And FINUFFT/cuFINUFFT Integration

Scope:

- `src/FFT/NUFFT/*`
- `src/FFT/Transform/NUFFT.*`
- FINUFFT/CUFFTMP build switches if not already merged
- NUFFT tests

Candidate commits:

- NUFFT portions of `13950c4f`
- FINUFFT/CUFFTMP portions of `7cff674a`
- NUFFT portions of `f0560f76`
- `0e7bc58d`
- `6c1c48ad`

Reason:

- NUFFT depends on PR 3 scatter/gather and PR 4 transform structure.

Risk:

- Native NUFFT depends on new scatter/gather.
- FINUFFT path has Dim-guard complexity and external dependency complexity.

Status: extracted on local branch `pr501-nufft` from `pr501-fft`.

Included:

- native NUFFT implementation under `src/FFT/NUFFT/`
- `FFT<NUFFTransform, Field>` specialization under `src/FFT/Transform/NUFFT.*`
- aggregate transform include updated to expose `NUFFTransform`
- NUFFT and NUFFT accuracy unit tests
- `IPPL_ENABLE_FINUFFT` and `IPPL_ENABLE_CUFFTMP` configure options and dependency wiring
- adapted the native NUFFT implementation to the PR 3 `Scatter`/`Gather` facades

Validation with FINUFFT disabled:

- Configure:
  `cmake -S . -B build-pr501-nufft-debug -DCMAKE_BUILD_TYPE=Debug -DIPPL_PLATFORMS=SERIAL -DIPPL_ENABLE_FFT=ON -DIPPL_ENABLE_UNIT_TESTS=ON`
- Build:
  `cmake --build build-pr501-nufft-debug --target ippl NUFFT NUFFTAccuracy -j 8`
- Test:
  `ctest --test-dir build-pr501-nufft-debug -R '^NUFFT$|^NUFFTAccuracy$' --output-on-failure`
- Result: `2/2` NUFFT tests passed, total `19.19 sec`.

Still to validate separately:

- `IPPL_ENABLE_FINUFFT=ON` CPU path
- CUDA/cuFINUFFT path
- optional cuFFTMp path with NVSHMEM available

### PR 6: PIF / Alpine Examples

Scope:

- `alpine/ElectrostaticPIF/*`
- Alpine manager wiring
- PIF-specific particle attrib convenience APIs if not included in NUFFT PR

Candidate commits:

- `99d21ba7`
- `95762403`
- PIF/example fixups

Reason:

- Examples should be reviewed after core APIs are accepted.

Risk:

- PIF examples currently depend on the full stack: particle update, scatter/gather, FFT transform refactor, NUFFT.

## Minimal Split Alternative

If six PRs is too much process overhead, a practical minimum is:

1. PCG allocation fix.
2. Particle communication/update infrastructure.
3. Scatter/gather + FFT backend + NUFFT core.
4. Alpine PIF examples and benchmarks.

This is less clean, but still much better than one 23k-line PR.

## Next Mapping Step

Prototype extraction order:

1. Create a branch from `origin/master` for PCG.
2. Cherry-pick `40438d17` and `be5f794c`.
3. Build and run solver tests.
4. If clean, this becomes the first small PR.

Then try the particle/communication split because it is probably the most valuable non-PIF improvement and has direct tests:

- `unit_tests/Particle/ParticleUpdate.cpp`
- `unit_tests/Particle/ParticleUpdateNonuniform.cpp`
- existing `ParticleSendRecv`
- existing `GatherScatterTest`

## Progress

### 2026-05-31: PR 1 Prototype - PCG Allocation / Solver Performance

Status: extracted cleanly.

Worktree:

```text
/private/tmp/ippl-pr501-pcg
```

Branch:

```text
pr501-pcg-split
```

Base:

```text
origin/master @ da43fd66a18848b65dcd82af90001f5b64a798ab
```

Cherry-picked commits:

- `40438d17` -> `a76097f0` (`PCG/Preconditioner: hoist per-iteration allocations out of solve loop`)
- `be5f794c` -> `092a04e5` (`PCG: pass Field by reference through OperatorF`)

Resulting diff:

- `src/LinearSolvers/PCG.h`
- `src/LinearSolvers/Preconditioner.h`
- `src/PoissonSolvers/PoissonCG.h`
- 3 files changed, 388 insertions, 250 deletions

Validation:

```text
cmake -S . -B build-pcg-debug -DCMAKE_BUILD_TYPE=Debug -DIPPL_ENABLE_UNIT_TESTS=ON -DIPPL_ENABLE_FFT=ON -DIPPL_DEFAULT_TEST_PROCS=1
cmake --build build-pcg-debug
ctest --test-dir build-pcg-debug --output-on-failure -O build-pcg-debug/ctest-rank1.log
```

Result:

```text
100% tests passed, 0 tests failed out of 34
Total Test time (real) = 54.17 sec
```

Assessment:

- This is a good standalone first PR.
- It has a narrow file set and clear motivation: remove repeated allocation work from the PCG/preconditioner loop.
- Recommended next validation before opening: run the relevant solver integration tests or CSCS GPU tests that originally showed PCG allocation/slowdown symptoms.

### 2026-05-31: PR 2 Prototype - Communication And Particle Update Infrastructure

Status: extracted on top of `pr501-pcg`.

Branch:

```text
pr501-communication-particle-update
```

Included PR501 areas:

- Communication buffer/archive updates from `9507a796` and `f2820064`
- Particle layout rewrite and sort buffers from `2878b90b`
- Particle update regression tests from the particle portion of `f0560f76`
- Local integration fixes needed to keep this split independent from the later interpolation/FFT/NUFFT PRs

Deliberately excluded:

- Higher-order interpolation/gather/scatter APIs
- FFT backend and NUFFT transform integration
- PIF example/application changes

Validation:

```text
cmake -S . -B build-pr501-communication-debug -DCMAKE_BUILD_TYPE=Debug -DIPPL_ENABLE_UNIT_TESTS=ON -DIPPL_ENABLE_FFT=ON -DIPPL_DEFAULT_TEST_PROCS=1
cmake --build build-pr501-communication-debug --parallel 8
ctest --test-dir build-pr501-communication-debug --output-on-failure
/opt/homebrew/Cellar/open-mpi/5.0.8/bin/mpiexec -n 2 build-pr501-communication-debug/unit_tests/Particle/ParticleSendRecv
/opt/homebrew/Cellar/open-mpi/5.0.8/bin/mpiexec -n 2 build-pr501-communication-debug/unit_tests/Particle/ParticleUpdate
/opt/homebrew/Cellar/open-mpi/5.0.8/bin/mpiexec -n 2 build-pr501-communication-debug/unit_tests/Particle/ParticleUpdateNonuniform
```

Result:

```text
100% tests passed, 0 tests failed out of 36
Total Test time (real) = 38.64 sec

2-rank ParticleSendRecv: passed
2-rank ParticleUpdate: passed
2-rank ParticleUpdateNonuniform: passed
```

Assessment:

- This is a coherent second PR after `pr501-pcg`.
- It is still larger than the PCG split, but it avoids the major algorithmic dependencies from interpolation, FFT, NUFFT, and PIF.
- Recommended next validation before opening: run the same particle tests on CUDA/HIP builds and larger MPI rank counts, especially the original `ParticleSendRecv` failure configurations.
