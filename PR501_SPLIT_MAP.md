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

# LUMI Results 

| Benchmark | Problem size | Nodes | Ranks | master | pr501-pcg | pr501-com | pr501-fft | pr501-hosg | pr501-nufft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FEM | `513_10` | 8 | 64 | 28.23 | 27.77 (-2%) | 27.71 (-2%) | 27.88 (-1%) | 28.23 (0%) | 27.95 (-1%) |
| FFT | `512_10` | 4 | 32 | 4.39 | 4.41 (0%) | 11.54 (+163%) | 11.53 (+162%) | 11.59 (+164%) | 11.53 (+162%) |
| FFT | `512_10` | 16 | 128 | 1.65 | 1.65 (0%) | 2.88 (+74%) | 2.88 (+75%) | 2.93 (+78%) | 2.93 (+78%) |
| PCG | `512_10` | 1 | 8 | 72.31 | 70.01 (-3%) |  | 76.99 (+6%) |  | 76.75 (+6%) |
| PCG | `512_10` | 4 | 32 | 34.60 | 32.93 (-5%) | 43.71 (+26%) | 33.62 (-3%) | 36.32 (+5%) | 33.55 (-3%) |
| PCG | `512_10` | 64 | 512 | 25.00 | 23.85 (-5%) | 22.89 (-8%) | 22.40 (-10%) | 22.74 (-9%) | 22.58 (-10%) |


## Update Timer Investigation

Observed issue:

- On LUMI, the `update` / `updateParticle` timer can increase strongly when moving from `pr501-pcg` to `pr501-communication-particle-update` / `pr501-com`.
- A first code inspection did not find an FFT solver change in this branch step: `pr501-pcg..pr501-communication-particle-update` has no changes under `src/FFT` or `src/PoissonSolvers`.
- The branch step does change particle migration, communication archives, field layout / halo `nghost` plumbing, and timing infrastructure.

Most likely immediate explanation:

- In the communication branch, `ParticleSpatialLayout::update()` posts sends and receives, then performs `MPI_Waitall` plus deferred receive finalization/deserialization inside the outer `updateParticle` timer, but outside the child timers `particleSend`, `particleRecv`, and `sendPreprocess`.
- This can make `updateParticle` look much larger while the child timers look artificially small.
- On the LUMI timing data this pattern is visible: for `PCG_strong_scaling/512_10/nodes_64`, `locateParticles`, `particleSend`, and `particleRecv` are small in `pr501-com`, while the unexplained part of `updateParticle` is large.

Secondary suspect:

- `Archive` contiguous scalar serialization/deserialization changed from `Kokkos::deep_copy` on unmanaged byte views to custom byte-copy kernels plus fences.
- On GPU, especially with GPU-aware MPI, that may be slower or introduce extra synchronization compared with the previous optimized device copy path.
- This would affect particle migration directly, because `ParticleBase::sendToRank()` serializes every active particle attribute before `MPI_Isend`, and receive finalizers deserialize after the wait.

Local Mac OpenMP check:

- Built clean comparison worktrees for:
  - `pr501-pcg`
  - `pr501-communication-particle-update`
- Configuration:
  - `IPPL_PLATFORMS=OPENMP`
  - `IPPL_ENABLE_ALPINE=ON`
  - `IPPL_ENABLE_UNIT_TESTS=ON`
  - `IPPL_ENABLE_FFT=ON`
  - Kokkos `5.0.0`
  - Homebrew LLVM `clang++` and `libomp`
- AppleClang did not find OpenMP automatically; explicit `libomp` include/library paths were required.

Mac OpenMP miniapp runs:

```bash
mpiexec -x OMP_NUM_THREADS=4 -x OMP_PROC_BIND=false -n 2 \
  ./LandauDamping 32 32 32 20000 5 FFT 0.01 LeapFrog --overallocate 2.0 --info 5

mpiexec -x OMP_NUM_THREADS=2 -x OMP_PROC_BIND=false -n 4 \
  ./LandauDamping 32 32 32 20000 5 FFT 0.01 LeapFrog --overallocate 2.0 --info 5
```

Mac OpenMP wall-max timings:

| Ranks | Branch | `update` | `updateParticle` | `locateParticles` | `sendPreprocess` | `particleSend` | `particleRecv` |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | `pr501-pcg` | 0.0911 | 0.1090 | 0.0906 | 0.0187 | 0.00431 | 0.00451 |
| 2 | `pr501-com` | 0.0250 | 0.0266 | 0.00577 | 0.000863 | 0.00150 | 0.000064 |
| 4 | `pr501-pcg` | 0.1550 | 0.1923 | 0.1303 | 0.0391 | 0.0180 | 0.00923 |
| 4 | `pr501-com` | 0.0495 | 0.0534 | 0.00872 | 0.00677 | 0.00291 | 0.000131 |

Conclusion so far:

- The Mac OpenMP run does not reproduce the LUMI slowdown. The communication branch is faster locally for this small CPU/OpenMP test.
- That points away from a generic algorithmic slowdown in the particle update rewrite.
- The remaining likely causes are GPU/MPI-specific behavior on LUMI: GPU-aware MPI wait behavior, archive device-buffer serialization/deserialization, or timer attribution around deferred receive finalization.

Recommended next diagnostic patch:

- Add or move timers around:
  - `MPI_Waitall` as `particleWait`, or charge it back into `particleSend`.
  - deferred receive finalizers as `particleDeserialize`, or charge them into `particleRecv`.
  - optionally `Archive::serialize(hash)` and `Archive::deserialize(offset)` if the wait/deserialization split is still unclear.
- Re-run the LUMI cases before changing algorithms, so the slowdown is attributed to wait time, packing/unpacking, or actual communication.

Diagnostic patch applied locally:

- `ParticleSpatialLayout::update()` now splits the previously hidden tail of `updateParticle` into:
  - `particleWait`: combined send/receive `MPI_Waitall`
  - `particleFreeBuffers`: communicator buffer release
  - `particleDeserialize`: deferred receive finalizers / unpacking into particle attributes
- Expected interpretation on LUMI:
  - If `particleWait` dominates, the issue is likely GPU-aware MPI progress / synchronization or a load imbalance exposed by the nonblocking exchange.
  - If `particleDeserialize` dominates, inspect `Archive::deserialize(offset)` and the custom byte-copy kernels.
  - If neither dominates but `updateParticle` remains high, there is still another untimed section in the particle update path.
