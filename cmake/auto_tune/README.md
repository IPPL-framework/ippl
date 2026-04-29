# Auto-tune presets

Pre-generated scatter / gather sweep CSVs that ship with IPPL. At configure
time, `ippl_configure_autotune_presets()` (in `cmake/AutoTunePresets.cmake`)
picks the subdirectory that matches the current build:

| Build                                                   | Tag       | Lookup directory          |
|---------------------------------------------------------|-----------|---------------------------|
| `IPPL_PLATFORMS=CUDA`, `Kokkos_ARCH_HOPPER90`           | `sm_90`   | `cmake/auto_tune/sm_90/`  |
| `IPPL_PLATFORMS=CUDA`, `Kokkos_ARCH_AMPERE86`           | `sm_86`   | `cmake/auto_tune/sm_86/`  |
| `IPPL_PLATFORMS=HIP`,  `Kokkos_ARCH_AMD_GFX942`         | `gfx942`  | `cmake/auto_tune/gfx942/` |
| `IPPL_PLATFORMS=HIP`,  `Kokkos_ARCH_AMD_GFX90A`         | `gfx90a`  | `cmake/auto_tune/gfx90a/` |
| `IPPL_PLATFORMS=OPENMP` (no GPU)                        | `openmp`  | `cmake/auto_tune/openmp/` |
| Serial only                                             | `serial`  | `cmake/auto_tune/serial/` |

If the matching directory contains `tile_sweep_sa_optimal.csv` and/or
`gather_sweep_optimal.csv`, those files are copied into
`<build>/share/ippl/auto_tune/` and the build-tree path is baked into the
library via the generated header `IpplAutoTunePresets.h`.

At runtime, `TileSizeCache::load()` and `GatherCache::load()` consult the
following sources in order:

1. `IPPL_TILE_CSV` / `IPPL_GATHER_CSV` env var
2. `tile_sweep_sa_optimal.csv` / `gather_sweep_optimal.csv` in cwd
3. The shipped preset for this build's arch (this directory)
4. Built-in defaults seeded by `Ippl::initialize`

To add a new preset:

1. Build IPPL for the target arch.
2. Run any executable with `IPPL_AUTO_TUNE=full` to produce the two CSVs in
   the run directory.
3. Copy them into `cmake/auto_tune/<tag>/` and commit.

Subsequent IPPL builds for that arch will pick up the CSVs automatically;
no env var or extra steps are needed at runtime.
