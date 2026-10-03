# Local linear cosmology model

`Cosmology` generates initial conditions and evolves a periodic cold-dark-matter
particle distribution using IPPL's distributed particle and field containers,
CIC assignment/interpolation, heFFTe transforms, and kick-drift-kick integration.
The validated local setup runs on **1–4 MPI ranks**, with an
OpenMP host backend. It is separate from the historical `StructureFormation`
demo and does not require that demo's `Data.csv` input.

The supported model is flat, radiation-free Lambda-CDM with one collisionless
matter species. Initial conditions use first-order Lagrangian perturbation theory
(1LPT/Zel'dovich), with a Gaussian spectrum, a single density mode, or a uniform
lattice. The background, BBKS spectrum, CMBFAST table conventions, and sigma8
normalization follow the supplied Zarija Lukic initializer. See
[the physics and data contract](LINEAR_PHYSICS.md) for the equations and explicit
differences from the legacy implementation.

## Build

Requirements are CMake 3.24 or newer, a C++20 compiler, MPI, and an OpenMP runtime.
IPPL discovers or fetches Kokkos and heFFTe through its usual CMake dependency
configuration. The local build uses Kokkos 5.2.0 and heFFTe v2.4.1 with the CPU
backend; CUDA is not required. Configure from the repository root:

```sh
cmake -S . -B build_openmp \
  -DCMAKE_BUILD_TYPE=Release \
  -DIPPL_PLATFORMS=OPENMP \
  -DIPPL_ENABLE_COSMOLOGY=ON \
  -DIPPL_ENABLE_FFT=ON \
  -DIPPL_ENABLE_SOLVERS=ON \
  -DBUILD_TESTING=ON \
  -DKokkos_VERSION=git.5.2.0 \
  -DHeffte_VERSION=git.v2.4.1
cmake --build build_openmp --target Cosmology TestCosmologyPhysics -j 4
```

`IPPL_ENABLE_SOLVERS` also supports the existing cosmology demo built from this
directory. The executables are `build_openmp/demos/cosmology/Cosmology` and
`build_openmp/demos/cosmology/TestCosmologyPhysics`; with
`IPPL_USE_STANDARD_FOLDERS=ON`, look under `build_openmp/bin` instead.

For the existing local worktree on branch `codex/cosmology-linear`, the executable
is `/Users/adelmann/git/ippl-cosmology-linear/build_openmp/demos/cosmology/Cosmology`.
Its cache already selects Homebrew LLVM 21, Open MPI, and libomp. Rebuilding this
configured directory only needs the `cmake --build` command above. To reproduce
that toolchain in a **new** build directory, append the following configuration
arguments to the configure command:

```sh
-DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@21/bin/clang++ \
-DMPI_CXX_COMPILER=/opt/homebrew/bin/mpicxx \
-DOpenMP_ROOT=/opt/homebrew/opt/libomp \
-DOpenMP_CXX_FLAGS=-fopenmp=libomp \
-DOpenMP_CXX_INCLUDE_DIR=/opt/homebrew/opt/libomp/include \
-DOpenMP_CXX_LIB_NAMES=libomp \
-DOpenMP_libomp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib
```

Existing downloaded sources can be reused without fetching them again by adding
`-DFETCHCONTENT_SOURCE_DIR_KOKKOS=/Users/adelmann/git/ippl/build_openmp/_deps/kokkos-src`
and
`-DFETCHCONTENT_SOURCE_DIR_HEFFTE=/Users/adelmann/git/ippl/build_openmp/_deps/heffte-src`.
These compiler and source paths are specific to this local machine; use the normal
compiler, dependency prefixes, and MPI launcher on other systems.

## Run on 1–4 ranks

The only application argument is the parameter file:

```sh
env OMP_NUM_THREADS=1 OMP_PROC_BIND=false \
  mpirun -n 1 build_openmp/demos/cosmology/Cosmology \
  demos/cosmology/input/linear-sine.par
```

The supplied input files are:

| Input | Initial condition | Mesh/particles | Evolution |
| --- | --- | --- | --- |
| [linear-sine.par](input/linear-sine.par) | Axis mode, initial density amplitude 0.001 | 32³ each | z=49 to 9, 100 steps |
| [linear-gaussian.par](input/linear-gaussian.par) | Gaussian BBKS, sigma8=0.8, fixed 64-bit seed | 32³ each | z=49 to 9, 100 steps |
| [linear-uniform.par](input/linear-uniform.par) | Uniform lattice, zero peculiar momentum | 16³ each | z=49 to 9, 10 steps |

`output` names a directory relative to the **run's working directory**, so use a
different working directory for each rank count. This example runs all four rank
counts sequentially without changing the input file or overwriting earlier runs:

```sh
# Run from the repository root.
cosmoSrc="$(pwd)"
cosmoRuns="$(mktemp -d /tmp/ippl-cosmology-manual.XXXXXX)"
for ranks in 1 2 3 4; do
  mkdir "$cosmoRuns/r$ranks"
  (
    cd "$cosmoRuns/r$ranks" &&
    env OMP_NUM_THREADS=1 OMP_PROC_BIND=false \
      mpirun -n "$ranks" \
      "$cosmoSrc/build_openmp/demos/cosmology/Cosmology" \
      "$cosmoSrc/demos/cosmology/input/linear-sine.par"
  ) || break
done
```

Each run writes into its own `cosmology-sine-output` directory. Change the last
argument to the Gaussian or uniform example for those cases. Use the same seed
and configuration when comparing rank counts. One OpenMP thread per rank avoids
oversubscribing a small machine; independent validation also exercises two
threads. If Open MPI reports insufficient slots, use its `--oversubscribe` option
for these small local tests.

## Inputs, units, and output

Inputs are `name=value` assignments with `#` or `//` comments. Missing values take
the defaults in [CosmologyConfig.h](CosmologyConfig.h); unknown or duplicate
parameters and unsupported physics are errors. `np` is the grid size **per
dimension**, and creates exactly `np³` particles; it must be even and at least
four. `nt` is the number of logarithmically spaced scale-factor steps. The seed
accepts an unsigned 64-bit integer.

`box_size` and particle positions are comoving Mpc/h. Particle snapshot fields
`px,py,pz` store canonical momentum `p=a² dx/d(H0 t)`, not velocity in km/s. The
physical peculiar velocity is `v_pec=100 p/a` km/s. The growth function is normalized
to `D(1)=1`; sigma8 specifies the z=0 linear spectrum. For `ic_mode=sine`,
`amplitude` instead specifies the density contrast at `z_in`, with integer
`mode_x`, `mode_y`, and `mode_z` selecting the periodic wave. Each component must
lie strictly below the mesh Nyquist frequency.

`TFFlag=4` selects the BBKS analytic transfer function. `TFFlag=0` also requires
`transfer_file`, whose first three columns are `k`, `T_cdm`, and `T_baryon`; a
relative path is resolved from the input file's directory. The supplied external
reference table is
`/Users/adelmann/git/zarija-cosmicic-b0e794e34384/cmb.tf`. Its cosmology must match
the parameters used in a physical run. The transfer-table convention, range
checks, and finite normalization cutoff are documented in
[LINEAR_PHYSICS.md](LINEAR_PHYSICS.md).

The run produces:

- `diagnostics.csv`: scale factor, growth factor/rate, exact particle count, mass
  error, density and phase-space RMS values, and a measured density-mode amplitude
  with its linear expectation. `diagnostics_every` controls its cadence.
- `metadata.txt`: the model conventions, configuration, rank/thread counts, and
  timing information.
- `particles_initial_rankN.csv` and `particles_final_rankN.csv`: one file per
  rank and epoch, with columns `id,x,y,z,px,py,pz`. Sort by global `id` when comparing
  outputs from different rank counts, since particle migration changes ownership.

`write_particles=false` disables particle CSV output. These snapshots are intended
for small diagnostic runs; they are not a scalable checkpoint/restart format.
Use a fresh output location for every run: the program rejects existing simulation
output to avoid overwriting it.

## Validate

The compiled tests cover configuration, background/growth, transfer functions,
and a manufactured spectral force plus forced particle migration on 1, 2, 3,
and 4 MPI ranks:

```sh
ctest --test-dir build_openmp -L cosmology --output-on-failure
```

To include the quick evolution suite in CTest and enable the full-validation
build target, configure with `-DIPPL_COSMOLOGY_PYTHON_VALIDATION=ON` and
`-DPython3_EXECUTABLE=/Users/adelmann/.venv-h6/bin/python` (or another environment
with NumPy and pandas). This is enabled in the local build. Run the full suite
with `cmake --build build_openmp --target cosmology_validate`. Both preserve
uniquely named result directories under the build tree. MPI launcher options
can be supplied at configure time, for example `-DMPIEXEC_PREFLAGS=--oversubscribe`.

The spectral test can also be run directly:

```sh
env OMP_NUM_THREADS=1 OMP_PROC_BIND=false \
  mpirun -n 4 build_openmp/demos/cosmology/Cosmology --self-test
```

The full evolution validation requires Python 3, NumPy, and pandas. On this local
machine the environment is `/Users/adelmann/.venv-h6/bin/python`. From the
repository root:

```sh
cosmoValidation="$(mktemp -d /tmp/ippl-cosmology-validation.XXXXXX)"
/Users/adelmann/.venv-h6/bin/python demos/cosmology/validate_linear.py \
  --exe build_openmp/demos/cosmology/Cosmology \
  --work-dir "$cosmoValidation"
```

Use `--mpiexec /path/to/mpirun` to select a launcher, and repeat `--mpi-arg=VALUE`
for launcher options, for example `--mpi-arg=--oversubscribe`. For a launcher with
a different process-count flag, use `--numproc-flag=-np`. `--timeout` sets a
per-simulation time limit in seconds. `--quick` retains all four MPI rank counts
but uses a smaller Gaussian mesh and omits convergence studies; it is not the
full validation gate. `--self-test` checks the independent growth oracle without
running the simulation executable.

The full suite checks zero-force uniform evolution; axis and oblique linear
modes; a late-time Lambda-dominated case; Gaussian IC Fourier normalization and
momentum; conservation and particle integrity; rank/thread agreement; and time
and mesh convergence. Gaussian validation uses a reduced sigma8 to isolate the
linear regime. Its density-mode measurements come from particle snapshots,
independently of the simulation's mesh diagnostics. Numerical tolerances are
declared in the script before any simulation runs.

Every validation invocation needs a new results directory. It preserves the
generated inputs, per-case logs and snapshots, and machine-readable `results.json`.
Omitting `--work-dir` creates a uniquely named directory in the current directory.
Within each case directory, `input.par` and `run.log` sit alongside an `output/`
subdirectory containing the simulation's diagnostics, metadata, and snapshots.
Success returns zero; a failed check or simulation returns nonzero. Inspect
`results.json` and the affected case's `run.log` to locate failures.

### Local qualification (2026-10-03)

The OpenMP build passed all six registered CTests and the full 22-simulation,
275-check evolution suite. No acceptance tolerances were relaxed after running.

| Measurement | Result |
| --- | --- |
| Manufactured force, ranks 1–4 | Maximum relative error about 2e-15 |
| Forced migration, ranks 1–4 | Correct ownership, all IDs/attributes preserved, exact endpoints handled |
| Independent Gaussian IC coefficients | Relative L2 error 4.8e-12 |
| Gaussian low-k linear growth, 32³ | Weighted complex error 0.333% |
| Late-Lambda sine growth, z=9 to 0 | Error 0.943% |
| Time convergence: positions / momenta | Observed orders 1.988 / 1.995 |
| Mesh refinement: 16³→32³→64³ | Growth-error reductions 3.899× / 3.849× |

Rank/thread phase-space comparisons and mass/particle checks passed their
predeclared tolerances. The unchanged shipped sine input also completed on all
four rank counts (z=49 to 9, growth error about 0.628%); the shipped sigma8=0.8
Gaussian input completed a four-rank smoke test. That smoke test is not a separate
nonlinear qualification. Exact result locations are recorded in the repository's
`COSMOLOGY_STATE.md`; regenerating the suite creates a fresh result directory.

## Scope of the result

These checks establish local linear-regime behavior, including the expected
finite-resolution particle-mesh force error. CIC suppresses short-wavelength
forces, so agreement with continuum growth is assessed at resolved wavelengths
and through convergence, rather than by requiring all modes to be exact.

This is a starting point for a trusted dark-matter application. It does not yet
validate nonlinear halo statistics, shell crossing, close encounters, 2LPT,
neutrinos, radiation, non-Gaussian initial conditions, or alternative dark energy.
The local OpenMP/MPI result does not establish GPU correctness or performance,
multi-node scaling, restart reliability, or exascale capability. Those require
separate physics comparisons and machine-scale qualification.
