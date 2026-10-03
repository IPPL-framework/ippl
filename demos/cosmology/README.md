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

## Compare initial conditions with Zarija

The independent reference is the supplied source tree at
`/Users/adelmann/git/zarija-cosmicic-b0e794e34384`. The comparison uses its actual
initializer executable and public cosmology routines in the shared flat,
Gaussian, radiation-free CDM subset. Both implementations receive the same
`hubble`, `Omega_m`, `Omega_bar`, `Sigma_8`, `n_s`, box size, resolution, and initial
redshift. The original reference source and its `cmb.tf` remain unchanged.

### Build and provenance

The reference requires MPI-enabled FFTW3. The build script copies the source into
its build area, downloads the pinned FFTW 3.3.10 release from
[the FFTW project](https://www.fftw.org/download.html), verifies its SHA-256, and
builds static MPI FFTW libraries locally. It performs no global installation and
applies no source patches. From the IPPL repository root, the local command is:

```sh
env ZARIJA_CC=/opt/homebrew/opt/llvm@21/bin/clang \
    ZARIJA_CXX=/opt/homebrew/opt/llvm@21/bin/clang++ \
    ZARIJA_MPICC=/opt/homebrew/bin/mpicc \
    ZARIJA_MPICXX=/opt/homebrew/bin/mpicxx \
    ZARIJA_JOBS=4 \
  bash demos/cosmology/reference/build_zarija.sh \
    /Users/adelmann/git/zarija-cosmicic-b0e794e34384 \
    build_zarija/reference
```

On other machines, select the corresponding compiler and MPI wrappers through
the same environment variables. An existing static MPI FFTW installation can be
selected with `ZARIJA_FFTW_PREFIX`; it must contain `include/fftw3-mpi.h` and the
static `libfftw3_mpi.a` and `libfftw3.a` libraries. Otherwise the script builds
FFTW inside its own output directory.

The durable local executable is
`/Users/adelmann/git/ippl-cosmology-linear/build_zarija/reference/init`.
The C++ build uses `-O2 -std=c++11 -DDOUBLE_REAL -DFFTW3 -DUSENAMESPACE` and
`-DOMPI_OMIT_MPI1_COMPAT_DECLS=0`; the C sources use `-O2 -std=c99 -DDOUBLE_REAL`.
The Open MPI declaration flag exposes retained MPI-1 symbols needed to compile
an unused datatype helper in the original code. It changes neither the original
source nor the initializer's executed physics. `TESTING` and `LONG_INTEGER` are
not enabled.

The build area preserves `source.sha256` and `copied-source.sha256` for all copied
source, header, and supplied input files; the script requires them to match.
`build-manifest.txt` records compiler flags and versions, source/build paths,
FFTW archive and library hashes, the executable hash, and the measured output
ABI. FFTW's configure, build, and install logs remain under `fftw-build/`.
For this local artifact, those historical FFTW logs retain the original temporary
build prefix; the relocated executable was relinked against the local static
archives and has no runtime dependency on that temporary directory. A fresh
invocation of the build script creates a complete build at its requested path.

### Reference output and comparison conventions

Use `PrintFormat=2` for per-rank binary snapshots. In the measured local ABI,
`real` is an 8-byte double, `integer` is 4 bytes, `IDtype` is an 8-byte `long`,
and byte order is little-endian. Every binary record is **52 bytes**:
`x,vx,y,vy,z,vz` as six doubles, followed by four bytes from the particle ID.
For these small runs the stored ID fits in that four-byte field. Serial binary
`PrintFormat=1` has a legacy `long`/`int` MPI ID-transfer mismatch, so it is not
used for multi-rank reference output. ASCII `PrintFormat=0` prints 16 significant
digits with `DOUBLE_REAL`; it is suitable for inspection but binary avoids decimal
round-trip loss.

Reference coordinates are comoving Mpc/h. Inspection of the actual reference
`set_particles` and `grid2phys` implementation shows that its `vZ` output is
`100 dx/d(H0 t)`, despite the generic km/s label in its README. Therefore the
comparison converts it to the IPPL canonical momentum as `p=a² vZ/100`; physical
peculiar velocity is `a vZ`. The reference lattice lies on integer grid sites;
IPPL uses cell centers. Comparisons recover displacements relative to each
lattice and account for that origin difference.

The reference parser stores seeds in a signed 32-bit integer, so matched campaigns
use seeds below `2147483648`. Its legacy random sequence differs from IPPL's
global-Fourier-index random generator: the same numeric seed does not imply the
same phases or pointwise-identical particles. The comparison must distinguish
deterministic background/transfer/normalization checks from statistical IC checks.
The reference's slab decomposition also requires the grid size to be divisible by
its rank count; for power-of-two grids use reference ranks 1, 2, or 4. IPPL's
independent 1–4-rank tests include the non-dividing three-rank case.

The public-API comparison can be run separately from the full initializer:

```sh
/Users/adelmann/.venv-h6/bin/python \
  demos/cosmology/reference/compare_zarija_physics.py \
  --reference-source /Users/adelmann/git/zarija-cosmicic-b0e794e34384 \
  --ippl-build build_openmp \
  --output-dir build_zarija/physics-comparison \
  --np 32 --box-size 168.75
```

Use a fresh output directory. The probe compiles the actual reference
`Cosmology.cpp` and `MT_Random.cpp`, preserves hashes before and after the run, and
writes per-quantity comparison CSVs and `results.json`. It records the known
legacy first-table-row normalization and zero-wave-number differences explicitly;
see [LINEAR_PHYSICS.md](LINEAR_PHYSICS.md) for the IPPL convention. The table
comparison includes both the original supplied file and a separately normalized
input copy; it does not edit the supplied file.

For the full matched-IC campaign, the optional CMake integration runs that public
API probe before the initializer comparisons. Configure the preserved source and
separately built executable explicitly:

```sh
cmake -S . -B build_openmp \
  -DIPPL_COSMOLOGY_PYTHON_VALIDATION=ON \
  -DPython3_EXECUTABLE=/Users/adelmann/.venv-h6/bin/python \
  -DIPPL_COSMOLOGY_ZARIJA_SOURCE=/Users/adelmann/git/zarija-cosmicic-b0e794e34384 \
  -DIPPL_COSMOLOGY_ZARIJA_EXECUTABLE=/Users/adelmann/git/ippl-cosmology-linear/build_zarija/reference/init
cmake --build build_openmp --target cosmology_validate_zarija
```

`cosmology_compare_zarija_physics` runs only the deterministic public-API stage.
The `cosmology_zarija_analysis` CTest checks the comparison analysis on synthetic
data, independently of an external reference executable. Neither an ordinary
build nor CTest downloads or builds the external initializer; it is selected only
through the explicit reference configuration above. Full matched-IC qualification
uses the dedicated build target and its saved outputs.

### Matched-IC protocol

The full campaign uses a 32³ mesh and 32³ particles, box length 168.75 Mpc/h,
`Omega_m=0.31`, `Omega_bar=0.0487`, `hubble=0.675`, `Sigma_8=0.82`, and
`n_s=0.965`. It exercises BBKS and the supplied transfer table at z=49 and 200,
with eight fixed, independent seeds per code and case. These 64 serial runs
are supplemented by 20 MPI runs: IPPL on 2/3/4 ranks and Zarija on 2/4 ranks
for the first seed in each case. The box length makes the original code's
single-precision output conversion factors exactly representable.

The analysis reconstructs the Lagrangian displacement from particle IDs and
wrapped positions, checks its longitudinal character and momentum relation,
and recovers the z=0 density Fourier coefficients. It excludes DC and every
Nyquist plane, where the implementations deliberately differ, and retains one
member of each conjugate pair. Each eight-seed ensemble has 119,160 independent
complex modes. Repeated seeds across redshifts or transfer functions are checked
for deterministic scaling, not counted as additional independent samples.

Power normalization is calculated independently: Gauss quadrature for BBKS and
knot-split Gauss16 integration for the table, not either implementation's
quadrature. Gaussian means, power moments, six radial power bins and inter-seed
correlations use predeclared six-standard-error limits. Reference power checks
also allow a 5e-4 deterministic floor for its coarse normalization quadrature.
The IPPL same-transfer redshift limit is 1e-9; its same-redshift transfer limit
is 1e-8, declared before the campaign because independent table integration and
production log-Simpson normalization differ by 6.34e-9 in power. No amplitude
or normalization is fitted to the generated particles.

The runner can be invoked directly with `--ippl-exe`, `--zarija-exe`, and
`--zarija-source`; `--transfer-file` defaults to `cmb.tf` in the reference source.
`--mpiexec`, repeated `--mpi-arg`, `--numproc-flag`, and `--timeout` select the
launcher and per-run timeout. `--work-dir` must be new or empty; otherwise a
unique `zarija-validation-*` directory is created in the working directory.
`--quick` uses two seeds and 36 runs, and is not the full qualification gate.
Inputs, particle outputs, logs, build ABI, source/executable/table hashes and
every numerical check remain in the results directory. Original source and
artifact hashes are verified again when the campaign finishes.

### Matched-IC result (2026-10-03)

**Passed:** 4,655 deterministic comparison gates and all 1,047 checks across
84 initializer runs. The seven registered cosmology CTests also passed,
including eight analytic self-tests of this comparison harness. No production
cosmology physics or original Zarija source was changed, and no limits were
relaxed after campaign execution.

| Measurement | Result |
| --- | --- |
| Resolved transfer functions, maximum relative difference | 4.94e-16 |
| BBKS / tabulated power, maximum relative difference | 4.76e-7 / 3.26e-5 |
| Growth D / derivative, maximum relative difference | 3.33e-7 / 7.25e-7 |
| Ensemble mean power / independent theory | IPPL 0.995737; Zarija 0.998164–0.998196 |
| Canonical momentum relation, maximum relative L2 error | IPPL 2.00e-12; Zarija 3.92e-7 |
| Longitudinal displacement, maximum relative L2 residual | 2.12e-13 |
| IPPL 1–4-rank position / momentum RMS difference | 6.51e-16 Mpc/h / 1.34e-17 canonical units |

The power ratios are finite-ensemble fluctuations, not fitted amplitudes; both
pass the predeclared statistical limits. Component means, second power moments,
all six power bins, seed correlations, redshift/transfer scaling, and all MPI
checks pass. These tests examine specified Gaussian statistics, not every
possible distributional property. Matching parameters does not produce identical
particles across codes because the random generators differ.

The result is restricted to common interior Fourier modes. Zarija leaves the
first supplied transfer row unnormalized, producing a malformed tiny-k interval;
its normalization quadrature misses that interval. The probe records this
defect separately and also tests a normalized **input copy**. It does not hide
the difference or modify the reference source. Neither that unresolved interval
nor the differing Nyquist treatment is qualified as equivalent.

The complete local evidence is preserved at:

- `build_openmp/demos/cosmology/zarija-physics-hd3av78e/results.json`
- `build_openmp/demos/cosmology/zarija-validation-f2c9zb42/results.json`
- `build_openmp/Testing/Temporary/LastTest.log`

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
