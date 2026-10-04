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

### Plot saved results

The plotting script reads existing evidence without launching simulations:

```sh
/Users/adelmann/.venv-h6/bin/python -B demos/cosmology/plot_zarija.py \
  --campaign build_openmp/demos/cosmology/zarija-validation-f2c9zb42/results.json \
  --physics build_openmp/demos/cosmology/zarija-physics-hd3av78e/results.json \
  --output-dir build_openmp/demos/cosmology/zarija-plots-final
```

Use a new or empty output directory and a Python environment with Matplotlib,
NumPy and pandas. It produces three PNG/SVG figures, `plot_data.json`, and a
source/output-hash manifest:

- `matched_ic_power`: measured shell power and per-mode residuals against
  independent theory, with Gaussian one-standard-error bars and the saved
  six-standard-error reference acceptance band (including its deterministic
  floor). Raw dimensional shell errors account for variation of P(k) within
  the shell; reconstructed ratios must agree with the saved validation checks.
- `matched_growth`: background D(a) and sub-ppm cross-code differences in D,
  its H0t derivative and f. This is not measured particle evolution.
- `mpi_rank_consistency`: RMS initial phase-space differences relative to each
  saved tolerance, compared with one rank. This is not a performance plot.

Power is reconstructed from Lagrangian IC displacements and extrapolated to
z=0, not measured by Eulerian density deposition. Only the z49 ensemble is shown:
z200 reuses the same phases and is not pooled as extra independent samples.
The mask excludes DC and all Nyquist planes; the largest radial bins contain
cube-corner modes and do not have complete angular coverage. The two transfer
cases also reuse phases, explaining the nearly identical normalized residuals.

## Frozen-force comparison and finite-amplitude pancake

This next stage separates the particle-mesh force operator from time evolution.
`CompareCosmologyForce` imports small, equal-mass fixtures and calls the existing
`Simulation::solveForce()` (CIC scatter, FFT solve, CIC gather). Its host-side CSV
I/O and ID checks are diagnostic only: production fields/particles remain in
their configured Kokkos execution/memory spaces and are distributed by IPPL.
In that frozen-force stage the only production-header addition was the diagnostic
method declaration; no force, initialization, or integration kernel changed. Conservation,
stability, floating-point ordering and production data movement are unchanged.
The adapter additionally copies fields and particles to host for output, so it
is not a performance or exascale benchmark.

### Build and run the native reference

`reference/build_fastpm.sh` builds official FastPM at commit
`15b6c4fd7502a81d99dd13f54fcc9cfa44be1331`, plus pinned GSL/PFFT/FFTW dependencies,
in a separate local directory. It verifies tracked upstream sources are
unchanged and records compiler/dependency/source/executable provenance. Nothing
is fetched during a normal IPPL configure, build, or CTest run. An existing
static double-precision MPI FFTW installation can be reused through
`FASTPM_FFTW_PREFIX`; otherwise the script builds FFTW locally.

```sh
cmake --build build_openmp --target CompareCosmologyForce Cosmology
FASTPM_CC=/opt/homebrew/opt/llvm/bin/clang \
FASTPM_MPICC=/opt/homebrew/bin/mpicc \
FASTPM_FFTW_PREFIX="$PWD/build_zarija/reference/fftw-install" \
  bash demos/cosmology/reference/build_fastpm.sh "$PWD/build_fastpm"

/Users/adelmann/.venv-h6/bin/python -B demos/cosmology/validate_frozen_force.py \
  --ippl-exe build_openmp/demos/cosmology/CompareCosmologyForce \
  --fastpm-exe build_fastpm/FastPMForce \
  --fastpm-manifest build_fastpm/build-manifest.txt --mpi-arg=--oversubscribe
```

The compiler paths above describe the tested macOS environment; choose local
MPI/compiler paths elsewhere. The frozen runner retains a fresh evidence
directory beside the IPPL executable, with input fixtures, logs, per-rank
snapshots, protocol, hashes and `results.json`. `--output` selects another new
or empty directory. Omitting `--fastpm-exe` tests IPPL against NumPy only and
must not be described as a cross-code comparison. `--quick` uses two fixtures
on ranks1/2; full coverage uses eight fixtures on ranks1/2/3/4, including
subcell shifts, axis/oblique deformations, shuffled/wrapped particles, a dense
cluster, two mesh sizes, and two matter densities.

Both executables accept `N L Omega_m input.csv output_dir`. CSV input is
`id,x,y,z,mass`, with exactly N³ unique IDs from0 to N³−1 and mass1. Positions
are comoving Mpc/h. Output is `forces_rankR.csv` (`id,x,y,z,fx,fy,fz`) and
`density_rankR.csv` (`ix,iy,iz,delta,fx,fy,fz`), plus convention metadata.
The independent NumPy oracle constructs CIC weights, deposition, mesh forces
and reciprocal gather without calling either solver.

| Convention | IPPL | Native FastPM diagnostic |
|---|---|---|
| Mesh origin | Cell centres `(i+1/2)L/N` | Node mesh; input shifted by `−L/(2N)` explicitly |
| Compared force | `F0=−grad(phi0)`, `laplacian(phi0)=1.5 Omega_m delta` | Native acceleration multiplied by `−1.5 Omega_m`, as in FastPM's kick |
| Force kernel | Spectral `ik/k²`, CIC twice, no deconvolution | `FASTPM_KERNEL_NAIVE`, CIC twice, no softening/deconvolution |
| Nyquist derivative | Zero differentiated component plane | Only eight self-conjugate corners zeroed; native R2C behavior preserved |
| Precision | Double | Double FFT/CIC/mesh, float32 wavevector tables and particle acceleration |

Because the native Nyquist operators differ, full particle forces are **not
assumed equivalent** for arbitrary fixtures. Tests retain and quantify their
raw difference, compare each full operator to its independent oracle, predict
the native difference, and compare common-band mesh modes. A specially designed
particle fixture deposits just one strictly sub-Nyquist mode and additionally
requires raw particle-force agreement. The diagnostic projection is never
applied inside either native solve. Reference limits account for native float32
tables/acceleration; density uses a separate double-precision limit. All limits
are recorded before execution; there is no fitted force normalization.

The full local frozen campaign passes **64 runs / 790 checks**, with all
source/executable hashes unchanged during execution. Maximum nonzero-fixture
relative RMS errors against each code's own independent oracle are6.75e−14
for IPPL particle forces and3.21e−8 for FastPM particle forces; mesh-force errors
are below6.48e−14. Raw particle-force disagreement is3.87e−8 for the common-band
fixture, but2.47% for the oblique deformation and11.71% for wrapped jitter.
The latter differences are predicted by the preserved native Nyquist operators,
not covered up by a relaxed raw-agreement tolerance. These are operator tests,
not evidence that arbitrary unfiltered native force fields are interchangeable.
Evidence: `build_openmp/demos/cosmology/frozen-force-5_li99qh/results.json`.

### Analytical planar evolution

```sh
cd build_openmp/demos/cosmology
/Users/adelmann/.venv-h6/bin/python -B \
  ../../../demos/cosmology/validate_pancake.py --exe ./Cosmology \
  --mpi-arg=--oversubscribe
```

The exact plane-symmetric growing solution is used **only before shell
crossing**: `x=q−A(a) sin(k.q) k/k²`, with independently computed growth and
canonical `p=a² E(a) f(a) (x−q)`. Final deformation amplitudes0.5 and0.8
correspond to continuum peak density contrasts1 and4. The comparison is of
particle trajectories, not the linear Eulerian density amplitude: nonlinear
Eulerian harmonics are recorded separately.

The full15-run protocol varies N16/32/64, nt16/32/64/128, axis/oblique orientation
and ranks1–4. Timestep convergence uses successive solution differences at
fixed mesh; spatial convergence uses continuum trajectory errors at fixed
nt128. Particles and mesh resolution remain coupled. Quick mode omits both
convergence studies. Limits in `validate_pancake.py` are predeclared engineering
budgets, not a promise of second-order spatial accuracy or local density accuracy.

The first full pancake campaign passes363 of364 checks: all trajectory, MPI,
and time/mesh convergence gates pass, but the N64,Afinal0.8 mass diagnostic
reaches2.072e−12 against its unchanged2e−12 limit. This failure remains recorded.
The uncompensated production mass reduction is a plausible roundoff source;
independent initial/final CIC sums alone do not prove conservation at the
intermediate worst epoch. No physics or tolerance was changed to make it pass.
Local displacement-gradient convergence is also not established: narrowing
grid-scale trajectory defects are consistent with slower-than-second-order
global spatial convergence. These issues require follow-up before declaring
nonlinear evolution qualified.

The final rebuilt-binary campaign reproduces the initial numerical results:
`build_openmp/demos/cosmology/pancake-validation-rs1o9glb/results.json`.
At N64,Afinal0.8, displacement/momentum RMS errors are0.414%/0.736%; measured
spatial orders across the suite are1.43–1.76. All12 cosmology CTests pass,
including the quick pancake subset; **that does not override the retained
failure in the full higher-resolution pancake campaign**.
Reproduce its additional, read-only diagnostic audit with:

```sh
/Users/adelmann/.venv-h6/bin/python -B \
  demos/cosmology/tests/analyze_pancake_diagnostics.py \
  --campaign build_openmp/demos/cosmology/pancake-validation-rs1o9glb \
  --output-dir build_openmp/demos/cosmology/pancake-diagnostics-new
```

The audit records endpoint mass sums, the unsaved intermediate-epoch caveat,
per-grid displacement/Jacobian profiles and the original failed gate. Existing
audit: `build_openmp/demos/cosmology/pancake-diagnostics-rs1o9glb/diagnostics.json`.

With `IPPL_COSMOLOGY_PYTHON_VALIDATION=ON`, CTest includes analytic oracle tests,
adapter valid/invalid-input tests and quick frozen/pancake tests. Full pancake
coverage is `cosmology_validate_pancake`. Set
`IPPL_COSMOLOGY_FASTPM_EXECUTABLE` and optionally
`IPPL_COSMOLOGY_FASTPM_MANIFEST` to enable `cosmology_validate_frozen_force`.
The frozen-force adapter does not advance particles. The separate matched
evolution adapter described below does; these are distinct qualification stages.

The follow-up requirements identified at that stage were: choose a documented common Nyquist treatment (or
explicitly limit which observables are compared); verify the worst-epoch mass
with accurate summation; and investigate grid/lattice-locking by separating
particle and force-mesh resolution or shifting the pancake phase. A native
kernel mismatch must not be interpreted as an integration error, and the
current spatial convergence does not establish nonlinear density convergence.
The user accepted these discrepancies as a local engineering baseline; original
measurements and the failed mass gate remain intact. The subsequent comparison
keeps native operators and explicitly limits its acceptance observables.

### Plot the saved force and pancake evidence

```sh
/Users/adelmann/.venv-h6/bin/python -B demos/cosmology/plot_pm_validation.py \
  --frozen build_openmp/demos/cosmology/frozen-force-5_li99qh/results.json \
  --pancake build_openmp/demos/cosmology/pancake-validation-rs1o9glb/results.json \
  --audit build_openmp/demos/cosmology/pancake-diagnostics-rs1o9glb/diagnostics.json \
  --output-dir build_openmp/demos/cosmology/pm-validation-plots-new
```

Requires Matplotlib, NumPy and pandas. Use a new or empty output directory;
existing final figures are in `build_openmp/demos/cosmology/pm-validation-plots-release`.
The script produces four PNG/SVG figures, `plot_data.json`, and a SHA256 manifest:

- `frozen_force_comparison`: each native operator versus its own independent
  oracle, plus raw/predicted cross-code differences and the residual field.
  Each plotted value is the maximum over ranks1–4. Exactly-zero uniform forces
  have undefined relative errors and are explicitly omitted from the log plot.
- `pancake_convergence`: global continuum trajectory errors versus resolution,
  and successive fixed-mesh timestep differences. Momentum and position use
  their respective analytical RMS scales. Slope guides are not fitted models.
- `pancake_local_errors`: displacement and interval-Jacobian errors along one
  transverse row. The exact map is finite-differenced identically. These are
  not Eulerian CIC density curves and do not establish local-density convergence.
- `pancake_mass_diagnostic`: saved N64 mass-error histories, the unchanged
  acceptance gate and the retained failure. Accurate endpoint sums do not
  verify the unsaved intermediate maximum.

No simulation or gate is changed. This plotting entry point checks the expected
saved campaign structure and retained failure so that these annotations cannot
silently be reused as an all-pass claim. Its extraction self-tests can be run
with `python -B demos/cosmology/tests/test_plot_pm_validation.py`.

## Matched-particle native plain-PM evolution

This stage uses exactly the same imported particle positions and canonical
momenta in IPPL and the pinned native FastPM `FASTPM_FORCE_PM` integrator.
It is not FastPM's modified stepping or COLA. `CompareCosmologyEvolution` calls
the production IPPL force and KDK methods. The KDK body was extracted without
changing its arithmetic. A diagnostic-only particle-count override permits
holding particles fixed while changing the force mesh: deposited cell mass is
normalized by `NM³/NP³` before subtracting one. The default `NP=NM` production
path retains its original subtraction and floating-point operation order.
Fields, scatter/gather and integration stay in their configured Kokkos memory
and execution spaces; import and diagnostic CSV output use host copies.

### Reproduce

First build the pinned frozen-force reference above. The separate evolution
linker verifies its sources and libraries without rebuilding or modifying them:

```sh
cmake --build build_openmp --target CompareCosmologyEvolution Cosmology
bash demos/cosmology/reference/build_fastpm_evolution.sh "$PWD/build_fastpm"
env OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/adelmann/.venv-h6/bin/python -B demos/cosmology/validate_evolution.py \
    --ippl-exe build_openmp/demos/cosmology/CompareCosmologyEvolution \
    --fastpm-exe build_fastpm/evolution/FastPMEvolution \
    --fastpm-manifest build_fastpm/evolution/build-manifest.txt
```

`--quick` selects eight smaller smoke runs, not the full refinement study.
`--output-dir` must be new or empty; otherwise a unique `matched-evolution-*`
directory is created beside the IPPL executable. Launcher options and `--timeout`
follow the other validators. Each invocation preserves the common initial CSVs,
commands, logs, synchronized snapshots, diagnostics, native time-factor probes,
source/input/executable hashes, fixed limits, every check, and `results.json`.
Snapshots are losslessly gzip-compressed after verifying their decompressed
SHA256; the recorded original CSV bytes remain recoverable by decompression.
The runner exits nonzero on a failed gate or incomplete execution.

Optional CMake integration (no automatic reference downloads or builds):

```sh
cmake -S . -B build_openmp \
  -DIPPL_COSMOLOGY_PYTHON_VALIDATION=ON \
  -DPython3_EXECUTABLE=/Users/adelmann/.venv-h6/bin/python \
  -DIPPL_COSMOLOGY_FASTPM_EVOLUTION_EXECUTABLE="$PWD/build_fastpm/evolution/FastPMEvolution" \
  -DIPPL_COSMOLOGY_FASTPM_EVOLUTION_MANIFEST="$PWD/build_fastpm/evolution/build-manifest.txt"
ctest --test-dir build_openmp -L cosmology --output-on-failure
cmake --build build_openmp --target cosmology_validate_evolution
```

The analysis tests and IPPL import regression are available without native
FastPM. Its explicit configuration additionally registers the matched quick
CTest and full build target. Import tests cover 19 MPI runs: exact state import,
ballistic motion on ranks1–4, a nonuniform unequal-mesh one-step CIC/KDK oracle,
production-driver equivalence and malformed/unsupported input rejection.

Both diagnostic executables take
`NP NM L Omega_m a_initial a_final n_steps n_checkpoints input.csv output_dir`.
The shared CSV has columns `id,x,y,z,px,py,pz,mass`, exactly `NP³` unique IDs
`0..NP³−1`, and unit masses. This is a diagnostic import contract, not a new
production initial-condition format or scalable restart interface.

### Physics and comparison contract

The reference retains native CIC, NAIVE spectral forces, no softening, no
deconvolution, float32 particle momentum/acceleration, and double positions/FFT.
Initial momenta are rounded once to float32 and those exact values supplied to
both codes. The explicit origin conversion is `x_native=wrap(x_IPPL−L/(2NM))`.
All exported coordinates, including checkpoint zero, come from actual native
state rebased into IPPL coordinates. The background is radiation-free flat
Lambda-CDM. Canonical momentum is `p=a² dx/d(H0t)` in both implementations.

The reference initializes native PM/store/cosmology directly, bypassing only
the upstream generated-lattice and mesh-divisibility precheck. This permits the
separately tested uneven three-rank decomposition. Force, migration, native
`fastpm_solver_evolve`, kick/drift factors and scheduling are unchanged. Output
uses native synchronized full-step transition events, not interpolated snapshots
or velocity-unit conversions. Both codes use logarithmic full endpoints and a
geometric kick midpoint. Native split drifts are additive counterparts of the
IPPL full drift. Actual native time factors are checked against independent
quadrature, with complete interval and row-count checks.

The full campaign has 32 runs, `NP=32`, `L=168.75 Mpc/h`, `Omega_m=.31`, and
`a=.02→.2`. The pancake has extrapolated final linear amplitude1.5; actual final
particle-sheet folding is required. The second fixture has nine coupled low-k
growing modes with final linear density RMS1. It is a deterministic synthetic
nonlinear test, **not** a sigma8-normalized Gaussian CDM realization. Each code
runs `nt=64/128/256` at `NM=32`, `NM=16/32/64` at `nt=256`, and ranks1–4 at
`NM=32,nt=128`, with nine synchronized checkpoints per run.

Particle differences are periodic and matched by global ID. Density coefficients
are measured directly as `mean(exp(-i k·x))`, independently of either mesh, on
unique conjugate pairs with `0<|k|/k_fundamental<=4`; only active axis modes are
used for the pancake. There is no CIC deconvolution or shot-noise subtraction.
Power, complex-coefficient residuals, and signed cross-correlation are distinct
observables. Raw3D trajectory differences and individual radial shells are
characterized, not gated as though the differing Nyquist operators were equal.

Acceptance limits precede runs and are saved verbatim. Aggregate resolved power
and complex residual limits are 5/2/1% for `NM=16/32/64`; minimum correlations are
.995/.999/.9995. Planar cross-code trajectory and momentum budgets are1e-3 cell
and1e-3 relative RMS. Rank budgets are1e-10 for IPPL (plus a position roundoff
allowance),5e-5 for FastPM. Mean momentum drift is relative to the actual imported
mean, not zero after quantization. Refinement compares successive timestep
differences before and after crossing, and the finest force-mesh pair at fixed
particles. Below an explicit analysis floor no order is inferred; that status
does not demonstrate machine-roundoff dominance. Fixed-particle mesh refinement
does not establish a continuum Vlasov limit.

IPPL's mass diagnostic is a CIC mesh sum; the native adapter's is a unit-particle
count. They are not interchangeable. The earlier strict2e-12 mesh-mass limit is
recorded separately without changing its previous failed result. Neither
post-crossing code agreement nor an aggregate low-k test proves local density
accuracy or an analytical post-crossing solution.

### Local evolution result (2026-10-03)

The full campaign completed all32 runs with **1,723/1,725 checks passing** at
`build_openmp/demos/cosmology/matched-evolution-5v7u3y98/results.json`. No source,
input, executable or limit changed during the campaign. All cross-code, imported
state, rank1–4, native factor, momentum-conservation and sheet-crossing checks
passed. The two failures are the final coupled3D momentum timestep differences:
`nt128→256` gives0.27634% IPPL and0.27556% FastPM versus the unchanged0.2% budget.
Both show difference-reduction factors near4; these are accuracy-budget failures,
not a failure of the measured timestep convergence trend. They remain recorded.

| Measurement across the full campaign | Result |
| --- | --- |
| Pancake cross-code position RMS / mesh spacing | <=1.72e-6 |
| Pancake cross-code momentum relative RMS | <=3.83e-7 |
| Coupled3D aggregate resolved-power difference | <=0.243% |
| Coupled3D complex density-coefficient difference | <=0.338% |
| Coupled3D signed density cross-correlation | >=0.99999449 |
| IPPL rank position / momentum relative differences | <=2.07e-15 cell /1.65e-15 |
| FastPM rank position / momentum relative differences | <=5.95e-8 cell /4.68e-8 |
| Final pancake / coupled3D finest-mesh power changes |3.63% /2.67–2.84% |
| Maximum IPPL CIC mass-sum residual |1.22e-12 |

The raw3D particle differences reach0.0321 mesh cell and1.01% relative momentum;
those are characterized, not accepted as identical-operator trajectories. The
pancake's16→32→64 mesh power changes are not monotone decreasing, despite the
finest-pair sensitivity being below5%. Do not infer continuum spatial convergence.
The historical higher-resolution analytical-pancake mass failure remains intact;
a smaller residual in these new fixed-NP tests does not overturn it.

The independently tested analysis has20 passing tests, the refinement extension
has11 synthetic tests, and the existing full linear22-run/275-check suite was
rerun and passes after the KDK extraction. All16 final registered cosmology
CTests pass. The separate targeted512-step
follow-up passes99/99 checks across two new runs:
`build_openmp/demos/cosmology/evolution-refinement-k2qeaylz/results.json`.
Final256→512 momentum differences are0.06962% IPPL and0.06943% FastPM, below the
unchanged0.2% budget; position differences are0.0004493/0.0004484 mesh cells.
Final difference-reduction ratios are about4, with observed orders1.97–2.01
for the assessed3D quantities. These are successive-resolution differences,
not errors against truth. Pre-crossing momentum differences fall below the
predeclared analysis floor, so no convergence order is inferred for those two
checks. This follow-up does not relabel the original full campaign as all-pass.

Reproduce that bounded follow-up with the saved parent campaign:

```sh
env OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/adelmann/.venv-h6/bin/python -B demos/cosmology/validate_evolution_refinement.py \
    --parent build_openmp/demos/cosmology/matched-evolution-5v7u3y98/results.json
```

This strict evidence extension verifies the original sources, executables,
manifest, exact input CSV and reused snapshots against their hashes. It runs
only the two additional single-rank `NP=NM=32,nt=512` coupled3D cases and applies
the unchanged limits to128/256/512-step differences. It never regenerates the
ICs or overwrites the parent report. `--check-only` performs read-only provenance
verification without simulations. A source change after the parent campaign
deliberately prevents this extension; create a fresh full campaign instead.
The new result preserves the parent's failed status and failures verbatim.

The targeted extension qualifies only the tested NP32/NM32 single-rank finer
stepping. MPI agreement comes from the original128-step rank study, not an
unperformed512-step multi-rank study. The default full-validation target retains
the original64/128/256 protocol and thus still reports those two failed budgets
on this setup; run the explicit follow-up above to inspect the finer-step result.

### Plot the matched evolution evidence

The saved-data plotting script does not run simulations or alter validation gates:

```sh
/Users/adelmann/.venv-h6/bin/python -B demos/cosmology/plot_evolution.py \
  build_openmp/demos/cosmology/matched-evolution-5v7u3y98/results.json \
  --output-dir build_openmp/demos/cosmology/evolution-plots-new
```

It produces PNG/SVG phase portraits at three epochs and a resolved-density
evolution figure, plus `plotted-data.json` and a SHA256 manifest. Requires
Matplotlib, NumPy and pandas. Snapshot hashes are verified; particle IDs are
matched. Phase-space curves connect the same32 particles in one transverse
Lagrangian row, with periodic-edge breaks, not an interpolated distribution.
The density panels use recorded direct particle coefficients and reproduce the
saved comparison norms. Linear residual axes retain true zeros; undefined
normalizations remain gaps. The two original failed timestep checks remain
visible even though they are not cross-code density failures.
Inspected release figures are under
`build_openmp/demos/cosmology/matched-evolution-5v7u3y98/figures-release`.

## Crossed resolution and common-phase Gaussian study

`validate_resolution_study.py` extends the unchanged native plain-PM comparison
with a predeclared 52-run study. It does not change either simulation executable.
Particle and force-mesh sizes are varied independently over32 and64, rather than
only doubling both together. Spatial tests evolve the pancake and coupled3D
fixtures through `a=.02→.2`, compare a fixed physical subcell translation, and
add512/1024/2048-step controls plus a three-rank finest-grid case.

The Gaussian stage uses one mode-keyed realization (seed20261003) sampled at
both particle resolutions, with identical rounded-once momenta supplied to both
codes. Its spherical initial band is `0<|n|<=12`; parameters are
`Omega_m=.31, Omega_bar=.0487, h=.675, n_s=.965, sigma8=.82, L=168.75 Mpc/h`.
Sigma8 normalizes the continuous BBKS spectrum, not the finite realization.
This is flat, radiation-free1LPT with no baryonic transfer features. Evolution
ends at `a=1`, with1024/2048/4096-step controls, starting redshifts49/99, and a
four-rank finest-grid case. Different starts are compared only at their exact
common final epoch using4096 steps, without snapshot interpolation.

Run from the worktree root after building the two evolution executables:

```sh
env OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/adelmann/.venv-h6/bin/python -B demos/cosmology/validate_resolution_study.py \
    --ippl-exe build_openmp/demos/cosmology/CompareCosmologyEvolution \
    --fastpm-exe build_fastpm/evolution/FastPMEvolution \
    --fastpm-manifest build_fastpm/evolution/build-manifest.txt
```

`--smoke` instead runs eight small pipeline checks; it is not a physical
resolution qualification. `--stage spatial` or `--stage gaussian` selects a
standalone stage. `--stop-after N` saves an incomplete batch after N additional
analyzed runs. New output defaults to a unique `resolution-study-*` directory
beside the IPPL executable, not `/tmp`. Completed stage comparisons are saved
before the next stage starts.

Every launch requires a conservative raw-output/archive peak plus at least1GiB
free-space reserve. A disk block returns exit3 with a resumable report; it is
neither a scientific failure nor a pass. Resume after making space with:

```sh
env OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/adelmann/.venv-h6/bin/python -B demos/cosmology/validate_resolution_study.py \
    --resume build_openmp/demos/cosmology/resolution-study-EXISTING
```

Resume verifies frozen source/build/input hashes and completed numerical
archives. Only this study's newly generated CSV snapshots are replaced, after
successful exit and verified archival, by exact uint64/float64 NPZ data. Original
CSV bytes are hashed but are not reconstructible from the numerical archive.
Earlier evidence is never removed. Ambiguous interrupted simulations are
retained and rejected, not silently rerun. This diagnostic storage is not a
production distributed checkpoint/restart system.

Scientific qualification uses direct particle Fourier sums on `0<|n|<=4`,
split into three contiguous radial shells. A shell edge at4.5 does not extend
the measured band beyond4. Fixed128³ interlaced PCS with sinc⁴ window correction
characterizes higher modes through12. Its low-band extraction must independently
agree with direct sums within `max(1e-12,1e-3*norm_direct)`; passing that check
does not qualify the higher band. PCS is analysis only: both forces remain CIC.
The initial CIC diagnostic's failed smoke report and exact sources are retained
under `resolution-study-_ajzgvkl`; improving the measurement did not relax a gate.

Budgets are recorded before execution: paired-code shell power/complex residuals
2% atNM32 and1% atNM64; particle/mesh sensitivity5%; translation power1% and
complex2%; start-redshift power2% and complex3%; finest temporal complex0.2%.
Each also retains its recorded correlation condition. Original import, native
factor, rank, momentum, planar trajectory and temporal controls remain in force.
Qualification reports the largest contiguous passing low-shell prefix and is
vetoed by failed global controls. Completion is separate from acceptance; no
failed budget is hidden or automatically relaxed. One seed and this finite band
cannot establish continuum or halo-statistics accuracy.

Plot a completed stage from the saved report without opening particle archives
or running simulations (Matplotlib is also required):

```sh
/Users/adelmann/.venv-h6/bin/python -B demos/cosmology/plot_resolution_study.py \
  --report build_openmp/demos/cosmology/resolution-study-EXISTING/results.json \
  --output-dir build_openmp/demos/cosmology/resolution-plots-NEW
```

The plotter independently recomputes displayed shell metrics from the recorded
Fourier coefficients and refuses incomplete stages. It produces PNG/SVG control
figures, plotted-data JSON, a byte-exact compressed copy of the input report and
a hash manifest. A finished spatial stage can be plotted while Gaussian runs
continue; the report version used by that figure is preserved. Failed checks
and qualified prefixes remain explicit, and higher-band FFT points are marked
as characterization. These controls are finite-resolution sensitivities, not
errors against truth.

### Spatial result (2026-10-03)

The spatial stage in `resolution-study-cffjva__` completed all 34 runs:
2,083 of 2,213 checks passed. All global import, native-factor, mean-momentum,
timestep and three-rank checks passed. The 130 retained failures are per-shell
controls: 56 particle-resolution, 51 mesh-resolution, 20 translation, and
3 paired-code checks. Both fixtures qualify only their first contiguous shell,
`0<|n|<1.5`, across the entire declared matrix (planar mode1; three-dimensional
integer modes through sqrt(2)). This is not all-band or continuum qualification.
The separately recorded historical2e-12 CIC mesh-mass diagnostic is exceeded
in four pancake runs, with a maximum2.951e-12. The previously accepted engineering
baseline keeps those flags separate from the current qualified-band controls;
neither their failed diagnostic status nor the historical limit is changed.

The crossed controls expose particle sampling and grid alignment that close
code-to-code agreement alone would miss. The maximum shell-power sensitivity
is16.77% for the pancake and19.27% for the coupled3D fixture, versus the5% budget.
The pancake's fixed-NP64 mesh refinement passes all three shells, but fixed-NP32
does not. Its finest-mesh translation fails the final modes3,4 power test at
1.336% versus1%. The coupled3D translation complex residual reaches4.16%
versus2%. The undersampled NP32/NM64 coupled3D cross-code comparison reaches
1.391% complex residual versus1%; the matched NP64/NM64 case stays below0.0025%.
All epochs, including weak early higher harmonics, remain in these maxima.

Time refinement is much tighter: final1024→2048 momentum differences are about
2.3e-6 for the pancake and2.68e-4 for coupled3D, versus the unchanged0.002 limit.
Coupled3D late position/momentum differences exhibit approximately second-order
reduction. Pancake momentum/global-density differences are below the declared
analysis floor; no measured order is claimed for those. Three-rank differences
are below5.9e-15 cell/5.1e-15 relative momentum for IPPL and1.16e-7 cell/8.63e-8
for native plain PM. These are local CPU results, not a performance comparison.

The inspected spatial figure and its exact report snapshot are saved under
`resolution-study-cffjva__/figures-spatial-release`. Gaussian execution is a
separate stage; its status must be read from the current `results.json`, not
inferred from the completed spatial figure.

The original macOS campaign ultimately stopped safely at44/52 runs because the
next launch required1,750,073,344 free bytes but only1,667,796,992 were available.
All34 spatial and10 Gaussian runs remain archived, with410 passing Gaussian
per-run checks. That partial Gaussian report does not establish its comparison
gates or a qualified band. No local simulation is running or awaiting a retry;
the next unstarted local case would be IPPL NP64/NM64,4096 steps,z49,r1.
The fresh Merlin campaign below is the CPU completion route. Its evidence is
separate, not a continuation of the macOS journal, and no earlier data is deleted.
The original peak-space guard and frozen-source requirement remain in force if
the macOS journal is ever resumed.

## Merlin6 CPU continuation

`merlin/cpu_validation.sh` prepares a fresh Linux CPU reference and IPPL build,
runs the cosmology CTests (including ranks 1–4), the eight-run pipeline smoke,
and the complete 18-run Gaussian matrix. This intentionally repeats the ten
Gaussian runs already executed on macOS: compiler, architecture, binaries and
absolute provenance paths differ, so the macOS journal cannot be resumed on
Merlin. The completed local spatial evidence remains separate and unchanged.
The new study uses the same seed, parameters and budgets, but does not assume
cross-platform byte-identical regenerated ICs. No GPU job is part of this script.

Use a new absolute evidence directory whose parent exists. The default mode
requires a four-CPU single-node Slurm allocation and refuses login hosts.
An explicit `--login` mode is available for authorized login-node CPU work:
it requires a `merlin-l-*` host and rejects an inherited Slurm job context.
Both modes refuse existing evidence directories. The script uses one controller,
four build workers, sequential
CTest/MPI runs, and one OpenMP thread per nonlinear rank. The linear regression
also checks one rank with two threads, still within the four-CPU limit.
No GPU is requested, even when CPU work is scheduled on `gwendolen`.

```sh
ssh merlin6
cd /data/user/adelmann/ippl-cosmology-linear
sbatch --clusters=gmerlin6 --account=gwendolen --partition=gwendolen \
  --nodes=1 --ntasks=4 --cpus-per-task=1 --ntasks-per-core=2 \
  --gres-flags=disable-binding --mem=16G --time=06:00:00 \
  --job-name=cosmology-cpu --output=cosmology-cpu-%j.log \
  demos/cosmology/merlin/cpu_validation.sh \
  /data/user/adelmann/cosmology-cpu-NEW
```

On 2026-10-04 the user explicitly authorized login-node CPU work up to eight
ranks while the GPU partition has a hardware problem. This launcher deliberately
keeps the smaller four-rank/four-worker limit. The login build is separate from
any later RHEL9 compute-node build; do not reuse its CMake cache across hosts.
The authorized direct launch is:

```sh
ssh merlin6
cd /data/user/adelmann/ippl-cosmology-linear
bash -l demos/cosmology/merlin/cpu_validation.sh --login \
  /data/user/adelmann/cosmology-cpu-login-NEW
```

Login-mode CPU execution completed successfully on 2026-10-04; the scientific
result and its limits are recorded below and in `COSMOLOGY_STATE.md`. This
permission does not authorize GPU use on the login node. Default Slurm mode
remains available when compute access returns.

The script pins GCC14.3/OpenMPI5.0.10, Kokkos5.2.0, heFFTe v2.4.1 and the
existing unmodified FastPM reference commit. A private Python3.11 environment
uses the same NumPy/pandas/Matplotlib package versions as the local study.
Tracked source hashes, compiler/MPI/package versions, CMake cache, executable
hashes and execution mode/host identity (plus Slurm job identity when applicable)
accompany the existing per-run provenance.
Builds, dependencies and scientific outputs remain on cluster storage.
An exit status of 1 from a completed study retains failed numerical gates;
completion is not synonymous with acceptance. Other setup/runtime failures and
the disk guard also remain explicit. This launcher does not implement automatic
retry or migration of an interrupted run.

After generating remote fixtures, `compare_fixture_files.py original.csv
remote.csv --box-size 168.75 --output NEW.json` records both CSV hashes, sorted
exact values and periodic position/canonical-momentum residuals. It validates
complete IDs and unit masses but applies no physical acceptance tolerance.
Exit zero means comparison completed, not that the inputs are identical.
Optional `--cell-grid N` selects the displacement normalization; otherwise it
is explicitly the particle-lattice spacing, not an assumed force mesh.
Local deployment-helper checks (no Slurm job or simulation is launched):

```sh
python -B demos/cosmology/tests/test_compare_fixture_files.py
python -B demos/cosmology/tests/test_merlin_cpu_launcher.py
```

After all 18 Gaussian runs finish, `merlin/audit_cpu_study.py` checks the
canonical matrix, the exact 1,355-check inventory, comparison/qualification
consistency, metadata, source/build/input hashes and all 162 numerical archives.
Exit zero means the evidence is complete and internally consistent, not that
all scientific gates passed. It preserves failed gates and does not independently
recompute the particle evolution. Its output path must be new.

```sh
python -B demos/cosmology/merlin/audit_cpu_study.py \
  /ABSOLUTE/EVIDENCE/gaussian/results.json \
  --source-dir /ABSOLUTE/FROZEN_CHECKOUT/demos/cosmology \
  --output /ABSOLUTE/EVIDENCE/cpu-completion-audit.json
```

The explicit source directory permits copying the auditor outside a frozen
checkout. Keep the campaign checkout pinned even after completion: its full
source manifest includes documentation, so pulling later handover updates would
invalidate that manifest. Use another worktree for later development/builds.
Focused audit tests: `python -B demos/cosmology/tests/test_cpu_completion_audit.py`.

### Completed Merlin CPU result (2026-10-04)

Evidence: `/data/user/adelmann/cosmology-cpu-login-20261004` on Merlin,
using the frozen checkout at `296fd04c0`. All 21 cosmology CTests passed,
including ranks 1–4; the eight-run pipeline smoke passed all 364 checks.
The fresh Gaussian stage completed 18 runs and 1,355 checks: 1,239 passed,
with 116 retained failures (58 particle-resolution and 58 mesh-resolution).
All integrity, measurement, paired-code, timestep, starting-redshift and
one-versus-four-rank checks passed. The controller's final exit 1 records these
scientific failures, not an execution failure. No numerical tolerance changed.

The full-matrix Gaussian qualification is **0/3 shells**. Even the lowest shell
exceeds the 5% power-sensitivity budget: particle refinement reaches 6.246%, and
mesh refinement 8.137%, at intermediate epochs. Both codes exhibit this behavior;
the lowest-shell complex/correlation controls still pass. Higher-shell maximum
power changes reach 27.91% and 30.61%, respectively. Selecting a better-behaved
resolution pair or only the final epoch would not qualify the declared matrix.

Matched-discretization agreement is much tighter. Across all direct-band paired
checks, the largest power difference is 0.2116%, complex difference 0.8027%, and
minimum correlation 0.9999678; all 243 paired-code gates pass. For NP=NM64,
z49, 2048 steps at a=1, the respective worst shell differences are 0.001208% and
0.003629%, with correlation at least 0.99999999934. These are code differences,
not errors relative to continuum truth.

Starting at z49 versus 99 changes final low-shell power by at most 1.463% and
complex coefficients by 2.649%, within the declared budgets. The largest final
2048→4096 position and momentum differences are 3.971e-6 cells and 2.448e-6
relative; the worst shell temporal complex difference is 1.303e-6. Final z49
position orders are 2.0003 for IPPL and 1.7405 for native plain PM. For the final
z49 momentum/density comparisons, differences are below the predeclared analysis
floor, so no order or machine-precision saturation is inferred for them; z99 has
only two step sizes. One-versus-four-rank position/momentum maxima are 3.14e-15
cells/1.27e-15 relative for IPPL and 2.33e-8 cells/1.85e-8 relative for native
plain PM.

The completion audit passed: 162 numerical archives and 268 file hashes verified,
with the exact check inventory and qualification consistency intact. Report
SHA256: `82cb91aae41986c7655a742e71c59150a175f926db1a5e1d2d2709a9ae8156b2`.
The separate Mac/Linux IC comparison found identical canonical momenta and
position RMS differences of about 1.9e-15 Mpc/h, not byte-identical CSVs.

Plots, their byte-exact compressed report, and the audit are under
`build_openmp/demos/cosmology/merlin-cpu-login-20261004` in the local worktree;
the inspected figure is `figures-gaussian-release/gaussian-controls.png` (also
SVG). This small bundle is about 15 MiB; particle archives and builds stay on
Merlin. The earlier 34-run local spatial result remains separate and unchanged.
This completes the planned CPU study, not Gaussian resolution qualification,
GPU validation, halo-statistics validation or exascale scaling. A larger crossed
particle/mesh matrix is the next accuracy study; A100 execution remains deferred
until the hardware is available.

## Scope of the result

These checks establish local linear-regime behavior and the qualified frozen
force comparisons above, plus matched-particle resolved-observable agreement
through the tested planar shell crossing and synthetic coupled3D evolution.
The finite-amplitude pancake and refinement evidence retain their stated
limitations. CIC suppresses short-wavelength
forces, so agreement with continuum growth is assessed at resolved wavelengths
and through convergence, rather than by requiring all modes to be exact.

This is a starting point for a trusted dark-matter application. It does not yet
validate general nonlinear halo statistics, arbitrary shell crossing, close encounters, 2LPT,
neutrinos, radiation, non-Gaussian initial conditions, or alternative dark energy.
The local OpenMP/MPI result does not establish GPU correctness or performance,
multi-node scaling, restart reliability, or exascale capability. Those require
separate physics comparisons and machine-scale qualification.
