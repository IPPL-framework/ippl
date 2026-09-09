## Overview
This guide is written for maintainers of IPPL and should provide an overview of how the build system works.
Since IPPL is designed to run on many different environments from your laptop to large clusters it is important to adhere to a set of rules that are consistent with those requirements.
This file should provide a first entry point for questions related to how the build system is structured as well as provide some information on how some extensions can be made.
A coarse overview on important considerations are:

- **Target Based:** We use target-based CMake and generator expressions to minimize exposed state, control dependency propagation, and avoid hidden global flags leaking into downstream projects.
- **Expose clean downstream usage:** installs with proper `IPPLConfig.cmake`, so projects can just `find_package(IPPL)`.
- **Scoped Modules:** Common functionality is factored into helper files under `cmake/` to avoid duplication and keep responsibilities clear.


## Directory Layout
The repository is organized so that build logic is separated from library code and optional components. Each folder has a clearly defined responsibility:


- `CMakeLists.txt` **(top-level)**
    Declares the project and global options. Includes files inside `cmake/` folder as well as other subdirectories.
- `cmake/` 
    Contains helper modules used across the project:
    - `ProjectSetup.cmake` - global policies and setup
    - `Dependencies.cmake` - adds external packages
    - `CompilerOptions.cmake` - compiler flags, sanitizers, coverage
    - `Platforms.cmake` - platform specific settings
    - `InstallIppl.cmake` - installation/export rules
    - `Version.cmake` - generates version header

- `src/`
    Defines the core target `ippl`. Each subdirectory (e.g. `Field/`, `Particle/`, `FEM/`) is a component that is compiled into this target. Optional solvers and FFT support are gated by build options.
- `unit_tests/`
    GoogleTest-based unit tests added when `IPPL_ENABLE_UNIT_TESTS=ON`
- `test/`
    Integration/system tests, added when `IPPL_ENABLE_TESTS=ON`
- `demos/`
    Demo and application modules (`alpine/`, `cosmology/`, `fel/`, `collisions/`, `electrostaticPIF/`). Each is gated by its corresponding option (`IPPL_ENABLE_ALPINE`, etc.).

## Core library Target (`src/CMakeLists.txt`)
The single library built by this project is `ippl` (alias `ippl::ippl`). All code under `src/` will be part of this one target. Downstream targets (projects that depend on IPPL) link **only** against `ippl::ippl`. One links against the alias since that is read-only and therefore won't accedentally be modified.

### What this CMake does
- **Generate a version header:** `ÌpplVersions.h` from `IpplVersions.h.in`; optional build metadata can be added via `IPPL_EMBED_BUILD_METADATA` (this is off by default to make builds reproducible and therefore cachable).
- **Declare the target:** `add_library(ippl)` + alias `ippl::ippl`.
- **Sets sane per-config flags:** `-O0 -g` (Debug), `-O3 -g` (RelWithDebInfo), `-O3` (Release).
- **Includes and visibility:**
    - Build: headers visible from `src/` via `$<BUILD_INTERFACE>:...>`.
    - Install: headers placed under `${CMAKE_INSTALL_INCLUDEDIR}/ippl` via `$<INSTALL_INTERFACE:...>`
- **Adds internal components:** subdirectories (e.g. `Field/`, `Particle/`, `FEM/`) contribute sources/headers into the `ippl` target. Optional tree `FFT/` is not a separately installed lib but also add to the `ippl` target.
- **Link required dependencies:** `Kokkos::kokkos`, `MPI::MPI_CXX`; **optionally** `Heffte::heffte` are linked against the `ippl` target, so just work for downstream targets.
- **Applies platform knobs:** Includes `cmake/PlatformOptions.cmake` for platform-specific options only visible inside the `ippl` target.
- **Installs/exports:** Include export and installation logic from `cmake/InstallIppl.cmake`.

### How components feed into the `ippl` target
- **Header-only folders:** don't require any additional CMake, they are included via `cmake/InstallIppl.cmake`.
- **Compiled sources:** in a subdirectory should privately link to the `ippl` target, an example for this can be found in `src/Communicate/CMakeLists.txt` 

### Maintainer rules of thumb
- **One library target:** keep adding to the `ippl` target; don't create new installed libs for internal components.
- **Scope settings to the target:** attach features/flags/includes to `ippl`, not globally. (Downstream should inherit everything it needs when linking `ippl::ippl`).
- **Gate optional code with options:** Mirror existing `IPPL_ENABLE_FFT`/`IPPL_ENABLE_SOLVERS` pattern.
- **Install once, centrally:** headers + `ippl` target are installed/exported by `InstallIppl.cmake`; don't add ad-hoc install rules in subdirectories.
- **Top-level toggles:** like `IPPL_ENABLE_FFT` live in the root `CMakeLists.txt`, keep add new options there with clear help strings/descriptions.

## Working with Tests
This project uses CTest with two thin wrappers to register tests:
- **Unit tests** live in `unit_tests/` and are added with the unit-test macro.
- **Integration tests** live in `test/` and are added with the integration-test macro.

> Enabling: Turn tests on at configure time (both OFF my default):
```
cmake -S . -B build \
  -DIPPL_ENABLE_UNIT_TESTS=ON \
  -DIPPL_ENABLE_TESTS=ON
cmake --build build -j
```

### Adding a unit test (pattern)
The following is an example that sets some settings, often just using defaults is enough (see other unit tests and how they were added):
```
# unit_tests/my_feature/CMakeLists.txt

add_ippl_test(
  NAME    my_feature
  NPROC   1
  TIMEOUT 60
  LABELS  unit
  ARGS    --gtest_color=yes
)

```
And a similar pattern is used for integration tests.
More details on options can be found inside `unit_tests/cmake/AddIpplTest.cmake` and `test/cmake/AddIpplIntegrationTest.cmake`.

###  Running tests
Common ways to run tests are:
```
# run everything
ctest --test-dir build -j --output-on-failure

# run only unit or only integration (by label)
ctest --test-dir build -L unit         --output-on-failure
ctest --test-dir build -L integration  --output-on-failure

# run a subset by name (regex)
ctest --test-dir build -R my_feature    --output-on-failure

# rerun only the previously failed tests
ctest --test-dir build --rerun-failed --output-on-failure
```
When developing a test, of course the test executable can also be ran individually without using `ctest`, this will be a very common usecase:
```
mpirun -np 4 ./unit_tests/my_feature/test_my_feature
```

It is considered good practice to run all of the tests at least locally before making a pull request. In any case, the CI will run the tests on different backends.


## Dependencies
IPPL keeps dependency logic **centralized** and **target-based**:
- Prohject-wide dependencies are discovered in `cmake/Dependencies.cmake` and linked into the `ippl` target in `src/CMakeLists.txt`
- **Test-only** dependencies live next to the tests that need them (don't pollute global dependencies). For an example see `test/maxwell/CMakeLists.txt`.

### Adding a project-wide dependency (library code needs it)
1. **Discover or create an imported target** in `cmake/Dependencies.cmake`
    - Prefer `find_package(Pkg CONFIG REQUIRED)` that provides `Pkg::pkg`
    - If no package exists, create a small imported/INTERFACE target that sets include dirs/libs **on the target**, not globally.
2. **Link it into `ippl` in `src/CMakeLists.txt`**
```
target_link_libraries(ippl PUBLIC Pkg::pkg)
```
Now any downstream `target_link_libraries(app PRIVATE ippl::ippl)` inherits what it needs.
3. **Gate optional dependencies** behind an option (OFF by default) and document the flag in the root `CMakeLists.txt`.

### Adding a test-only dependency (example: Maxwell FDTD images)
In `test/maxwell/CMakeLists.txt`, the tests need `stb_image_write.h` **only for those tests**.
1. Acquire the header.
2. Create local targets.
3. Link the test to the local target.

As can be seen in `test/maxwell/CMakeLists.txt`.

### Rules of thumb
- **Prefer imported targets** (`Pkg::pkg`) and attach include/libs to targets, not globally.
- **Keep test-only deps local** dependencies not required by the IPPL source code should not link against the `ippl` target.
- **Install/export only `ippl`**, never locally used imports. (Install is centralized in `InstallIppl.cmake`.)

## Installation and Downstream Usage

### What gets installed
- **Library target:** A single target, `ippl` (alias `ippl::ippl`). It's the only library installed/exported.
- **Headers:** Public headers under `${CMAKE_INSTALL_INCLUDEDIR}/ippl`. Build-tree includes point at `src/`, install-tree includes point at the install include dir via `$<INSTALL_INTERFACE:...>`.
- **Package config:** installed via the centralized rules in `cmake/InstallIppl.cmake`, so downstreams can `find_package(IPPL CONFIG REQUIRED)`.
- **Layout:** Paths follow `GNUInstallDirs` (e.g., `include/`, `lib/`).

### How to install
```
# configure (choose your prefix)
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$PWD/install <other_options>
cmake --build build -j
cmake --install build
```
So just add the `CMAKE_INSTALL_PREFIX` variable to the desired build command and install as shown.

### How downstream projects should use IPPL (recommended)
In the consumer project's `CMakeLists.txt`:
```
# Point CMake to your IPPL install
set(CMAKE_PREFIX_PATH "/path/to/ippl/install")

find_package(IPPL CONFIG REQUIRED)   # finds ippl::ippl

add_executable(app main.cpp)
target_link_libraries(app PRIVATE ippl::ippl)
```



## Kokkos Kernels and host eigenanalysis

`IPPL_ENABLE_KOKKOS_KERNELS` defaults to `OFF`. Set it to `ON` to enable this
support. IPPL first finds an installed Kokkos Kernels package (default minimum
version 5.2.0), then falls back to FetchContent.
`KokkosKernels_VERSION=git.<tag-or-sha>` requests a source build. The package uses
the Kokkos target already selected by IPPL. An external package must have been
built against a compatible Kokkos with the required backends.

Host eigenanalysis is selected independently of `IPPL_PLATFORMS`:

- `IPPL_KOKKOS_KERNELS_HOST=LAPACKE` (default): provide LAPACKE headers and a
  complete LAPACKE/LAPACK/BLAS link line. Set `LAPACKE_ROOT`, or
  `LAPACKE_INCLUDE_DIRS` and `LAPACKE_LIBRARIES`. Static link lines must include
  transitive dependencies, such as the Fortran runtime. Library names can be
  resolved through `LAPACKE_LIBRARY_DIRS`; absolute paths are supported.
- `IPPL_KOKKOS_KERNELS_HOST=MKL`: provide an MKL CMake package through
  `CMAKE_PREFIX_PATH`/`MKL_DIR` (modern oneMKL).
- `IPPL_KOKKOS_KERNELS_HOST=NONE`: use portable/GPU kernels without requesting
  host eigenanalysis; only the GEMM regression is registered in the test binary.
- `IPPL_ENABLE_KOKKOS_KERNELS=OFF` (default): omit the dependency and its tests entirely.

Kokkos does not install these external libraries. If host LAPACKE is not found,
IPPL now downloads reference LAPACK 3.12.1 (SHA256-verified) and builds its BLAS,
LAPACK and LAPACKE libraries. The fallback requires host C and Fortran compilers;
set `CMAKE_Fortran_COMPILER` or `FC` if automatic discovery cannot find gfortran.
The host C compiler is selected independently via `CC` or the fallback toolchain.
It supports LP64 and ILP64 through `IPPL_LAPACK_INTEGER_BYTES` and builds static
position-independent libraries for use by shared or static IPPL.

The first configure builds this dependency in an isolated host project under
`_deps`, before the existing compile/link/runtime probes run. This can take a few
minutes; subsequent configures reuse the build. CUDA/HIP compiler launchers and
IPPL's directory flags are not applied to the host project. Its logs are in
`_deps/ippl-host-lapack-<integer-bytes>-build/{configure,build,install}.log`.
`IPPL_LAPACKE_BUILD_JOBS` controls build parallelism (default 4).
For offline builds, point `FETCHCONTENT_SOURCE_DIR_IPPL_REFERENCE_LAPACK` at an
already unpacked reference LAPACK 3.12.1 source tree.

Set `IPPL_FETCH_LAPACKE=OFF` to require an installed provider. An explicit
`LAPACKE_LIBRARIES` remains authoritative: missing headers or broken linkage
produce errors instead of silently substituting a different provider. A Kernels
package already built without its LAPACKE TPL still needs rebuilding; fetching
LAPACKE cannot enable features in an installed Kernels library.

For a cross build, supply `IPPL_LAPACKE_TOOLCHAIN_FILE` with a C/Fortran toolchain
for the target CPU, or use an installed provider. Runtime checks remain deferred
in cross builds. Fetched libraries, headers, license and the relocatable
`IPPLHostLapack` CMake package are bundled with IPPL's installation; the compatible
Fortran compiler runtime remains a system dependency.

On macOS, an installed OpenBLAS build with LAPACKE or reference LAPACK can also be
used. Apple's Accelerate alone does not supply this LAPACKE interface.
Installation paths belong in site presets/toolchain files, not repository CMake.
For example, source builds can use these preset cache entries:

```json
{
  "IPPL_ENABLE_KOKKOS_KERNELS": "ON",
  "IPPL_KOKKOS_KERNELS_HOST": "LAPACKE",
  "LAPACKE_ROOT": "/site/path/to/lapacke",
  "IPPL_LAPACK_INTEGER_BYTES": "4"
}
```

Use headers and libraries from one compatible provider. Integer size defaults to
4 bytes (LP64); 8 selects ILP64 and propagates `LAPACK_ILP64` or `MKL_ILP64` to IPPL
consumers. For MKL source builds it also selects `MKL_INTERFACE`. Changing this
option cannot convert an installed library to another ABI. Configure checks the
header integer size, links an actual `LAPACKE_dgeev` call, and, for native builds,
runs a small spectrum/status check. Cross builds defer runtime checks to tests.
These checks are smoke tests, not a substitute for a consistent vendor build.
Providers with renamed symbols (for example, `LAPACKE_dgeev64_`) need matching
headers exposing the standard LAPACKE call; integer-size selection alone does
not adapt symbol names. Incompatible installations are rejected by the link probe.

CUDA source builds enable cuBLAS, cuSOLVER and cuSPARSE. HIP source builds enable
rocBLAS, rocSOLVER and rocSPARSE (the sparse libraries are required by the solver
TPL configuration). Supply `CUDAToolkit_ROOT` or ROCm's `CMAKE_PREFIX_PATH` in site
presets. Host LAPACKE/MKL remains necessary when host eigenanalysis is requested.
Installed Kernels configurations are inspected for the requested TPLs; changing
cache options cannot add a missing TPL to an installed package. All link
requirements propagate through `Kokkos::kokkoskernels` and `IPPL::ippl`.

Source builds default to on-demand kernel instantiation rather than default ETI.
`KokkosKernels_ADD_DEFAULT_ETI=ON` can enable upstream's preinstantiations.
Supernodal SPTRSV defaults off because it requires LayoutLeft ETI; applications
needing it should configure the required ETI and enable it explicitly.

The `KokkosKernelsLinearMap` unit test constructs coupled 4x4 maps using GEMM in
the default execution/memory space, explicitly copies results to host memory,
and verifies analytical eigenvalues and normalized right-eigenvector residuals.
It covers stable rotations, conjugate branches, coupled modes, real/complex
instabilities, and near-integer/neutral modes. Eigenvalue/residual tolerance is
1e-12 for small, well-conditioned double-precision maps; GEMM tolerance is 1e-14.
No production physics algorithm or reduction ordering is changed.

The test calls the configured host `LAPACKE_dgeev` directly with column-major
Kokkos host views. Kokkos Kernels 5.2.0's experimental `SerialEigendecomposition`
header contains an invalid Householder template call rejected by GCC 15 and
Clang 21; its
device implementation is unfinished and its host wrapper discards LAPACKE's
status. IPPL does not patch upstream or suppress compiler diagnostics. This test
therefore validates portable map construction and an explicit host eigenanalysis
boundary, not a GPU eigensolver or the experimental wrapper.

```sh
cmake --build build_openmp --target KokkosKernelsLinearMap
OMP_NUM_THREADS=2 ctest --test-dir build_openmp -R '^KokkosKernelsLinearMap$' --output-on-failure
```

Reference: [Kokkos Kernels 5.2.0 eigenanalysis implementation](https://github.com/kokkos/kokkos-kernels/blob/5.2.0/batched/dense/impl/KokkosBatched_Eigendecomposition_Serial_Internal.hpp).

A standalone consumer test checks dependency propagation while linking only
`IPPL::ippl`. Configure it against the IPPL build directory, then repeat with
`IPPL_DIR=<install-prefix>/lib/cmake/ippl` after installation:

```sh
cmake -S cmake/tests/KokkosKernelsConsumer -B build/kernels-consumer -DIPPL_DIR="$PWD/build_openmp"
cmake --build build/kernels-consumer
OMP_NUM_THREADS=2 build/kernels-consumer/consumer
```

Use the same compiler/toolchain as the IPPL build. The CSCS dashboard script
forwards the host-provider, fallback, compiler and dependency-path options above.
Site images can supply LAPACKE or host C/Fortran compilers for the fallback.
