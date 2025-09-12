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
- `alpine/`, `cosmology/`, `examples/`
    Extra modules. Each is gated by its corresponding option.

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


