# TestCosmologyPhysics

## Purpose/Model

Checks configuration parsing, flat Lambda-CDM background and growth calculations,
numerical quadrature, transfer functions, and spectrum normalization. Growth is
also compared with an independent ordinary differential equation calculation.
This is a host-side physics sanity test; it does not evolve a distributed particle
simulation or test the particle-mesh force kernels.

Source: [tests/TestCosmologyPhysics.cpp](tests/TestCosmologyPhysics.cpp).
Implementation under test: [CosmologyPhysics.h](CosmologyPhysics.h) and
[CosmologyConfig.h](CosmologyConfig.h).
Equations and assumptions: [LINEAR_PHYSICS.md](LINEAR_PHYSICS.md).

## Build Instructions

Run from the repository root. In the configured local OpenMP build:

```sh
cmake --build build_openmp --target TestCosmologyPhysics -j 4
```

For a fresh build, follow the [README build instructions](README.md#build).
The cosmology directory requires MPI, Kokkos, heFFTe, and
`IPPL_ENABLE_COSMOLOGY=ON`, `IPPL_ENABLE_FFT=ON`, and `BUILD_TESTING=ON`.
With `IPPL_USE_STANDARD_FOLDERS=ON`, executables are under `build_openmp/bin`
instead of `build_openmp/demos/cosmology`.

## Run Instructions

Run directly without arguments or an MPI launcher:

```sh
build_openmp/demos/cosmology/TestCosmologyPhysics
```

Or run its registered CTest:

```sh
ctest --test-dir build_openmp -R '^cosmology_physics$' --output-on-failure
```

Success returns exit code zero and prints a passed message. A failed assertion
returns a nonzero exit code with the failing check. No simulation input file is
required. Distributed force and migration checks are separate:

```sh
ctest --test-dir build_openmp -R '^cosmology_spectral_[1-4]r$' --output-on-failure
```
