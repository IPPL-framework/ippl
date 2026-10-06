# CompareCosmologyForce

## Purpose/Model

Evaluates particle-mesh gravity for imported, frozen particle positions. It
reuses `Simulation::solveForce` in [CosmologySimulation.h](CosmologySimulation.h):
periodic CIC deposition, spectral force calculation, and CIC interpolation.
The force convention is `F0 = -grad(phi0)`, with
`laplacian(phi0) = 1.5 Omega_m delta`; there is no CIC deconvolution.
Particles are not evolved. This adapter supports independent NumPy checks
and matched native FastPM force comparisons.

Source: [tests/CompareCosmologyForce.cpp](tests/CompareCosmologyForce.cpp).
Physics conventions: [LINEAR_PHYSICS.md](LINEAR_PHYSICS.md).

## Build Instructions

Run from the repository root. In the configured local OpenMP build:

```sh
cmake --build build_openmp --target CompareCosmologyForce -j 4
```

For a fresh build, follow the [README build instructions](README.md#build).
The cosmology directory requires MPI, Kokkos, heFFTe, and
`IPPL_ENABLE_COSMOLOGY=ON`, `IPPL_ENABLE_FFT=ON`, and `BUILD_TESTING=ON`.
With `IPPL_USE_STANDARD_FOLDERS=ON`, executables are under `build_openmp/bin`
instead of `build_openmp/demos/cosmology`.

## Run Instructions

The validator generates fixtures and runs the adapter automatically:

```sh
/Users/adelmann/.venv-h6/bin/python -B demos/cosmology/python/validate_frozen_force.py \
  --ippl-exe build_openmp/demos/cosmology/CompareCosmologyForce --quick
```

This checks IPPL against the NumPy oracle. Remove `--quick` for the full suite.
For native FastPM, see [reference build and comparison instructions](README.md#build-and-run-the-native-reference).

Direct invocation, once an input fixture exists:

```sh
env OMP_NUM_THREADS=1 OMP_PROC_BIND=false \
  mpirun -n 1 build_openmp/demos/cosmology/CompareCosmologyForce \
  16 100 0.3 input.csv output-force
```

Arguments are `N L Omega_m input.csv output_dir`: mesh size per dimension,
box length in comoving Mpc/h, matter density parameter, input CSV, and output
directory. The CSV header must be `id,x,y,z,mass`, with exactly `N^3` particles,
unique IDs `0..N^3-1`, and unit masses. Use a fresh output directory.
Outputs include `forces_rankR.csv`, `density_rankR.csv`, and convention metadata.
Use `-n 2`, `-n 3`, or `-n 4` to check other MPI decompositions.
