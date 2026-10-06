# CompareCosmologyEvolution

## Purpose/Model

Evolves imported particle positions and canonical momenta using the production
`Simulation` force, kick, drift, and migration routines. The model is periodic,
flat, radiation-free Lambda-CDM with collisionless matter, CIC particle-mesh
gravity, and kick-drift-kick stepping with endpoints uniform in `log(a)`.
Particle count and mesh resolution can vary independently for refinement studies.
This adapter supplies matched initial states for native plain-PM FastPM comparisons;
it does not generate initial conditions.

Source: [tests/CompareCosmologyEvolution.cpp](tests/CompareCosmologyEvolution.cpp).
Shared implementation: [CosmologySimulation.h](CosmologySimulation.h).
Equations and units: [LINEAR_PHYSICS.md](LINEAR_PHYSICS.md).
The diagnostic CSV import is not a scalable restart format.

## Build Instructions

Run from the repository root. In the configured local OpenMP build:

```sh
cmake --build build_openmp --target CompareCosmologyEvolution -j 4
```

For a fresh build, follow the [README build instructions](README.md#build).
The cosmology directory requires MPI, Kokkos, heFFTe, and
`IPPL_ENABLE_COSMOLOGY=ON`, `IPPL_ENABLE_FFT=ON`, and `BUILD_TESTING=ON`.
With `IPPL_USE_STANDARD_FOLDERS=ON`, executables are under `build_openmp/bin`
instead of `build_openmp/demos/cosmology`.

## Run Instructions

Run the adapter regression, which generates its inputs and checks imported
states and evolution without requiring FastPM:

```sh
/Users/adelmann/.venv-h6/bin/python -B demos/cosmology/python/tests/test_evolution_adapter.py \
  --exe build_openmp/demos/cosmology/CompareCosmologyEvolution \
  --production-exe build_openmp/demos/cosmology/Cosmology
```

Build `Cosmology` as well for this regression. For the native FastPM comparison,
see [matched-evolution reproduction instructions](README.md#reproduce).

Direct invocation, once an input fixture exists:

```sh
env OMP_NUM_THREADS=1 OMP_PROC_BIND=false \
  mpirun -n 1 build_openmp/demos/cosmology/CompareCosmologyEvolution \
  16 16 100 0.3 0.02 0.1 100 10 input.csv output-evolution
```

Arguments are `NP NM L Omega_m a_initial a_final n_steps n_checkpoints input.csv output_dir`.
`NP^3` is the particle count; `NM` is the mesh size per dimension; `L` is the
box length in comoving Mpc/h. Scale factors obey `a = 1/(1+z)`.
The CSV header must be `id,x,y,z,px,py,pz,mass`, with unique IDs `0..NP^3-1`
and unit masses. Momenta are canonical `p = a^2 dx/d(H0 t)`, not km/s.
Use a fresh output directory. Outputs contain synchronized checkpoint particle
snapshots, diagnostics, checkpoint information, and metadata.
Use `-n 2`, `-n 3`, or `-n 4` for other MPI decompositions.
