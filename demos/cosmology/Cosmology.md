# Cosmology

## Purpose/Model

`Cosmology` generates initial conditions and evolves a periodic cold-dark-matter
particle distribution. The model is flat, radiation-free Lambda-CDM with one
collisionless matter species. Initial conditions use first-order Lagrangian
perturbation theory (1LPT/Zel'dovich), with a uniform lattice, a single sine mode,
or a Gaussian spectrum. The spectrum supports BBKS or a supplied transfer table.

Gravity uses CIC deposition and interpolation with a spectral particle-mesh
force calculated through heFFTe. Evolution uses kick-drift-kick integration
with steps uniform in `log(a)`. The documented local qualification covers
OpenMP and 1–4 MPI ranks. General nonlinear halo statistics, 2LPT, GPU performance,
and production checkpoint/restart reliability require separate qualification.

Source: [Cosmology.cpp](Cosmology.cpp), with the implementation in
[CosmologySimulation.h](CosmologySimulation.h), [CosmologyPhysics.h](CosmologyPhysics.h),
and [CosmologyConfig.h](CosmologyConfig.h).
See [LINEAR_PHYSICS.md](LINEAR_PHYSICS.md) for equations, conventions, and limitations.

## Build Instructions

Run from the repository root. Rebuild the existing configured local OpenMP build:

```sh
cmake --build build_openmp --target Cosmology -j 4
```

For a fresh build, use CMake 3.24 or newer, a C++20 compiler, MPI, and OpenMP:

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
cmake --build build_openmp --target Cosmology -j 4
```

The [README build section](README.md#build) supplies the local macOS compiler,
MPI, OpenMP, and reusable dependency paths. With `IPPL_USE_STANDARD_FOLDERS=ON`,
the executable is under `build_openmp/bin` instead of `build_openmp/demos/cosmology`.

## Run Instructions

From the repository root, run a supplied parameter file:

```sh
env OMP_NUM_THREADS=1 OMP_PROC_BIND=false \
  mpirun -n 1 build_openmp/demos/cosmology/Cosmology \
  demos/cosmology/input/linear-sine.par
```

Use `-n 2`, `-n 3`, or `-n 4` for other MPI decompositions. On a small local
machine, Open MPI may also need `--oversubscribe`.

| Parameter file | Initial condition | Mesh and particles | Evolution |
| --- | --- | --- | --- |
| [linear-sine.par](input/linear-sine.par) | Sine mode, initial amplitude 0.001 | 32³ each | z=49 to 9, 100 steps |
| [linear-gaussian.par](input/linear-gaussian.par) | Gaussian BBKS spectrum | 32³ each | z=49 to 9, 100 steps |
| [linear-uniform.par](input/linear-uniform.par) | Uniform lattice | 16³ each | z=49 to 9, 10 steps |

The application accepts one parameter-file argument. Inputs use `name=value`
assignments; `#` and `//` introduce comments. `np` is the size per dimension of
both the generated particle lattice and mesh; the particle count is `np³`.
`nt` is the number of steps. `z_in` and `z_fi` set the initial and final redshift.
Unknown and duplicate parameters are rejected.

`output` is resolved relative to the run's working directory. Use a fresh output
location for each run: existing simulation output is rejected. For repeated
rank comparisons, use separate working directories as shown in the
[README run section](README.md#run-on-14-ranks).

Outputs include `diagnostics.csv`, `metadata.txt`, and initial/final particle
CSVs per rank. Set `write_particles=false` to disable particle snapshots.
Positions and box lengths are comoving Mpc/h. Snapshot `px,py,pz` are canonical
momenta `p = a² dx/d(H0 t)`; peculiar velocity is `100 p/a` km/s.

Run the built-in spectral-force and migration checks without a parameter file:

```sh
env OMP_NUM_THREADS=1 OMP_PROC_BIND=false \
  mpirun -n 4 build_openmp/demos/cosmology/Cosmology --self-test
```

For broader checks, see the separate [physics test guide](TestCosmologyPhysics.md),
[frozen-force guide](CompareCosmologyForce.md), and
[imported-evolution guide](CompareCosmologyEvolution.md).
