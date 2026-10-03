# Cosmology task state

Goal: provide a local IPPL cold-dark-matter model with linear evolution verified
on one through four MPI ranks. Worktree `/Users/adelmann/git/ippl-cosmology-linear`,
branch `codex/cosmology-linear`, based on `edb8794cd`.

Status: implementation and validation COMPLETE for the explicit user goal:
"have a local 1 to 4 rank linear evolution model avaidable".
The original `/Users/adelmann/git/ippl` worktree contains unrelated solver work
and is preserved.

Approach: use current IPPL; add a small supported cosmology driver alongside
the historical demo, typed input, flat Gaussian LCDM, Zarija-compatible transfer
and growth conventions, deterministic distributed 1LPT initial conditions,
and actual particle-mesh KDK evolution. Validate single modes and Gaussian
fields, mass/particle conservation, Fourier normalization, growth, rank/thread
independence and timestep convergence. Document units and numerical limits.

Task ownership: root implements distributed initialization/evolution/build;
background agent owns CosmologyConfig.h and CosmologyPhysics.h and focused
tests; physics audit agent independently checks conventions.

Implemented: new Cosmology driver, CosmologySimulation.h, typed configuration,
background/transfer physics headers, example inputs, physics documentation,
focused C++ tests, CMake integration, and validate_linear.py. Delivery branch is
`codex/cosmology-linear`; no pushes. Core IPPL and the legacy demo are unchanged.
OpenMP build in build_openmp succeeded using local Kokkos/heFFTe sources.

Checks completed before pausing: focused cosmology physics tests passed; all five
cosmology-labelled CTests passed (physics and spectral force on ranks 1, 2, 3, 4).
A one-rank 32^3 sine evolution from z=49 to z=9 completed, with final amplitude
0.00496649 versus linear theory 0.00499790 (about 0.628% low).
After resumption, the full 22-run MPI evolution/IC/convergence validation passed
all 275 checks in build_openmp/validation-full/results.json. Measured time order
1.988 (positions), 1.995 (momenta); spatial-error reduction factors 3.899 and
3.849. Gaussian Fourier coefficient relative L2 error about 4.8e-12; late-Lambda
growth error 0.943%. Quick 16-run evolution CTest also passed.

First full-suite attempt exposed a harness-only output-directory conflict:
input.par/run.log made the simulation output directory nonempty. Stopped only
the task's validation processes; retain logs in build_openmp/validation-resumed.
Runner now keeps outputs in a separate subdirectory, verifies actual rank/thread
counts, supports MPI process-count flags, and creates unique result directories.

Additional forced-migration regression exposed the existing core PeriodicBC's
large-overshoot/zero-endpoint limitation. Implemented a cosmology-only floor wrap
before all particle updates, with core particle BC disabled (field layout remains
periodic). Tests cover multi-box shifts, exact endpoints, local ownership, unique
IDs, and every migrated attribute. Core IPPL files remain unchanged.
Final wrap fix verified: all six CTests PASS, including the 16-run quick evolution
test and spectral force plus forced migration on ranks 1, 2, 3, 4. Migration's
attribute/position error was zero on all four rank counts. The full CMake target
`cosmology_validate` reran all 22 simulations / 275 checks and PASSED:
`build_openmp/demos/cosmology/cosmology-validation-na4rs6e0/results.json`.
CTest log: `build_openmp/Testing/Temporary/LastTest.log`.
No numerical tolerances have been relaxed.

Shipped examples also executed unchanged: sine on every rank count 1–4, final
growth error about 0.628%, and Gaussian sigma8=0.8 on four ranks (smoke test only).
Outputs: `build_openmp/shipped-examples/`. Legacy StructureFormation also builds.

Moved the isolated worktree to its durable sibling path above; reconfigured and
rebuilt build_openmp there. Prior temporary-location build is preserved as
build_previous_location. README has build/run/validation instructions.

Final reviews covered source, physics conventions, test independence, new migration
wrapper, and CMake wiring. `git diff --check` passed. Original dirty checkout is
preserved. Build/run/validation commands are in demos/cosmology/README.md.

Scope: flat radiation-free LCDM, one collisionless species, 1LPT ICs, periodic CIC
PM gravity and KDK. This qualifies local CPU linear evolution, not nonlinear halo
physics, GPU execution, multi-node/exascale scaling, or production restart/I/O.
Next work beyond this completed goal requires a separate agreed scope.
