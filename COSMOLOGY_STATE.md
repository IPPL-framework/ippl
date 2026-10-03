# Cosmology task state

## Plotting follow-up

User requested plots of the validated results. Added a reproducible Matplotlib
script, demos/cosmology/plot_zarija.py, to read the saved campaign and scalar
probe without rerunning simulations or changing physics. Planned figures:
dimensional shell power plus mean per-mode residuals at z49 (z200 phases reused,
not pooled), background D and sub-ppm cross-code differences, and MPI rank
consistency as a fraction of each recorded tolerance. Reconstruct raw shell
power from snapshots; theoretical Gaussian errors account for varying P(k)
within each shell. Finished: three PNG/SVG figures, plot_data.json and a hash
manifest are in build_openmp/demos/cosmology/zarija-plots-final. Reconstructed
24 shell means agree with the saved checks within1e-12, with exact pair counts.
Analysis-script and captured-table hashes match the validated campaign. All
figures were visually inspected; clarified acceptance labels and moved a growth
annotation to avoid a curve. Initial draft outputs remain in zarija-plots.
README documents reproduction. No new simulations, physics changes or tolerance
changes; these plots visualize the existing evidence and its scope limits.
Independent review confirmed the weighted shell variance and plot conventions;
all eight analysis tests and final source/output hash verification pass.

## Matched-cosmology IC validation against Zarija

User approved the existing model and requested validation against the supplied
Zarija generator. Scope remains the shared flat, radiation-free Gaussian CDM
subset, with BBKS and supplied CMBFAST-table spectra; no additional physics or
legacy-input compatibility is being implemented.

Plan: (1) build a traceable reference from preserved original sources; (2) directly
compare original/new transfer, spectrum normalization, growth and velocity
factors; (3) run matched-parameter IC ensembles, reconstruct displacement/density
Fourier modes and compare statistical power, momentum and lattice conventions;
(4) check IPPL rank independence and record known reference differences explicitly.
RNGs differ, so identical seeds do not authorize particle-by-particle comparisons
between codes. Acceptance limits must be stated before the relevant run.

Ownership: background agent builds reference and reproducible build script;
validation agent implements deterministic physics probes; physics-audit agent
implements the ensemble runner and analytic tests; root owns integration, final
runs and report. Original source tree remains untouched.

Current state: unmodified reference source compiles with LLVM21/OpenMPI5 and a
locally built, SHA256-pinned FFTW3.3.10. An OpenMPI macro exposes deprecated MPI-1
declarations; no reference source patches. Double-precision 16^3 smoke runs passed
on 1, 2 and 4 ranks. Reference executable and build manifest are durable under
build_zarija/reference; static FFTW libraries have no temporary runtime dependency.

Protocol fixed before ensemble execution: N=32, L=168.75 Mpc/h (legacy float
output conversion factors exactly representable), Omega_m=.31, Omega_bar=.0487,
h=.675, sigma8=.82, n_s=.965, z=49 and 200, TF flags4 and0; eight positive int32
seeds. Reference format2 avoids its serial-MPI long/int ID-transfer mismatch.
Compare independent interior Fourier pairs, excluding DC and Nyquist planes;
correct node/cell-centered lattice and x/z-fastest ID conventions explicitly.
Reference generic velocities map to canonical p=a^2*v/100. Check amplitude,
shape, Gaussian moments, seed independence, longitudinality and p/displacement;
rank tests compare IPPL1–4 directly, reference1/2/4 statistically.

Initial public-API probe result: all 4655 deterministic gates passed for BBKS,
raw supplied table, and input-normalized table, at the final matched L=168.75.
Max relative differences roughly 5e-16 in sampled T, 4.8e-7 BBKS P, 3.3e-5 table P,
3.3e-7 D, 7.3e-7 Ddot. Evidence:
build_openmp/demos/cosmology/zarija-physics-8p5qjy_c/results.json.
This scalar-only run was superseded by the complete two-stage target below.
Known reference first-row normalization/extrapolation defect remains explicit:
its low-k transfer is malformed, but its coarse normalization quadrature misses
that narrow interval. No reference-physics edits or post-run tolerance relaxation.

Before the first ensemble run, independent knot-split Gauss16 integration found
a 6.33736e-9 P normalization difference from IPPL's log-Simpson table integral.
Therefore IPPL transfer-scaling tolerance is separately declared as 1e-8;
same-transfer redshift scaling remains 1e-9. No fitted normalization is used.
Runner synthetic tests cover origin/order/units, binary ABI, unique-mode count,
and intentional bad momentum, transverse displacement and amplitude.

Completed validation: full CMake target cosmology_validate_zarija succeeded.
Final deterministic probe: all 4655 gates pass; original source unchanged.
Evidence: build_openmp/demos/cosmology/zarija-physics-hd3av78e/results.json.
Final initializer campaign: all 84 runs / 1047 checks pass, zero failures.
Evidence: build_openmp/demos/cosmology/zarija-validation-f2c9zb42/results.json.
44 IPPL runs cover ranks1/2/3/4; 40 reference runs cover ranks1/2/4. End-of-run
source, executable, table and analysis hashes match the captured provenance.
All seven cosmology CTests pass, including 8 analytic comparison self-tests,
the prior quick evolution suite, and spectral/migration checks on ranks1–4.

Measured ensemble mean P/theory: IPPL .995736642–.995736648; reference
.998164308–.998196423, within predeclared finite-ensemble limits. Maximum
momentum-relation relative L2 errors: IPPL1.998e-12, reference3.921e-7.
Maximum longitudinal residual2.111e-13. All12 IPPL rank comparisons pass:
maximum position RMS6.502e-16 Mpc/h; momentum RMS1.339e-17 canonical units.
Same-TF redshift scaling: IPPL6.263e-13, reference5.138e-8; transfer scaling:
IPPL3.169e-9, reference1.604e-5. No fitted normalization, no post-run tolerance
relaxation, no production physics or original reference source changes.

Changed files for this goal: CMakeLists.txt and README.md in demos/cosmology;
new validate_zarija.py, tests/test_validate_zarija.py, reference/build_zarija.sh,
reference/ReferenceABI.cpp, reference/CompareZarijaPhysics.cpp,
reference/compare_zarija_physics.py; this state file. Build/test artifacts remain
local and ignored. Reproduction protocol and result summary are in the README.

Qualification remains the shared flat Gaussian CDM 1LPT subset on local CPUs,
for this matched cosmology and the common non-DC/non-Nyquist Fourier modes.
The reference low-k table defect, different RNGs and different Nyquist handling
are explicit exclusions from equivalence. This does not qualify additional
reference physics, nonlinear evolution, GPUs, or exascale performance.
Final review complete: independent audit checked all 84 input hashes and required
logs/snapshots, 1047 unique passing checks, exact ensemble/rank coverage, scalar
gate completeness and preserved provenance. All nine intentionally ungated
legacy-difference diagnostic rows remain in the scalar report. git diff --check
and reference build-script syntax check pass. Only task-generated Python caches
were removed; all numerical evidence remains. Original dirty IPPL checkout is
unchanged. Goal COMPLETE for the stated scope; saved on codex/cosmology-linear,
with no push. Further physics or machine-scale qualification is a separate goal.

## Completed preceding goal

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
