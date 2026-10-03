# Cosmology task state

## Current work: analytical pancake and plain-PM FastPM comparison

User approved the next validation stage, starting with frozen forces. Preserve
all earlier qualification; do not infer nonlinear or exascale validity from IC
agreement. Work remains on codex/cosmology-linear in the isolated sibling tree.

Plan: (1) build a pinned independent FastPM reference and expose IPPL's existing
force path through a diagnostic particle-import/force-output adapter; (2) feed
identical equal-mass particles to both and an independent NumPy CIC+spectral
oracle, checking force/density, indexing, units, periodic wrapping and ranks1–4;
(3) validate finite-amplitude planar evolution against the analytical solution
before shell crossing with timestep/mesh convergence; (4) only after these gates
pass, establish the requirements for matched-particle nonlinear evolution.

Ownership: background agent owns FastPM acquisition/build/native-force harness;
validation agent owns IPPL adapter in CosmologySimulation.h and separate test
executable; physics-audit agent owns kernel review and pancake runner/self-tests;
root owns frozen-force fixtures/oracle/runner, CMake integration, final runs/docs.
Source physics in FastPM must not be altered to force agreement. Any build-only
compatibility changes must be isolated, hashed and documented. Main IPPL solver
and current initialization/evolution algorithms are not to change in this stage.

Contract: CSV input id,x,y,z,mass with N³ unique IDs and unit masses. Positions
are comoving Mpc/h; compare canonical force F=-grad(phi0), whose source is
1.5 Omega_m delta. Output particle forces by ID and density by global cell index.
IPPL field samples are cell-centered; any FastPM grid-origin conversion must be
explicit. Use plain PM, CIC scatter/gather, spectral ik/k², no deconvolution or
extra softening. IPPL uses +i*k*1.5*Omega_m/k^2. FastPM native acceleration has
the opposite sign, and its normal kick supplies -1.5*Omega_m; the export applies
that documented conversion. FastPM node coordinates are x_IPPL-h/2 modulo L.
IPPL zeros each differentiated Nyquist plane; native FastPM only zeros the eight
fully self-conjugate corners. Preserve this difference and predict it with an
independent native R2C oracle; raw forces must agree only on fixtures without
Nyquist power. Compare all common-band mesh modes, never filter native solves.
FastPM double FFT/CIC/grid fields retain float32 k/k^2 tables and particle acc.

Predeclared frozen limits: IPPL density RMS 2e-12+2e-12*reference_RMS,
force RMS 2e-13*L+2e-11*reference_RMS; FastPM density 5e-12+5e-12*reference_RMS
(double CIC confirmed before any reference run), force 5e-9*L+5e-6*reference_RMS
(native float32 k/acc). Additional position, net force, rank-invariance, kernel,
gather and cross-code residual gates are explicit in validate_frozen_force.py.
No post-run relaxation. Eight deterministic fixtures include a common-band
single-mode deposition, deformations, wrapped/shuffled jitter and a dense cluster.

Pancake protocol: Omega_m=.31, L=168.75, zi49->zf9, final deformation .5/.8;
N16/32/64, nt128 for mesh study; nt16/32/64/128 for time study; x/y/z and (1,1,0);
ranks1-4. Finest-grid RMS x/p error budgets 1%/2% at .5 and 2%/4% at .8;
mesh error reduction >=2 and time successive-difference ratio 2.8..5.5.
These are engineering budgets, not assumed accuracy or exascale qualification.

Current status: FastPM official source pinned to
15b6c4fd7502a81d99dd13f54fcc9cfa44be1331 under build_fastpm/source; build and
API inspection completed; double FFTW/PFFT/GSL reference build and smoke testing
underway. IPPL adapter 11 smoke/negative cases pass, and the independent oracle
passes 7 analytical self-tests. IPPL full frozen campaign passed32 runs/362 checks
at build_openmp/demos/cosmology/frozen-force-ig_p42r9/results.json. An earlier
analysis attempt stopped on header-only empty-rank particle files in the cluster
fixture; the reader is corrected without numerical changes.

First pancake campaign:15 runs/364 checks,363 pass, one retained failure:
N64,Afinal=.8 mass diagnostic max2.072e-12 exceeds predeclared2e-12.
Evidence build_openmp/demos/cosmology/pancake-validation-f3couj_8/results.json.
All trajectory/MPI/time/mesh gates pass, but measured spatial orders are not
asymptotically second order and local minimum-Jacobian convergence is not proved.
Mass failure under independent summation audit; no tolerance relaxation.

Completed frozen stage: pinned, unmodified FastPM reference built successfully
(double FFTW/PFFT/GSL; compile harness by basename to avoid upstream fixed-size
__FILE__ diagnostic buffer overflow; standard transposed FFTW path). Native
high-k and common-band smokes pass ranks1-4. Full CMake target
cosmology_validate_frozen_force passes64 runs/790 checks, no failures:
build_openmp/demos/cosmology/frozen-force-5_li99qh/results.json.
Executable/source/manifest hashes unchanged through the campaign. Particle
force relative RMS vs independent own-operator oracle: IPPL<=6.75e-14,
FastPM<=3.21e-8; native mesh errors<=6.48e-14. Raw cross-code common-band
particle error3.87e-8. Raw oblique/jitter discrepancies2.47%/11.71% are correctly
predicted by the different Nyquist conventions; arbitrary full forces are NOT
qualified as equivalent. No original FastPM/Zarija physics changed.

Final rebuilt Cosmology binary repeats identical pancake results at
build_openmp/demos/cosmology/pancake-validation-rs1o9glb/results.json:
15 runs,363/364 checks pass, same retained mass failure. Current executable and
all physics/analysis hashes match the final campaign. Read-only audit is at
build_openmp/demos/cosmology/pancake-diagnostics-rs1o9glb/diagnostics.json and
reproduced by tests/analyze_pancake_diagnostics.py. Initial/final redeposition
with math.fsum gives exact total262144; worst epoch (step89) has no saved field,
so roundoff diagnosis is strong evidence, not a proof for that epoch. Standard
uncompensated-sum error bound2.91e-11 covers the observed2.07e-12 discrepancy.
All trajectories and convergence budgets pass; N64 Af.8 displacement/momentum
RMS0.414%/0.736%. Spatial orders1.43-1.76 and nonconvergent local Jacobian L∞
errors indicate unresolved grid-scale defects (grid locking is an inference).
No post-crossing/random-CDM/local-density/nonlinear-reference claim is made.

All12 cosmology CTests pass (including12 adapter launches with six negative
cases,7 frozen-oracle unit tests,8 pancake-oracle unit tests, previous spectral,
linear and Zarija-analysis tests). The quick CTests intentionally do not hide
the full N64 pancake gate failure. Independent code review found no blockers;
bash syntax and git diff --check pass. Original dirty checkout untouched.

Changed files: production header diagnostic declaration only; CMake/README;
new CompareCosmologyForce.cpp, FastPMForce.c/build_fastpm.sh, frozen-force and
pancake runners, their unit tests, adapter regressions and pancake audit.
Next: agree shared Nyquist comparison policy; export/check the peak-epoch field
with accurate summation and improve diagnostic reduction if justified; test
independent particle/mesh resolution and pancake phase before matched nonlinear
FastPM evolution. No tolerance relaxation, production physics edits, GPU/multi-
node/exascale claims, external publication or repository push in this stage.

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
