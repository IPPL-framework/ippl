# Cosmology task state

## Current work: authorized login-node CPU completion

2026-10-04: user explicitly authorizes CPU execution on the Merlin login node,
up to eight MPI ranks, while the GPU partition has a hardware problem. This
supersedes the older compute-only restriction for this CPU task. We retain a
conservative four-worker/four-rank limit, one thread per nonlinear rank and
serial campaigns. GPU execution remains deferred. No unrelated process will
be stopped (an existing OPALX regression/salloc process is unrelated).

Plan: add an explicit guarded --login mode to the deployment launcher, verify
RHEL8 login compiler/Python/MPI compatibility, push and fast-forward only the
isolated CPU worktree, then build fresh and run 21 cosmology regressions,
eight-run pipeline smoke and all 18 Gaussian cases. Keep the completed local
spatial evidence separate. Do not transplant or resume the macOS journal.
Retain failed numerical gates and produce Gaussian plots when the stage is
complete. Frozen numerical sources, local evidence and tolerances are unchanged.

Initial inspection: merlin-l-001.psi.ch, RHEL8.10, 88 logical CPUs, load about1.5,
192 TiB free on /data/user. Isolated CPU worktree clean at3b59ef1a6. Toolchain
probe passed: GCC/G++14.3.0, CMake4.4.0, Python3.11.11, OpenMPI5.0.10_slurm;
UCX_TLS=sm,self with four-rank hostname launch succeeds (libxml version warning).
Agent validation owns launcher and mocked guard tests; root owns deployment,
execution, docs and final evidence. Root reviewed launcher changes; unchanged
fixture-comparison tests11 pass. New evidence path will be
/data/user/adelmann/cosmology-cpu-login-20261004 with task-only tmux session
cosmology-cpu-20261004, for resilience to SSH disconnection. No build/study has
started yet. Next: finish helper tests, commit/push/deploy, then launch once.

## Latest execution: 44/52 local CPU runs; compute and disk blocked

2026-10-03 21:38 UTC: three additional local CPU runs completed with the frozen
sources, executables, inputs and unchanged acceptance budgets. Each was launched
separately with `--resume resolution-study-cffjva__ --stop-after 1` after passing
the original peak-space guard, including its 1 GiB reserve:

- Native FastPM, NP64/NM64, nt2048, z49, rank1: 43/43 per-run checks passed.
- IPPL, NP64/NM64, nt1024, z49, rank1: 39/39 per-run checks passed.
- Native FastPM, NP64/NM64, nt1024, z49, rank1: 43/43 per-run checks passed.

All three processes returned zero and all 27 checkpoints were archived. An
independent audit verified archive file hashes and numerical digests, complete
IDs, finite values, retained artifact/input/executable hashes, all 28 original
provenance hashes and 12 source-copy hashes. No frozen numerical source changed.
Campaign now has 44/52 runs: 34 spatial and 10/18 Gaussian. Gaussian comparisons
and qualification remain incomplete; per-run passes do not establish cross-code,
resolution, starting-redshift, timestep or rank agreement. Earlier spatial
failures and its restricted first-shell qualification are unchanged.

The next guarded resume (session68516) returned exit3 with an actual disk block:
need 1,750,073,344 free bytes, have 1,667,796,992. No simulation was launched.
Report/storage are resumable; next case is Gaussian NP64/NM64 nt4096 z49 IPPL.
Eight runs remain. The earlier sessions42825,93400,16621 are finished; no local
simulation/controller is running. Do not reduce the reserve or delete evidence.

Merlin was checked again at 21:35 UTC: merlin-g-100 is still down/not responding,
user queue empty, both isolated deployments clean. Only gmerlin6/gwendolen is
authorized; the CPU maintenance-account dry-run was rejected. No actual Slurm
job was submitted and no login-node build or simulation was run. This is the
third consecutive goal turn with the same external compute blocker. Local disk
now blocks further numerical progress too. Goal is not achieved; mark blocked
until an authorized compute allocation or sufficient local space is available.
No automatic monitoring was requested or created.

Next: CPU first. Restore gwendolen or obtain another authorized CPU queue, run
the prepared fresh Linux campaign in /data/user/adelmann/ippl-cosmology-linear,
then the isolated A100 workflow. Do not resume Mac journals on Linux. Alternatively
resume this exact local frozen study after freeing sufficient space for the
remaining eight runs. GPU branch codex/cosmology-a100-validation contains the
prepared metadata/launcher successor (functional commit6bd52b76b); see
/Users/adelmann/git/ippl-cosmology-a100/COSMOLOGY_STATE.md. CUDA has not been built
or runtime-validated. Preserve the original dirty user checkout on both hosts.

## Current work: Merlin6 CPU continuation before A100

2026-10-03: user authorized ssh merlin6, branch push/pull, and subsequently
explicitly requested finishing CPU validation before GPU work. Read OPALX
HANDOFF.md and October build/MPI scripts, plus remote IPPL HANDOFF.md/AGENTS.md.
Pushed codex/cosmology-linear at affc58ece1a10f662c39806699200783ed940fc4.
Preserved remote ~/git/ippl (symlink to /data/user/adelmann/ippl), branch
593-gh200-warnings, HEAD e6b996346, untracked HANDOFF.md. Fetched and created a
separate detached worktree /data/user/adelmann/ippl-cosmology-linear at affc58ece.
Remote filesystem has approximately192TiB available; no local scientific data
was deleted or transferred. Original local dirty checkout is untouched.

Plan: fresh Linux OpenMP build plus pinned native FastPM, all cosmology tests,
eight-run smoke, then complete18-run Gaussian matrix; preserve completed local
spatial results and all41 old runs. Do not use --resume with Mac provenance on
Linux. Same Gaussian protocol/seed, but regenerated IC byte equality is not
assumed. A new standalone fixture comparison utility will record input hashes
and numerical differences without inventing a cross-platform tolerance.
New merlin/cpu_validation.sh is compute-only, exclusive-new-evidence-directory,
four-CPU guarded, serial tests/run controller, -j4 build, one thread/nonlinear
rank. Same numerical sources/budgets remain frozen and unchanged. GPU work is
deferred; its future MPI/FFT communication must be qualified explicitly because
use_heffte_defaults sets GPU-aware behavior independent of the CMake default.

INFRASTRUCTURE BLOCK: ssh succeeds; accessible gmerlin6/gwendolen node
merlin-g-100 is DOWN+NOT_RESPONDING, since2026-10-03T03:25:33, reason Not responding.
No user jobs are active. Default visible CPU partitions are absent for user;
`sinfo -a -M merlin6` exposes other groups' and maintenance partitions, but
`sbatch --test-only -M merlin6 -A merlin -p cpu-maint` is denied with invalid
account/account-partition. gmerlin6/gwendolen CPU-only dry-run reports requested
node configuration unavailable. Account query shows gmerlin6/gwendolen only.
No actual job submitted, no build/scientific execution on login node. Asked user
whether another authorized CPU account/partition is available or to wait for
gwendolen. No automatic monitoring created. Do not claim CPU/GPU validation ran.
Launcher review complete: record explicit final report state (exit1 can also be
an execution exception), actual dependency Git revisions, source/executable
hashes, and script SHA. Python module switch3.14.4→3.11.11 verified on login host
(version query only); no environment installed yet. Local helper tests pass:
fixture comparison11, launcher guards5 (arguments, non-Slurm, allocation bounds,
login-host refusal, shell syntax). Frozen sources and numerical budgets untouched.
Deployment helpers committed/pushed as6bfc807f963b9b64f7685f4b49023fe0908e4f0d;
isolated remote worktree fast-forwarded to that commit, clean, remote bash -n
passes. Original remote checkout still has only its pre-existing HANDOFF.md.
All24 local frozen source/source-copy hashes verified again; user queue empty.
Next: run on compute nodes only after access/hardware is available. Prepared
launcher has NOT been runtime-validated. This entry supersedes
the old no-push/disk-only next-action notes below; Mac resume remains possible.

## Current work: spatial robustness and common-phase Gaussian evolution

User authorized points 1 and 2 on 2026-10-03; they will make disk space.
Work remains in isolated codex/cosmology-linear at baseline 19a7d952e. Preserve
every earlier report/source/executable; no production physics or tolerance edits.
Root owns a new study runner, integration/docs; background owns Gaussian IC
fixtures/tests; validation owns disk-aware numerical storage/tests; physics audit
owns spectrum extraction/tests and independent protocol review.

Predeclared spatial matrix: NP,NM in {32,64}, pancake and coupled3D, both codes,
nt1024, ai=.02 to af=.2 (16 runs); rigidly translated NP64 at NM32/64 (8 runs),
same physical shift (.37,.23,.41)*L/64; finest64/64 nt512 and2048 (8 runs), plus
finest coupled3D nt1024 rank3 (2 runs). Compare translated positions after removing
the known offset and Fourier coefficients after removing the known phase.
This is a crossed study, not only simultaneous particle/mesh refinement.

Predeclared Gaussian matrix: physical-mode-keyed seed20261003, one continuous
BBKS realization, fixed spherical initial band 0<|n|<=12, same field sampled at
NP32/64. Omega_m=.31,Omega_bar=.0487,h=.675,n_s=.965,sigma8=.82,L=168.75.
Sigma8 normalizes the continuous spectrum, not realized finite-box variance;
pure BBKS has no baryonic transfer features. Radiation-free flat Lambda, 1LPT.
z49 NP/NM crossed32/64 nt2048 both codes (8 runs); finest64/64 z49 nt1024/4096
(4 runs); finest64/64 z99 nt2048/4096 (4 runs); finest z49 nt2048 rank4 (2 runs).
af=1, nine logarithmic checkpoints; compare starting redshifts only at exact
common final a=1 using4096 outputs. No interpolation between unequal epochs.
Combined full study52 runs. Shared rounded-once float32 p as before.

Budgets frozen before simulations: original rank, initial-state, native factor,
momentum and timestep limits retained. Finest temporal dx/h and dp/pRMS<=.002,
ratio>=1.5 final (pre-cross toy ratio2–6; below-analysis-floor is not order).
New shell density temporal finest difference<=.002. Shell edges .5,1.5,2.5,4.5,
6.5,8.5,10.5,12.5. Paired-code per-shell P and complex budgets2% at NM32,1% at
NM64, with correlation>=.999/.9995. Particle/mesh sensitivity per-shell P/complex
<=5%, correlation>=.999. Toy translation P<=1%,complex<=2%,correlation>=.9998.
Gaussian z49/z99 P<=2%,complex<=3%,correlation>=.999. All defined-signal shared
epochs considered; widest contiguous low-k prefix passing applicable controls
is reported, with all failures retained. Nonmonotonic changes are not hidden.

Qualification capped at |n|<=4, measured by direct particle Fourier sums. The
Gaussian spectrum through12 also uses fixed128^3 interlaced/window-corrected PCS
for characterization; direct low4 extraction comparison has absolute/relative
budget max(1e-12,1e-3*norm_direct). Passing low4 does not qualify extraction above4.
No continuum/halo precision/GPU/exascale claim. Preserve local pancake phase-space
and sampled Jacobian diagnostics, including pre-cross analytic residuals.

Disk: initially1.9GiB free. New storage reserves1GiB plus conservative full-run
CSV/archive peak before each launch. Only current study's generated CSVs may be
replaced by verified exact uint64/float64 numerical NPZ archives; original CSV
bytes are hashed but not recoverable from NPZ. Old evidence is untouched. Disk
blocks are explicit/resumable, not passes or scientific failures. No ambiguous
interrupted simulation is automatically rerun. Next: finish/review helpers and
runner, run synthetic/pipeline tests, then execute guarded spatial and Gaussian
batches as capacity permits. Save results after each run for safe continuation.

Progress: storage helper16 tests, Gaussian fixture15 tests, original spectrum
helper18 tests and new runner19 tests pass. Source snapshots now accompany new
studies, and prepared initialization plus stale-report resume were hardened.
Initial pipeline smoke resolution-study-_ajzgvkl completed8 runs/364 checks with
36 failed auxiliary Gaussian FFT extraction checks (0.9–1.14% vs unchanged0.1%
limit); every import/physics/rank/storage check passed. This is coherent particle
lattice aliasing in the diagnostic CIC estimator, not a solver disagreement.
Exact direct low-mode measurements remain independent. All12 source versions
for that smoke are preserved and hash-verified in its source-snapshot directory.
Physics audit replaced only the diagnostic assignment with interlaced PCS
(four-point cubic B-spline, sinc^4 window correction) at128; retain all CIC APIs,
tests and failed evidence. Simulation CIC forces and measurement budgets do not
change. All24 spectrum tests pass; problematic Gaussian initial-state extraction
residuals improved from0.75–1.14% to below0.0001%, retaining the0.1% gate. Runner
now saves completed-stage comparisons before starting the next stage, so a later
disk block cannot hide spatial results, including exact --stop-after boundaries.
Independent PCS review confirmed the piecewise cubic kernel within4.45e-16;
all20 registered CTests pass (44.84s), including new15/24/16/23 helper tests.
Fresh pipeline resolution-study-_v58pjf8 passes8 runs/364 checks. Full52-run
campaign now running in build_openmp/demos/cosmology/resolution-study-cffjva__;
no source/protocol changes permitted while executing/resuming that campaign.
Root owns run monitoring/docs; physics audit owns a new saved-evidence plotting
script/tests (now implemented,15 tests pass; optional plotting CTest passes).
Independent final storage review found no blocker: do a read-only final source
snapshot hash audit in addition to the runner's original-source hash audit.
Atomic journal replacement/process-interruption recovery is tested; do not claim
guaranteed sudden-power-loss durability (parent directory entries not fsynced).
No production executable changed. Disk subsequently measured3.7GiB
free; /tmp and worktree share the same filesystem, explained to the user.
Provisional read-only inspection after12 runs: all four unshifted pancake pairs
have shell-power cross-code differences<=2.155e-6. Mesh32 translation passes;
mesh64 finala=.2 shell2 (axis modes3,4) translation power sensitivity is
1.3362646%IPPL/1.3362803%FastPM, above the unchanged1% budget in both. These
will be recorded as failures by completed-stage comparisons; not a reason to
adjust budgets. Source fixtures/coefficients already persist in results.json.
Independent pancake audit after all16 runs:12 archives (two codes × three nt ×
checkpoints4/8) passed SHA, numerical digest and IDs; independently recovered
density coefficients match saved values within3.54e-13. Finest1024→2048 final
dx/h=7.20695e-6IPPL/8.99486e-6native, dp/p=2.28023e-6/2.32935e-6; shell complex
residuals<=2.066e-6, all below .002. Position ratios pass (early4.119/3.435,
late4.042/3.241). Momentum/global-density differences below1e-5 analysis floors
do not establish an order. FixedNP32 mesh refinement fails n2/n3,4 (max powers
6.514%/16.772%); fixedNP64 mesh refinement passes all3 shells. Particle refinement
itself fails n3,4 complex/correlation gates at both meshes. Only planar n1
survives examined spatial controls; completed-stage/global qualification pending.
Spatial stage is now COMPLETE:34 runs,2083/2213 checks pass;130 failures are
cross3,particle56,mesh51,translation20. Bothfixtures qualify firstshellonly;
allglobalcontrols/time/rank pass. Rank3 maxima IPPLdx/h5.89051e-15,dp/p5.02162e-15;
native1.15919e-7/8.62141e-8. Independent coupled3D archive/source audit passed;
finestfinaldx/h .000267841/.000267870,dp/p .000268284/.000268373; lateorders~2.
Saved and root-inspected spatial PNG/SVG/data/manifest at
resolution-study-cffjva__/figures-spatial-release, shown to user. Exact inputreport
SHA11ee94361853d92f87d66c703cc0145f4bd3f2ab003e7895d071c261b49d56d7 is preserved
byte-for-byte in input-report.json.gz; report was still overall incomplete.
Gaussian stage now running:36 totalruns,82 Gaussian per-run checks allpass as of
last inspection; no Gaussian qualification yet. Free disk1.887GiB; guard remains.
Historical strictmass diagnostic flags separately retained in four current
pancake NP64NM64 IPPLruns (base1024,shift1024,base512,base2048), max2.95097e-12
versus unchanged2e-12. Current band qualification uses the previously accepted
engineering baseline and does not relabel those flags as passes. README/user
notified explicitly. Mean-momentum conservation gates allpass as stated.
CURRENT STOP: main study exited3 (disk guard),41/52 analyzed runs. All34 spatial
plus7 Gaussian runs completed and archived;285 Gaussian per-run checks pass,
but fullGaussian controls/qualification are NOT complete. Next nativeplainPM
GaussianNP64NM64nt2048z49r1 was not launched. Preflight required1750073344 bytes,
had1669689344 (about1.555GiB). Told user to free roughly3GiB extra for remaining11
runs; no prior campaign files removed. No simulation process remains from this
main study. Resume exactly from worktree root:
env OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/adelmann/.venv-h6/bin/python -B demos/cosmology/validate_resolution_study.py --resume build_openmp/demos/cosmology/resolution-study-cffjva__
Keep all12 frozen source files, native manifest/artifacts, executables and inputs
unchanged. Source content hashes (not Git HEAD) govern resume. Root final checks:
all21CTests PASS in51.50s;560 provenance/source-copy/input/retained-output/archive
hashes PASS, covering369 numerical snapshot archives from41 completed runs.
Plot output hashes, source hash and decompressed exact-report hash also pass.
Independent audit reproduced all1107 spatial shell comparisons/pass flags within
6.66e-16. Final diff reviewed; git diff --check clean. New campaign occupies2.0GiB;
about1.5GiB remains. Next action requires more disk, then the exact resume command
above; do not infer Gaussian qualification from its285 passing per-run checks.
No push authorized/performed. Original worktree untouched.

## Current work: matched-particle plain-PM evolution

User explicitly accepted the preceding discrepancies as an engineering baseline
and authorized the next evolution stage. Original measurements and failed mass
gate remain unchanged; acceptance does not imply universal nonlinear validity.

Plan: import one identical x,p CSV into both real integrators; compare synchronized
checkpoints through planar shell crossing and a coupled smooth3D nonlinear test;
measure timestep self-convergence, fixed-particle force-mesh refinement and MPI1–4.
Use pinned native FastPM FORCE_PM/CIC/NAIVE, no modified FastPM/COLA stepping,
no upstream source changes, no Nyquist filtering or fitted normalizations.

Shared CLI: NP NM L Omega_m a_initial a_final n_steps n_checkpoints input.csv output_dir.
CSV: id,x,y,z,px,py,pz,mass, exactly NP³ unique IDs0..NP³−1, unit masses; p=a²dx/d(H0t).
Logarithmic full-step endpoints and geometric kick midpoints; outputs at full
synchronized steps only, avoiding native snapshot interpolation/velocity conversion.
FastPM v/acc remain float32 and x/FFT double; physical coordinates shifted by
−L/(2NM) on import and restored on output. Radiation-free flat Lambda background.

IPPL change scope: extract existing KDK body without arithmetic changes; diagnostic
import count override; normalize deposited mass by NM³/NP³ only when counts differ
(preserve exact existing subtract-one branch for NP=NM). No general IC/config
API expansion. Fields/kicks/drifts remain in configured Kokkos memory/execution
spaces; only diagnostic CSV input/output copies host data. Unequal-grid support
changes the intended mean-density normalization, not the force kernel.

Native full solver initialization rejects NM not divisible by ranks although
FFTW/ghost exchange already passed uneven-slab frozen tests. The reference adapter
directly initializes native PM/store/cosmology/VPM for imported particles,
bypassing the lattice/divisibility precheck only; actual solver_evolve, force,
kick/drift and decomposition stay native. Explicitly qualify rank3, not assume it.
Old FastPMForce executable/manifest remain untouched; separate evolution linker,
executable and manifest verify pinned source/library hashes before and after build.

Ownership: background agent native evolution adapter/build; validation agent
IPPL adapter/refactor and import regressions; physics-audit agent independent
integrator/protocol review and analysis tests; root campaign/fixtures/CMake/docs.
Fixed bounded campaign: NP32, NM16/32/64, nt64/128/256, ai.02→af.2, Omega_m.31,
L168.75; planar linear-final amplitude1.5 and coupled few-mode3D IC. Numerical
gates were fixed before comparisons; native Nyquist differences stay an
explicit comparison uncertainty, not an excuse to label unmatched fields equal.
Predeclared limits (before any matched comparison): compare actual imported x
within 128 eps64 L and p exactly after one shared float32 quantization; independent
native drift/kick quadrature 1e-8 relative. Rank differences: IPPL dx/h and dp/pRMS
1e-10 (plus position roundoff allowance), FastPM 5e-5. Planar cross-code dx/h and
dp/pRMS 1e-3. Direct particle Fourier modes 0<|m|<=4, unique +/- pairs: aggregate
power ratio and complex-coefficient difference 5/2/1% for NM16/32/64, correlation
at least .995/.999/.9995. Characterize raw3D trajectories and shell powers;
do not gate them as if Nyquist force operators were identical.

Timestep differences nt64→128 versus128→256: ratio2–6 at checkpoint4 (pre-crossing),
at least1.5 at final; finest final dx/h and dp/pRMS <=.002. Predeclared analysis
floors 1e-6 cells,1e-5 relative momentum and1e-5 relative complex modes; below-floor
status is not evidence of roundoff dominance or a measured convergence order.
NM32→64 final aggregate resolved power change<=5% at fixed NP32; not a continuum
limit. Mean momentum conservation relative to actual imported mean (not zero)
<=1e-10 IPPL /5e-5 FastPM, normalized by observed pRMS. Original mass gate2e-12
remains separately recorded, not relaxed. IPPL CIC mesh-sum residual and native
unit-particle count residual are explicitly distinct. Actual planar final sampled
Jacobian must be<-.01. Snapshots remain recoverable as hash-verified lossless gzip.

Adapter regressions:19 MPI runs/13 checks passed in evolution-adapter-6sh1wmue,
including exact import, ballistic ranks1–4, nonuniform unequal-mesh oracle and
production-driver equivalence. Unsupported tiny local mesh extent is explicitly
rejected before core halo assertion. Native8-run smoke passed ballistic/EdS
shell-crossed cases1–4r; checkpoint0 export was corrected to actual native
state instead of echoed input before comparison execution. Native final smoke
evidence: build_fastpm/evolution/smoke-6_q_d9uu/results.json. Executable SHA256
177cb831e0379648566b749286801862a481d9709404c6af28d3a9b1d1108a5e; manifest
9ddb833a5304efc188320e9970e741c6cc0947748e01184bd7d2fd86eac2fa11.

Completed quick8/441 checks in matched-evolution-_x5sqrt5 (all pass),15 registered
CTests (all pass),20 independent analysis tests (all pass;5 additions reject
malformed native factor schedules), and full existing linear22/275 (all pass)
in cosmology-validation-irdp24fl. Full32 campaign completed under
build_openmp/demos/cosmology/matched-evolution-5v7u3y98:1723/1725 checks pass.
All source/input/executable hashes unchanged before/after. Failures only final
coupled3D128→256 momentum difference:IPPL .002763434661810358,FastPM
.002755553597039562 versus unchanged .002 budget. Both reduction ratios~4.03.
All cross-code/rank/initial/factor/netmomentum/shellcross and mesh-budget gates
pass. Raw3D trajectories not gated; maximumresolvedpower .243%,complex .338%.
Mesh32→64 power changes:pancake3.63%,3D2.67–2.84%; pancake mesh changes are NOT
monotone decreasing, so no continuum convergence claim. New maxCICmass1.22e-12
does not invalidate earlier retained analytical-pancake failure.

Bounded follow-up completed: separate validate_evolution_refinement.py and two
coupled3D NP32/NM32/r1 nt512 runs, exactoriginal imported input, unchangedlimits,
reuse128/256 savedstates. Results:evolution-refinement-k2qeaylz,99/99 checks pass.
Final256→512 momentum differences .0006962128512831455IPPL and
.0006942805963183412FastPM, below .002. Position differences .00044926/.00044838
mesh cells. Finalmeasuredorders~1.97–2.01; pre-cross momentum belowanalysisfloor
does notyieldorder. This is single-rank timestep qualification, not newMPI/mesh
coverage, and noterroragainsttruth. All original/new source/executable/input,
parentreport and reusedcompressed snapshot hashes unchanged. Originalfullreport
SHA25634568aa5978c7c651410f1b8187478648ded7dd47bb5b98ac3a2ed77bd101b1d;
two originalfailures verbatim retained, failures_superseded=false.

Plotting completed:two publication-quality PNG/SVG summaries, plotted-data JSON
andverifiedSHAmanifest in matched-evolution-5v7u3y98/figures-release. RootvisualQA
confirmed titleoverlapfixed, canonicalunits/uniqueFourierband/actualzeroresults
legible; original128→256 failedgates explicitlycaptioned. Seven plot tests pass.
Newrefinementhelper11synthetic tests pass (includingmockedonlytwo-runorchestration)
and all13parentprovenance/eightreusedsnapshot hashes preflightpass. Source/config
diff reviewed independently; originalmaincheckout/oldreferenceartifacts untouched.

Next scientific work:joint particle/mesh refinement and phase-shift tests before
claiming continuum density convergence; then common-phase GaussianCDM nonlinear
statistics at several epochs against nativeplainPM. GPU parity, scalableI/O,
loadbalance andmultinode scaling remain separate qualification stages. No general
nonlinearLCDM, continuum, GPU or exascale validity claimed. Final16/16 registered
cosmologyCTests passed in40.54s; git diff --check clean. Source changes complete,
ready for local commit; no push authorized or performed.

## Plotting frozen-force and pancake evidence

User requested plots of the completed validation stage. Reproduce static PNG/SVG
figures from frozen-force-5_li99qh, pancake-validation-rs1o9glb and its diagnostic
audit, without rerunning simulations or changing tolerances. Root owns plotting
script, data/hash manifest, visual inspection and README; independent reviewer
audited the saved quantities and annotations. Show native-operator disagreement,
global/time convergence, unresolved local gradients and the retained mass gate.
Do not floor undefined uniform-force relative errors onto a logarithmic axis,
subtract RMS magnitudes to estimate residuals, or turn the partial pancake
qualification into an overall pass.

Completed: four publication-quality PNG/SVG figures plus plot_data.json and
source/output SHA256 manifest at build_openmp/demos/cosmology/pm-validation-plots-release.
New plot_pm_validation.py reads saved evidence only; no model runs or source-data
changes. README contains reproduction command. Ten focused synthetic plot tests
pass (residual norms, component/vector RMS scaling, missing ranks, exact zeros,
and marker visibility). Independent review found and fixed a clipped small-error
marker. Visual QA corrected legend/footnote overlap and crowded log tick labels;
all figures now inspected, annotations derive numerical orders/counts from data.
Earlier drafts remain under pm-validation-plots and pm-validation-plots-final.
Retained pancake failure and local-gradient nonconvergence remain explicit.
No physics, tolerance, or parallel execution changes; next scientific work is
unchanged from the validation handoff below.

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
