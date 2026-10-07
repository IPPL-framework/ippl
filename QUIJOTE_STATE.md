## Completed Quijote fiducial run — 2026-10-07

The actual standard fiducial realization 0 completed from z=127 to z=0 with
134,217,728 particles, a 512^3 force mesh, and four NVIDIA A100-SXM4-40GB GPUs
on Merlin compute node merlin-g-100. Job 354392 is COMPLETED, exit 0:0 and
derived exit 0:0. The objective is verified achieved.

- 512 base timesteps became 514 KDK intervals to hit requested output epochs.
  Snapshots: z=127,1,0.5,0 at steps 0,439,471,514, four binary shards each.
- Controller full-payload checks passed: complete unique IDs, finite x/p,
  particle count/mass/cosmology/epochs and file hashes. The independent final
  audit passed all 18 checks, including actual headers and byte lengths of all
  16 shards. Total production snapshot bytes: 30,064,773,120.
- The full-count pilot passed with exact imported positions and momenta by ID.
- All 36 production diagnostic rows are finite and retain the full particle
  count. Maximum recorded mass-normalization error: 1.11e-16; maximum inverse
  imaginary residual: 2.6173e-14. No tolerances or problem sizes were relaxed.
- Native Simulation::run wall time: 1,659.769 s (27m40s), excluding constructor
  setup, metadata writing and external audits. Timed production execute plus
  input/output auditing: 1,893.36 s (31m33s). Entire Slurm job including pilot:
  2,214 s (36m54s). Sampled GPU peaks: 12,107–12,241 MiB per 40,960 MiB device.
  GNU/Slurm RSS figures in the report are not simultaneous aggregate MPI memory.
- All eight original HDF5 IC files are unchanged in the user-supplied directory:
  /data/user/adelmann/quijote-fiducial-512-20261007/ics-quijote-fiducial.
  Standard fiducial/2LPT provenance is declared from the authenticated official
  listing and user handoff. Remote staged bytes were hashed; no pre-transfer
  checksum or witnessed transfer is claimed. No IC regeneration or rescaling.
- Remote results:
  /data/user/adelmann/quijote-fiducial-512-20261007/run/fiducial/run.
- Local retained evidence:
  build_openmp/demos/cosmology/quijote-merlin-20261007/production-evidence.
  completion.json inventories hashes; audit.json has the 18 checks;
  resource-summary.json has diagnostics/resources; slurm-accounting.txt records
  successful scheduler completion; records/ preserves the final controller,
  benchmark, metadata, snapshots manifest, diagnostics, source and receipts.
- Build354388, validation354389, conversion354391 and production354392 all passed.
  Observer exec5000 and all related command sessions finished. No jobs need
  monitoring/restarting. Do not rerun or overwrite this completed campaign.
- Correct worktree: /Users/adelmann/git/ippl-cosmology-linear, branch
  codex/cosmology-linear, base a2fefe0aee48853517e193a8f96e52a3f5a0e153.
  Preserve existing user AGENTS.md and doxygen/html changes. No commit/push made.
- Execution/data-integrity qualification is complete. Agreement with evolved
  Quijote/GADGET references, force/time convergence and performance scaling
  remain unqualified. Parallel runtime HDF5 was deferred until this successful
  run; current HDF5 support is offline serial conversion only.

## Historical task state

# Quijote fiducial implementation, 2026-10-07

## Goal and authorization
Current active goal: run the actual 512^3 standard Quijote fiducial on Merlin6 A100. This supersedes the implementation-only stage and authorizes build, validation and the large Slurm run. Parallel simulation HDF5 I/O remains deferred until after success. No commit/push requested.

## Baseline
- Branch codex/cosmology-linear at a2fefe0ae.
- Existing user changes: AGENTS.md modified; doxygen/html/ untracked. Preserve them.
- Read AGENTS: physical correctness, OpenMP efficiency, regression checks, mathematical Doxygen for code changes; documentation edits on Mac.

## Design
- Keep existing periodic CIC/spectral PM/KDK physics and generated IC defaults.
- Add production external IC mode with independent particle_count and force mesh np.
- Canonical little-endian phase-space binary header 128 bytes: magic IPPLPS01, uint64 local/global counts, doubles a,L,Omega_m,Omega_Lambda,h,physical particle mass, uint64 flags, 48 zero reserved bytes. Records uint64 ID + six float64 x,p (56 bytes). Input sorted by contiguous zero-based ID; snapshot shards unordered.
- Initial Gadget format1 conversion is serial/chunked; runtime root reads bounded chunks and distributes with MPI. No HDF5 dependency or parallel file reader in this stage.
- Physical positions are Mpc/h and p=a*v_pec/100. Raw Gadget u requires p=a^(3/2)*u/100 exactly once. Imported particles retain original 2LPT realization and amplitudes.
- Output exact requested redshifts as synchronized KDK endpoints; binary per-rank snapshots and epoch manifest. RSD comparison is implemented with p/(a^2 E) and raw P2/P4; production restart remains later work.
- Reuse analysis conventions and add bounded-particle-chunk same-estimator comparisons and reproducible run manifests; no new force compensation.

## Work ownership
- Root: C++ configuration/import/output/scheduling, build registration/docs/integration.
- catalogue agent: Python Gadget converter/wire helpers and tests.
- bao_frontier agent: Quijote runner/analysis and tests.
- quijote_integration_tests agent: production MPI regression script.

## Implemented files
- C++: CosmologyConfig.h, CosmologySimulation.h, Cosmology.cpp, new PhaseSpaceIO.h and CosmologyParticleIO.hpp; TestCosmologyPhysics.cpp and CMakeLists.txt registration.
- Python: quijote_io.py (converter), quijote_analysis.py (common estimator), quijote_benchmark.py (prepare/execute/compare), and three new test files.
- Documentation: README.md, docs/contracts.dox, docs/mainpage.dox, docs/quijote.dox. Strict Doxygen includes every new API.

## Checks and findings
- Built Cosmology, CompareCosmologyEvolution, CompareCosmologyForce and TestCosmologyPhysics with existing OpenMP Release toolchain.
- Converter: 20 tests pass. Runner/estimator: 14 tests pass, including permitted metadata roundoff versus real drift.
- External suite: MPI1/2/4 import/KDK/density RMS, exact-epoch ballistic checks, old CSV adapter agreement, and seven malformed-input failures pass.
- Permanent complete pipeline test passes: split/shuffled Gadget -> converter -> prepare -> execute -> snapshot audit -> real/RSD comparison; actual MPI2/OpenMP2 verified.
- CTest external integration passes (14 launches, 15 checks), log /tmp/ippl-quijote-external-ctest.log.
- Final full CTest: 33/33 passed in 54.66 s with frozen sources, including physics, spectral MPI1/2/3/4, prior force/evolution/reference suites and all new Quijote tests. Log: /tmp/ippl-quijote-ctest-final.log.
- Earlier broad CTest initially 31/32 passed: pancake numerical checks passed but its source-provenance gate detected my concurrent comment-only Cosmology.cpp edit. Clean frozen-source rerun passed. No physical acceptance tolerance relaxed.
- Final strict Doxygen passed (77 files, 847 Python declarations, zero warnings/missing contracts). Log: /tmp/ippl-quijote-docbuild-complete.log; HTML: /tmp/ippl-quijote-docs-complete/html/index.html. Final callback @cond annotation after CTest was verified Python-AST-identical; it only resolves a Doxygen member-reference parsing artifact.
- Independent C++ and estimator reviews found no remaining blocker. git diff --check passes.
- Fixed diagnosed MPI error path: abort inside Simulation lifetime, before collective destructors, prevents malformed-root-input deadlock.
- Fixed density RMS normalization to mesh cells, independently tested with NP != NM.
- Exact final endpoint now uses aFinal; possible last-bit trajectory change relative to exp(log(aFinal/aInitial)). Requested epochs split intervals, so convergence schedule must remain fixed across comparisons.

## Remaining qualification and next step
- Implementation, code review, complete local regressions, strict documentation and diff whitespace checks are complete.
- Runner metadata audit uses the existing C++ 1e-10 relative background/epoch matching tolerance for permitted serialization roundoff; count and imported physical mass remain exact. A focused regression rejects physical drift and changed counts/masses. This is an input/output metadata contract, not a relaxed physical-accuracy gate.
- Next: select the actual standard fiducial realization and original Gadget IC/reference files, prepare a manifest with measured hardware budgets, then qualify the 512^3 pilot and force/time/analysis refinement.
- No real Quijote dataset downloaded and no 512^3 execution, resource/performance claim, or accuracy qualification made.
- Serial root I/O, host mirrors and serial analysis FFT mesh need measured production memory/I/O pilot. Analysis particle chunks are bounded; exact ID audit uses N bytes. FFT/interlaced-CIC aliasing needs analysis-grid convergence.
- Catalogue identity (fiducial vs fiducial_ZA), LPT order and spectrum parameters cannot be proved from Gadget headers; manifest records this.
- No production restart and no parallel HDF5 in this stage. No commit or push.

## Active A100 run, 2026-10-07

Previous goal turn classification: progress (implementation and local regressions).
The current goal authorizes the actual published 512^3 fiducial on Merlin6 A100.
The user now has Globus access on this Mac; the authenticated session must be
made available to browser automation. Build and GPU validation continue.
Do not substitute synthetic ICs for the objective or claim full-size success from
small tests. No parallel runtime HDF5 and no commit/push are authorized here.

Fresh cluster evidence at 06:04 UTC: merlin-g-100 is MIXED, with eight A100-SXM4
40 GB GPUs, four assigned to other jobs and four available. User queue was empty.
/data/user has 192 TiB free. Historical GPU-down notes are superseded by this
live scheduler evidence. Preserve previous CPU/A100/A11 campaigns and sources.

Campaign root: /data/user/adelmann/quijote-fiducial-512-20261007.
A dedicated Python 3.11 environment is installed there: NumPy 2.4.6, pandas 3.0.3,
SciPy 1.17.1, matplotlib 3.10.9, h5py 3.16.0, hdf5plugin 7.1.0. Prior CPU Python
is unchanged. Build output uses build_cuda, separate from action evidence build/.

Frozen source snapshot: 427 build/runtime/test/doc files; SHA256
c0ffcb03ddc3090898930787cf3cafa91cf7b0b3c3695d1786d14724b1ff4a4b.
Includes new untracked Quijote headers, Python and tests as well as all core src
and CMake modules. Base Git commit a2fefe0aee48853517e193a8f96e52a3f5a0e153.
Source and controller archives were verified and the CUDA build was submitted
exactly once as gmerlin6 job 354388. Deployment SSH session 26198 exited zero.
Configuration SHA256: 01bfbe2bf82f74637b36e123282c9c9f99d3f9d819d09360a6159ce650d6fffd.
Do not duplicate submission if observation times out; inspect job 354388 and
campaign submission.json. Validation is not yet submitted.

Local source additions for current release: offline serial HDF5-to-canonical
conversion (not parallel simulation I/O), with the official lossless Blosc
conventions, source hashes and CompressionInfo; MPI2/OpenMP2 launcher allowance
within four total allocated CPUs. IO 35/35 and launcher 18/18 tests pass, including
real compressed files and independent format-1 equivalence. Python discovery ran
315 tests: the only failure was a sandbox /dev/fd permission in an old launcher;
that entire 11-test file passed with escalation. Strict Doxygen passes with
77 files/878 Python declarations, zero warnings or missing contracts.

Reviewed controller artifacts have separate build, validate and run phases.
Build requests one A100/four CPUs using the accepted A11 allocation syntax;
validation/production request four A100s, four MPI ranks/one CPU each, 128 GiB
host memory. All use CUDA-aware UCX settings from the previously successful
A11 retry, and require actual Cuda execution/CudaSpace metadata and GPU bindings.
Scheduler dry-run accepted both final build and validation layouts. An initial
build dry-run combining one task/four CPUs with --ntasks-per-core=2 was rejected;
no job was submitted by any of those dry-runs.

Before production, run the actual full-count IC through one z127->126 KDK step,
full snapshots, and an exact chunked initial x,p-by-ID comparison. Only then run
512 base steps to z0 with z1,0.5,0 outputs. Source, executable, transfer receipt,
converter and snapshot hashes are preserved. A peer MPI migration message must
fit INT_MAX: only the actual full-count pilot can qualify this redistribution.
No numerical tolerance or particle/mesh/step count is lowered after failure.

Data source required: official Rusty Globus collection
e0eae0aa-5bca-11ea-9683-0e56c063f437, /Snapshots/fiducial/0/ICs/.
Expected eight lossless HDF5 IC shards comes from the official compression
workflow; live listing and actual transfer size remain unverified. The user
provided Globus access on this computer. Codex browser initially lacked a session;
the user is enabling Chrome computer access. Native computer control initially
reported permission unavailable. Do not extract browser authentication tokens. Header matches cannot distinguish
fiducial from fiducial_ZA; retain an authenticated transfer receipt and original
source files. The offline HDF5 reader intentionally rejects lossy evolved
reference snapshots. Real data not yet downloaded; no 512^3 job has run.

Local durable deployment evidence: build_openmp/demos/cosmology/quijote-merlin-20261007.
Next: inspect the single build submission, follow that job, and submit validation
once the matching build completes. Production waits for the actual IC receipt.
No blocked audit is due while these independent actions remain available.

Latest observation: build 354388 RUNNING at 7m34s, compiling Cosmology after
libippl completed. The integration agent alone owns submitting the already
deployed submit_validation.py once sacct confirms COMPLETED 0:0; do not duplicate
that submission from root. Catalogue agent is preparing a bounded Slurm offline
conversion helper and receipt generation, without submitting or altering the
frozen source snapshot. Root owns the Globus browser interaction.

## Latest data-access and GPU status

Build 354388 COMPLETED 0:0 (10m58s). Validation 354389 was submitted exactly
once by the integration agent and COMPLETED 0:0 (2m21s). Host physics, CUDA
spectral/migration on ranks 1/2/3/4 and all external IC launches passed (14 launches,
15 checks). All seven native audits report Cuda execution and Cuda memory;
MPI2/OpenMP2 pipeline verified. Build manifest SHA256:
c1f96d53c7e358966598180efcb2f752114d43580d34f5fc713f519e18c7a1a5.

Chrome native control is now enabled. Authenticated source listing verified
Quijote_simulations2, official collection/path above, eight ics.0.hdf5 through
ics.7.hdf5 plus 2LPT.param and backup. Displayed HDF5 sizes total about 2.1 GB;
exact byte sizes and hashes must come from downloaded files. PSI Merlin destination
collection e62bc2a9-3d92-49bb-a979-349055ca5247 exists but requires a separate
PSI OIDC identity link. User explicitly chose DOWNLOAD VIA THIS MAC then SSH copy
instead. Do not pursue the PSI destination identity link.

Browser Download for 2LPT.param opened a separate Quijote collection login tab.
It requires fresh PSI/CILogon authentication and a Continue button accepting
Globus terms; user has been asked to handle this concrete sign-in step. No files
downloaded yet. Source listing access alone is not download completion. Native
Chrome handle: globusChrome in cua_repl; use fresh AX state before actions.
Codex in-app browser tab was closed by user; it is not the active data route.

Reviewed conversion helper frozen in /private/tmp/quijote-merlin-conversion;
seven tests and independent review passed, existing controller receipt/build
identity compatibility checked. Serial CPU conversion in a one-GPU Slurm
allocation (GPU unused), 32 GiB host, bounded chunks. Original files preserved.
Integration agent owns staging helper artifacts to campaign/conversion-helper
and creating received/fiducial/0/ICs plus acquisition-evidence directories, with
scheduler test-only and no conversion submission. Root owns download, actual
hash evidence, SSH data staging and later conversion submission.

Conversion staging now complete: seven helper artifacts verified on Merlin,
received/fiducial/0/ICs and acquisition-evidence fresh and empty. Scheduler
sbatch --test-only accepted one task, 32 GiB, one GPU (unused by serial CPU
conversion; reports two allocated processors). No conversion job submitted.
Build/validation manifests, exit files, sacct and staging evidence are copied
to the durable local deployment evidence directory with conversion helpers.
Next required user action: finish the Quijote source browser-download login
currently open in Chrome; root must then verify the first download and acquire
the eight actual IC shards. Full goal remains active and incomplete. This turn
made meaningful build, validation and staging progress; no blocked audit applies.

## Goal continuation: source login succeeded; user handling downloads

Previous goal turn classified as progress (CUDA build/validation completed and
conversion staged). Fresh Chrome evidence shows a completed 2LPT.param download
(3.7 KB) and the authenticated source listing. User now says the file will be
available soon; root left browser downloads to the user and asked the exact
Mac folder or Merlin path. Do not interfere with their ongoing browser actions.
No particle file has yet been verified or staged. Desktop/Downloads/Documents
enumeration is denied by macOS privacy, including under escalation, so earlier
empty glob output did NOT prove absence. No browser internals were read.
Finder UI access was not approved; do not retry it without user authorization.
Next: use the supplied accessible file path, inspect actual 2LPT.param and all
eight HDF5 IC files, preserve acquisition evidence and hashes, copy to Merlin,
then submit the prepared conversion job. No conversion/512^3 job running yet.
The source-login condition changed this turn; goal remains active awaiting the
actual particle-file handoff.

## User-supplied Merlin destination

User supplied merlin6:\data\adelmann\quijote-fiducial-512-20261007\ics-quijote-fiducial.
Neither /data/adelmann/... nor the intended /data/user/adelmann/... leaf existed.
A bounded parent listing found the empty directory
/data/user/adelmann/dataadelmannquijote-fiducial-512-20261007ics-quijote-fiducial.
It appears the shell consumed backslashes. Preserve this user-created directory.
Root created and verified the correct destination, currently empty:
/data/user/adelmann/quijote-fiducial-512-20261007/ics-quijote-fiducial/.
User was given the exact SCP destination using forward slashes and /user/.

Accepted companion helper adds truthful user-staged-remote acquisition metadata:
remote staged hashes only, transfer unobserved, no pre-transfer hash comparison.
Eight tests pass, including existing controller receipt/build-identity compatibility.
Original frozen source/controller and original helper remain unchanged. New helper
artifacts retained at build_openmp/demos/cosmology/quijote-merlin-20261007/
conversion-helper-user-staged; manifest SHA256
7989533609787e6b59da33c12b3b478d07a5efb12a2c9efcdd9fbdf7a6c9c020.
Root is staging these to the separate remote conversion-helper-user-staged
directory. No real-data receipt, conversion, pilot, or production submission yet.

## Real IC upload started; goal active

Correct input directory now has ics.0.hdf5=273058501 bytes and a growing
ics.1.hdf5 (last observed124223488 bytes). Earlier ics.0 grew from109805568
to231309312 before reaching its listed source size. Transfer is making real
progress; do not submit conversion until all eight originals are complete.
Catalogue agent is reading only first-shard headers/compression metadata;
root owns remaining transfer checks and eventual conversion submission.
New remote conversion-helper-user-staged was copied and all seven artifact
hashes verified against manifest7989533609787e6b59da33c12b3b478d07a5efb12a2c9efcdd9fbdf7a6c9c020.
No code rebuild needed; frozen source/controller remain unchanged.

User asked exactly which tests ran on Merlin. Root explained CPU background
physics;16^3 CUDA spectral+attribute migration on ranks1/2/3/4;512-particle
external suite14launches/15checks, including independent NumPy KDK, CSV
adapter agreement, ballistic output epochs, seven malformed-input failures,
and split synthetic Gadget MPI2/OpenMP2 with real/RSD self-comparison.
Measured force errors2.11--2.22e-15 versus1e-11; imaginary8.88e-16; migration0
versus1e-12. These are correctness gates, not Quijote accuracy, BAO, refinement
or performance results. Local external-results.json was fetched and SHA256
verified against the validation completion record, then durably copied.
