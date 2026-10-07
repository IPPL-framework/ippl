# Native 1LPT implementation and validation

Updated 2026-10-07. Implementation, scientific validation and verified figure delivery complete.

## Source and scope
- Authorized replacement of Python IC preparation in /Users/adelmann/git/ippl-cosmology-linear, branch codex/cosmology-linear, base 11ec658c1. No commits/pushes made. Existing untracked doxygen/html preserved.
- Native distributed FFT/lattice/1LPT now supports opt-in mode_hash_v1 SHA256 mode-keyed Gaussian draws and a common spherical integer-mode cutoff. Both amplitudes and phases refer to physical Fourier modes independently of grid/rank layout. No cutoff power renormalization.
- Optional float32 momentum rounding reproduces the saved campaign; double remains the default. Legacy RNG remains the default for old inputs.
- ic_only writes checked initial state with zero KDK steps; initial force/diagnostics still evaluated. Native generation requires no Python IC preparation.
- Always-on pre-evolution checks write ic_check.json and pk_initial.csv: counts, finite/bounded phase space, weights/physical mass, ID signatures/native lattice, Fourier support/reality, 1LPT momentum, independently recovered and declared selected complex modes. Hard failure stops before the first force solve. Gaussian shell scatter is diagnostic only.
- pk_initial.csv is the initial LINEAR displacement-field spectrum, not CIC particle power. ID signatures detect corruption but are not a uniqueness proof; independent saved-data validation checks complete ID coverage exactly.
- No force/integration algorithm change. New mode choices deliberately change the realization; cross-backend FFT/transcendental/reduction rounding is not bitwise identical. Normalization, unit PM weights and physical mass conventions are preserved.
- Device particle/field arrays remain distributed. Validation uses device reductions and short host reports; no extra complete particle/mesh gather in the production gate.

## Changed files
- New: CosmologyICRandom.h, CosmologyICCheck.hpp, input/native-common-field.par, TestCosmologyICRandom.cpp, TestCosmologyICCheck.cpp, python/validate_native_ic.py and test_native_ic_validation.py.
- Updated: Config/Simulation integration, CMake, physics/config tests, Merlin CPU/A100 launchers and GPU target allowlist/tests, README and mathematical Doxygen docs.
- Example: demos/cosmology/input/native-common-field.par. Set ic_only=false to evolve.

## Verified on Merlin
Isolated root: /data/user/adelmann/ok-check/native-ic-20261007. All builds/scientific calculations/tests/plots on Merlin. Mac only authored, verified transport hashes and viewed results; Markdown edits local.
- CPU/OpenMP and CUDA builds passed with Kokkos5.2.0/heFFTe2.4.1, GCC14.3, OpenMPI5.0.10 and CUDA12.9.1.
- Twelve selected CPU regressions passed, including physics/config, spectral1–4r, linear evolution, external IC and independent analysis tests. Eight final CPU gate/launcher/analysis tests passed (overlap with first set).
- CUDA RNG oracle vectors passed; production IC gate and spectral self-tests passed at 1/2/4 GPU ranks with distinct binding records. Gate tests accept five valid modes and reject eight deliberately corrupted states collectively before force evaluation.
- Native saved-IC comparisons all passed: 64^3 CPU MPI1/OpenMP1 and MPI2/OpenMP2 plus CUDA MPI1/2/4; 128^3 CUDA MPI1/2; 256^3 CUDA MPI1. Complete IDs, immutable input hashes, all retained Fourier coefficients and momentum rounding audited independently.
- Worst initial coefficient relative L2 error: 1.0122191924493611e-12, versus inherited unchanged 1e-8 tolerance. Initial position RMS differences approximately 1.08e-13 Mpc/h.
- Native64^3 and128^3 each evolved2400steps on one A100 to a=1/z=0, exact counts and finite diagnostics. Compared against saved runs with the same configuration.
- Matched z=0 raw spectra use common256^3 CIC analysis, no shot-noise subtraction. Maximum measured relative differences across entire shells below both Nyquist limits: 5.150605e-9 (64) and4.628674e-9 (128), equivalently5.150605e-7% and4.628674e-7%. These measure initializer equivalence, not physical accuracy or convergence.
- Near k=0.3145615 h/Mpc: changes -1.080622e-8% (64), -1.101760e-7% (128). Near k=0.9860688: -7.437866e-8% (64), +1.517312e-7% (128).
- Independent reviewer rehashed all five saved reference particle files against original provenance and verified source/estimator/metadata/receipt consistency. Final nonlinear comparisons remain descriptive: no new acceptance threshold invented.
- Strict Doxygen passed83files/920Python declarations, no warnings/missing contracts. Final diff reviewed; git diff --check passed.

## Jobs and recovered infrastructure failures
- 354431 compiled CPU targets; MPI2+ launches hit one-task Slurm slot discovery. 354432 recovered using --bind-to none --oversubscribe within eight allocated CPU cores, passed tests and CUDA build; exit0:0.
- 354433 passed all GPU tests and eight native runs, then failed importing missing pandas in analysis interpreter. No scientific comparison ran in that failing command; no simulation rerun.
- 354434 CPU-only continuation passed all initial/final comparisons and generated figures, exit0:0. Isolated analysis-extra adds pandas2.2.3/Matplotlib3.9.4 while preserving NumPy1.26.4/SciPy1.14.1/Pylians0.12 estimator versions.
- 354435 completed 0:0 in 19 seconds and improves only residual-axis presentation to parts per billion. Original figures preserved in figures/; revised figures in figures-v2/. Numerical results and all four validation inputs are unchanged.

## Evidence and limits
- Local /Users/adelmann/git/ok-check/native-ic-20261007: 117 small evidence files transported and SHA256 verified. Archive ac46c16a989a36076e4aebbd9b2f7dfcacd224c0fb75e501a426e259ef37ed26. qualification.json retains original failed analysis stage; qualification-analysis.json records successful continuation.
- Primary code/docs stayed in cosmology worktree. Existing production source/executables and prior science outputs unchanged.
- 256^3 qualification is IC-only. Native512^3/1024^3 capacity/evolution are not qualified by this matrix. This is1LPT, not2LPT; no physical force-resolution accuracy claim follows from matching the saved initializer.
- Earlier frozen external-IC1024^3 attempt354430 failed before evolution with a6GiB particle-attribute allocation failure on four A100s. Evidence retained separately in ok-check/large/failure-1024-gpu4; no1024curve fabricated. That result delivered and heartbeat PAUSED. Quijote downloads remain stopped, older heartbeat paused.

## Completion and delivery
The temporary PSI DNS interruption was resolved after the user restored the network. Revised PNG/PDF, rendered PDF preview, script, Slurm receipt and provenance were retrieved and all eight file checksums verified. Figure archive SHA256: e1a1598a25896a2e24408e0e6ec4eeb53e03dd455bd35a4e38417e6d8fe83933.

Visual inspection of both the PNG and PDF render passed: axes/units/legends/captions readable, clipped label repaired, residual scale explicitly stated in parts per billion. Receipts: FIGURE_V2_TRANSPORT_VERIFIED.json and FIGURE_V2_VISUAL_QA.json in the local evidence root. Deliver figures-v2/native_ic_validation.png and .pdf. No further work remains for this implementation; new source changes remain uncommitted.
