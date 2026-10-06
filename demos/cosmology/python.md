# Cosmology Python tools

All cosmology Python sources live under [`python/`](python/), including tests, reference probes, cluster helpers, GADGET-2 tools, and paper asset preparation. Subdirectories retain their previous grouping. Run the commands below from the repository root.

Use `/Users/adelmann/.venv-h6/bin/python` for these tasks. Numerical tools require NumPy and pandas; plotting tools also require Matplotlib. Simulation validators require the corresponding compiled executables and an MPI launcher. Cluster tools require their documented Merlin configuration and saved campaign inputs.

Commands use `python` as shorthand for that environment. Uppercase argument values are placeholders. Use `--help` on argparse tools for optional settings; use fresh output locations where required. Saved scientific reports and manifests are preserved as historical evidence, including their original paths and hashes. Moving or editing a source invalidates strict provenance checks against a campaign that froze its previous path/hash; retain the original checkout for those audits.

## Validation, analysis, and plotting

### [`analyze_zeldovich_benchmark.py`](python/analyze_zeldovich_benchmark.py)

**Purpose:** Audit saved broadband IPPL/FastPM snapshots and compute CIC power spectra.

**Usage:**

```sh
python -B demos/cosmology/python/analyze_zeldovich_benchmark.py --campaign CAMPAIGN_JSON --output NEW_ANALYSIS_JSON
```

### [`compare_fixture_files.py`](python/compare_fixture_files.py)

**Purpose:** Read-only cross-host comparison of common-particle cosmology fixture CSVs.

**Usage:**

```sh
python -B demos/cosmology/python/compare_fixture_files.py ORIGINAL COMPARISON --box-size BOX_SIZE --output OUTPUT
```

### [`gaussian_fixture.py`](python/gaussian_fixture.py)

**Purpose:** Generate common-phase, band-limited Gaussian 1LPT fixtures for matched particle studies.

**Usage:**

```sh
PYTHONPATH=demos/cosmology/python python -c "import gaussian_fixture"
```

Importable helper module; used by the validators and regression tests. No standalone command-line interface.

### [`plot_evolution.py`](python/plot_evolution.py)

**Purpose:** Render two static summaries of a completed matched-evolution campaign.

**Usage:**

```sh
python -B demos/cosmology/python/plot_evolution.py RESULTS --output-dir OUTPUT_DIR
```

### [`plot_pm_validation.py`](python/plot_pm_validation.py)

**Purpose:** Static scientific plots of saved PM-force and pre-crossing pancake evidence.

**Usage:**

```sh
python -B demos/cosmology/python/plot_pm_validation.py --frozen FROZEN --pancake PANCAKE --audit AUDIT --output-dir OUTPUT_DIR
```

### [`plot_resolution_study.py`](python/plot_resolution_study.py)

**Purpose:** Plot recorded resolution-study evidence; never open snapshots or run models.

**Usage:**

```sh
python -B demos/cosmology/python/plot_resolution_study.py --report REPORT --output-dir OUTPUT_DIR
```

### [`plot_zarija.py`](python/plot_zarija.py)

**Purpose:** Reproduce static scientific figures from an existing matched-Zarija campaign.

**Usage:**

```sh
python -B demos/cosmology/python/plot_zarija.py --campaign CAMPAIGN --physics PHYSICS --output-dir OUTPUT_DIR
```

### [`runtime_metadata.py`](python/runtime_metadata.py)

**Purpose:** Parse and validate execution-space, memory-space, rank, and thread metadata.

**Usage:**

```sh
PYTHONPATH=demos/cosmology/python python -c "import runtime_metadata"
```

Importable helper module; used by the validators and regression tests. No standalone command-line interface.

### [`source_paths.py`](python/source_paths.py)

**Purpose:** Resolve moved Python sources and unmoved C++/shell sources for provenance checks.

**Usage:**

```sh
PYTHONPATH=demos/cosmology/python python -c "import source_paths"
```

Importable helper module; used by the validators and regression tests. No standalone command-line interface.

### [`study_spectra.py`](python/study_spectra.py)

**Purpose:** Measure particle density Fourier coefficients and independent interlaced PCS power spectra.

**Usage:**

```sh
PYTHONPATH=demos/cosmology/python python -c "import study_spectra"
```

Importable helper module; used by the validators and regression tests. No standalone command-line interface.

### [`study_storage.py`](python/study_storage.py)

**Purpose:** Manage disk budgets, execution journals, locks, and numerical particle archives.

**Usage:**

```sh
PYTHONPATH=demos/cosmology/python python -c "import study_storage"
```

Importable helper module; used by the validators and regression tests. No standalone command-line interface.

### [`validate_evolution.py`](python/validate_evolution.py)

**Purpose:** Matched imported-particle evolution: IPPL versus pinned native plain-PM FastPM.

**Usage:**

```sh
python -B demos/cosmology/python/validate_evolution.py --ippl-exe IPPL_EXE --fastpm-exe FASTPM_EXE
```

### [`validate_evolution_refinement.py`](python/validate_evolution_refinement.py)

**Purpose:** Opt-in 128/256/512-step follow-up of the completed coupled3d evolution case.

**Usage:**

```sh
python -B demos/cosmology/python/validate_evolution_refinement.py --parent PARENT
```

### [`validate_frozen_force.py`](python/validate_frozen_force.py)

**Purpose:** Frozen CIC/PM qualification; never advances particles or fits a normalization.

**Usage:**

```sh
python -B demos/cosmology/python/validate_frozen_force.py --ippl-exe IPPL_EXE
```

### [`validate_linear.py`](python/validate_linear.py)

**Purpose:** Reproducible CPU/MPI validation of the IPPL linear cosmology demonstration.

**Usage:**

```sh
python -B demos/cosmology/python/validate_linear.py --exe build_openmp/demos/cosmology/Cosmology --quick
```

### [`validate_pancake.py`](python/validate_pancake.py)

**Purpose:** Validate finite-amplitude planar collapse before shell crossing.

**Usage:**

```sh
python -B demos/cosmology/python/validate_pancake.py --exe EXE
```

### [`validate_resolution_study.py`](python/validate_resolution_study.py)

**Purpose:** Disk-guarded crossed-resolution and common-phase Gaussian PM studies.

**Usage:**

```sh
python -B demos/cosmology/python/validate_resolution_study.py \
  --ippl-exe build_openmp/demos/cosmology/CompareCosmologyEvolution \
  --fastpm-exe build_fastpm/evolution/FastPMEvolution \
  --fastpm-manifest build_fastpm/evolution/build-manifest.txt \
  --stage all --smoke --output-dir NEW_OUTPUT_DIR
```

### [`validate_zarija.py`](python/validate_zarija.py)

**Purpose:** Compare actual Zarija and IPPL Gaussian 1LPT initial conditions.

**Usage:**

```sh
python -B demos/cosmology/python/validate_zarija.py --ippl-exe IPPL_EXE --zarija-exe ZARIJA_EXE --zarija-source ZARIJA_SOURCE
```

## Regression tests and diagnostic analysis

Run the complete Python regression suite with:

```sh
python -B -m unittest discover -s demos/cosmology/python/tests -p 'test_*.py'
```

The frozen/evolution adapter scripts also run compiled binaries; their individual commands below supply the executable arguments.

### [`tests/analyze_pancake_diagnostics.py`](python/tests/analyze_pancake_diagnostics.py)

**Purpose:** Read-only follow-up of saved pancake mass and local-gradient diagnostics.

**Usage:**

```sh
python -B demos/cosmology/python/tests/analyze_pancake_diagnostics.py --campaign CAMPAIGN --output-dir OUTPUT_DIR
```

### [`tests/test_a100_completion_audit.py`](python/tests/test_a100_completion_audit.py)

**Purpose:** Regression checks for a100 completion audit. Synthetic evidence contract checks; no hardware or physics pass is fabricated.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_a100_completion_audit.py
```

### [`tests/test_analyze_zeldovich_benchmark.py`](python/tests/test_analyze_zeldovich_benchmark.py)

**Purpose:** Regression checks for analyze zeldovich benchmark. Tests for periodic CIC power normalization and compression integrity helpers.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_analyze_zeldovich_benchmark.py
```

### [`tests/test_compare_fixture_files.py`](python/tests/test_compare_fixture_files.py)

**Purpose:** Regression checks for compare fixture files. Small synthetic fixture comparison tests; no MPI or physical tolerances.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_compare_fixture_files.py
```

### [`tests/test_cpu_completion_audit.py`](python/tests/test_cpu_completion_audit.py)

**Purpose:** Regression checks for cpu completion audit. CPU audit contract tests with synthetic reports and tiny numerical archives.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_cpu_completion_audit.py
```

### [`tests/test_evolution_adapter.py`](python/tests/test_evolution_adapter.py)

**Purpose:** Regression checks for evolution adapter. Imported-evolution adapter checks: input integrity, KDK reuse and MPI1--4.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_evolution_adapter.py --exe EXE --production-exe PRODUCTION_EXE
```

### [`tests/test_evolution_refinement.py`](python/tests/test_evolution_refinement.py)

**Purpose:** Regression checks for evolution refinement. Synthetic provenance and numerical-analysis tests; no MPI or simulations.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_evolution_refinement.py
```

### [`tests/test_frozen_adapter.py`](python/tests/test_frozen_adapter.py)

**Purpose:** Regression checks for frozen adapter. MPI input/output regressions for CompareCosmologyForce.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_frozen_adapter.py --exe EXE
```

### [`tests/test_gaussian_fixture.py`](python/tests/test_gaussian_fixture.py)

**Purpose:** Regression checks for gaussian fixture. Independent synthesis, statistics, units, and provenance checks; no simulations.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_gaussian_fixture.py
```

### [`tests/test_merlin_a100_launcher.py`](python/tests/test_merlin_a100_launcher.py)

**Purpose:** Regression checks for merlin a100 launcher. A100 allocation guards; no modules or device queries run in these tests.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_merlin_a100_launcher.py
```

### [`tests/test_merlin_cpu_launcher.py`](python/tests/test_merlin_cpu_launcher.py)

**Purpose:** Regression checks for merlin cpu launcher. Launcher preflight tests; never run modules, builds, MPI, or network calls.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_merlin_cpu_launcher.py
```

### [`tests/test_merlin_gpu_mpiexec.py`](python/tests/test_merlin_gpu_mpiexec.py)

**Purpose:** Regression checks for merlin gpu mpiexec. Pure/mocked launcher checks. No MPI, GPU, Slurm command, or network call.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_merlin_gpu_mpiexec.py
```

### [`tests/test_merlin_gpu_rank.py`](python/tests/test_merlin_gpu_rank.py)

**Purpose:** Regression checks for merlin gpu rank. GPU binding tests with fake CUDA functions and exec; no GPU is queried.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_merlin_gpu_rank.py
```

### [`tests/test_plot_evolution.py`](python/tests/test_plot_evolution.py)

**Purpose:** Regression checks for plot evolution. Small mathematical/provenance regression tests for the static summary.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_plot_evolution.py
```

### [`tests/test_plot_pm_validation.py`](python/tests/test_plot_pm_validation.py)

**Purpose:** Regression checks for plot pm validation. Focused numerical extraction tests for saved-evidence PM figures.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_plot_pm_validation.py
```

### [`tests/test_plot_resolution_study.py`](python/tests/test_plot_resolution_study.py)

**Purpose:** Regression checks for plot resolution study. Bounded synthetic report tests; no executable, particle archive or MPI I/O.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_plot_resolution_study.py
```

### [`tests/test_resolution_study.py`](python/tests/test_resolution_study.py)

**Purpose:** Regression checks for resolution study. Independent matrix, comparison, qualification and control-flow checks.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_resolution_study.py
```

### [`tests/test_runtime_metadata.py`](python/tests/test_runtime_metadata.py)

**Purpose:** Regression checks for runtime metadata. CPU/GPU execution metadata regressions; no device or simulation required.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_runtime_metadata.py
```

### [`tests/test_study_spectra.py`](python/tests/test_study_spectra.py)

**Purpose:** Regression checks for study spectra. Synthetic independent density-estimator tests; no simulations or MPI.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_study_spectra.py
```

### [`tests/test_study_storage.py`](python/tests/test_study_storage.py)

**Purpose:** Regression checks for study storage. Temporary, mocked execution tests. No MPI, simulation, or previous data use.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_study_storage.py
```

### [`tests/test_validate_evolution.py`](python/tests/test_validate_evolution.py)

**Purpose:** Regression checks for validate evolution. Independent synthetic/corruption tests for matched-particle evolution analysis.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_validate_evolution.py
```

### [`tests/test_validate_frozen_force.py`](python/tests/test_validate_frozen_force.py)

**Purpose:** Regression checks for validate frozen force. Analytical tests of the frozen-force oracle; no simulation binaries needed.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_validate_frozen_force.py
```

### [`tests/test_validate_pancake.py`](python/tests/test_validate_pancake.py)

**Purpose:** Regression checks for validate pancake. Independent analytic pancake oracle tests; no simulation or MPI required.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_validate_pancake.py
```

### [`tests/test_validate_zarija.py`](python/tests/test_validate_zarija.py)

**Purpose:** Regression checks for validate zarija. Fast analytic tests of the independent cross-generator analysis (no MPI).

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_validate_zarija.py
```

## External-reference physics probes

### [`reference/compare_zarija_physics.py`](python/reference/compare_zarija_physics.py)

**Purpose:** Build and run public-API probes against unmodified Zarija sources.

**Usage:**

```sh
python -B demos/cosmology/python/reference/compare_zarija_physics.py --reference-source REFERENCE_SOURCE --ippl-build IPPL_BUILD
```

## Merlin launch and completion tools

### [`merlin/audit_a100_study.py`](python/merlin/audit_a100_study.py)

**Purpose:** Verify complete A100/CPU study scope and saved hashes, not physical acceptance.

**Usage:**

```sh
python -B demos/cosmology/python/merlin/audit_a100_study.py GPU_REPORT --cpu-report CPU_REPORT --launch-evidence LAUNCH_EVIDENCE --output OUTPUT
```

### [`merlin/audit_cpu_study.py`](python/merlin/audit_cpu_study.py)

**Purpose:** Audit a completed 18-run CPU Gaussian study without rerunning simulations.

**Usage:**

```sh
python -B demos/cosmology/python/merlin/audit_cpu_study.py REPORT --source-dir SOURCE_DIR --output OUTPUT
```

SOURCE_DIR may be the cosmology directory or its python directory. Audits saved integrity and completion; a successful audit does not imply scientific acceptance.

### [`merlin/gpu_mpiexec.py`](python/merlin/gpu_mpiexec.py)

**Purpose:** Strict, evidence-producing MPI launcher for the single-node Merlin campaign.

**Usage:**

```sh
python -B demos/cosmology/python/merlin/gpu_mpiexec.py --config CONFIG_JSON -n RANKS TARGET [TARGET_ARGUMENTS ...]
```

Strict configured MPI launcher; see the module docstring for the required config schema.

### [`merlin/gpu_rank.py`](python/merlin/gpu_rank.py)

**Purpose:** Bind one OpenMPI local rank to one scheduler-visible GPU, then exec.

**Usage:**

```sh
python -B demos/cosmology/python/merlin/gpu_rank.py TARGET [TARGET_ARGUMENTS ...]
```

Invoked per MPI rank by gpu_mpiexec on an allocated GPU node.

## GADGET-2 comparisons

### [`gadget2/analyze_gadget2_np256.py`](python/gadget2/analyze_gadget2_np256.py)

**Purpose:** Verify a completed matched Gadget-2 256^3 run and measure its z=0 spectrum.

**Usage:**

```sh
python -B demos/cosmology/python/gadget2/analyze_gadget2_np256.py --campaign CAMPAIGN_JSON --output NEW_ANALYSIS_JSON
```

### [`gadget2/convert_shared_ic.py`](python/gadget2/convert_shared_ic.py)

**Purpose:** Convert the frozen shared cosmology CSV IC to GADGET-2 format-1.

**Usage:**

```sh
python -B demos/cosmology/python/gadget2/convert_shared_ic.py --input INPUT --output OUTPUT --redshift REDSHIFT --expected-sha256 EXPECTED_SHA256
```

### [`gadget2/plot_three_code_np256.py`](python/gadget2/plot_three_code_np256.py)

**Purpose:** Plot matched 256^3-particle IPPL, FastPM, and GADGET-2 z=0 spectra.

**Usage:**

```sh
python -B demos/cosmology/python/gadget2/plot_three_code_np256.py --campaign CAMPAIGN_JSON --analysis ANALYSIS_JSON --output-dir NEW_PLOT_DIR
```

### [`gadget2/run_gadget2_np256.py`](python/gadget2/run_gadget2_np256.py)

**Purpose:** Run an audited GADGET-2 TreePM 256^3-particle/mesh comparison on Merlin.

**Usage:**

```sh
python -B demos/cosmology/python/gadget2/run_gadget2_np256.py
```

Cluster campaign controller; defaults refer to the fixed Merlin shared campaign. Add --login for login-node submission.

## Paper assets

### [`ippl-cosmology/scripts/prepare_validation_assets.py`](python/ippl-cosmology/scripts/prepare_validation_assets.py)

**Purpose:** Copy nine released validation PNGs without rerendering or changing evidence.

**Usage:**

```sh
python -B demos/cosmology/python/ippl-cosmology/scripts/prepare_validation_assets.py --validation-root VALIDATION_ROOT
```


## Local Figure A11 campaign

### [`run_three_code_campaign.py`](python/run_three_code_campaign.py)

**Purpose:** Build missing FastPM, GADGET-2 and Cosmology executables; run matched local 128³ evolution; produce Figure A11 and its plotted data.

**Usage:**

```sh
python -B demos/cosmology/python/run_three_code_campaign.py --ranks 4
```

Use `--build-only`, `--plan`, `--smoke`, or `--analyze-only CAMPAIGN` for individual stages. See [ThreeCodeCampaign.md](ThreeCodeCampaign.md) for model, options and outputs.

### [`tests/test_three_code_campaign.py`](python/tests/test_three_code_campaign.py)

**Purpose:** Check shared IC conventions, cached-build behavior, final snapshot integrity, and the Figure A11 FastPM ratio denominator.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_three_code_campaign.py
```

### [`build_documentation.py`](python/build_documentation.py)

**Purpose:** Build the strict Doxygen scientific/API manual, reject generator warnings and check extracted source/API/parameter coverage.

**Usage:**

```sh
python -B demos/cosmology/python/build_documentation.py
```

### [`doxygen_filter.py`](python/doxygen_filter.py)

**Purpose:** Normalize bare required Python annotations only for the documentation parser; it never executes transformed text or changes the original source.

**Usage:**

```sh
python -B demos/cosmology/python/doxygen_filter.py SOURCE_PYTHON_FILE
```

### [`tests/test_documentation.py`](python/tests/test_documentation.py)

**Purpose:** Check source-preserving parser normalization, class scope identity and strict failure on missing API/parameter contracts.

**Usage:**

```sh
python -B demos/cosmology/python/tests/test_documentation.py
```
