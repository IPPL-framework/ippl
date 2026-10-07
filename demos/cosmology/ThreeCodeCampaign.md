# Local and Merlin6 campaign (Figure A11)

## Purpose/Model

[python/run_three_code_campaign.py](python/run_three_code_campaign.py) builds
missing executables, runs a shared-initial-condition comparison of IPPL,
native FastPM plain PM, and GADGET-2 TreePM, then creates the two-panel power
spectrum figure shown as Figure A.11 on page 32 of
[ippl-cosmology-paper/main.pdf](ippl-cosmology-paper/main.pdf).

The default local experiment uses 128³ particles, 128³ force and analysis
meshes, a 168.75 Mpc/h box, seed 20261003, spherical modes 0<|n|≤48, and
Gaussian BBKS 1LPT initial conditions at z=99. The background has
Omega_m=0.31, h=0.675, n_s=0.965, sigma8=0.82 and no radiation.
Both PM integrations take 2,400 steps uniform in log(a) to z=0.
GADGET-2.0.7 uses periodic TreePM with PMGRID=128, double-precision particles
and FFTs, synchronized adaptive timesteps and softening 26.3671875 kpc/h.
Its format-1 input/output position and velocity blocks are float32.

`Cosmology` and its import driver `CompareCosmologyEvolution` are built.
The run uses the import driver, which calls production Cosmology kernels,
so both PM codes receive the same phase-space CSV. GADGET receives an audited
unit conversion of that CSV. The main `Cosmology` executable generates its
own initial conditions and cannot import this shared diagnostic CSV.

All final spectra use the same external periodic-CIC estimator, CIC
power-window correction, Poisson shot-noise subtraction, and radial Fourier
bins. The lower panel shows IPPL/FastPM−1 and GADGET-2/FastPM−1 in percent.
Nonpositive subtracted powers are omitted from logarithmic plots and ratios,
while the unmodified values remain in the data files. This is a single-realization
comparison; its high-k bins are characterization, not a convergence qualification.
TreePM's short-range forces and adaptive steps differ from the two PM models.

This produces a new local reproduction, not the original Merlin measurements
or a byte-identical copy of the paper figure. Existing paper assets and campaign
results are preserved. Only initial and final PM states are saved by default
to limit local disk use; this does not change the integration schedule.

## Build Instructions

Run from the repository root using the OPALX Python environment:

```sh
/Users/adelmann/.venv-h6/bin/python -B \
  demos/cosmology/python/run_three_code_campaign.py --build-only
```

Each existing executable is reused without recompiling it. Default locations:

| Code | Executable |
| --- | --- |
| Cosmology | `build_openmp/demos/cosmology/Cosmology` |
| Shared-IC IPPL driver | `build_openmp/demos/cosmology/CompareCosmologyEvolution` |
| Native FastPM | `build_fastpm/evolution/FastPMEvolution` |
| GADGET-2 | `build_gadget2/Gadget2-TreePM-128-double` |

Missing IPPL targets use CMake. Missing FastPM targets use the existing pinned
[reference builders](reference/build_fastpm.sh). Missing GADGET targets use
[reference/build_gadget2.sh](reference/build_gadget2.sh), which downloads
checksum-pinned GADGET-2.0.7 and MPI FFTW2.1.5, and reuses GSL from the FastPM
installation. It changes no upstream simulation source. FFTW2 uses a generic
ARM configure target on Apple Silicon because its configure scripts predate arm64.

Requirements: a C/C++ toolchain, MPI, CMake, make, curl, NumPy, pandas and
Matplotlib. No system-wide library installation is performed. Builds and
logs stay in the selected local build directories.

Options `--ippl-build`, `--fastpm-build`, `--gadget-build`, `--gsl-prefix`,
`--jobs` and repeated `--cmake-arg=-D...` select other build locations and
fresh-build settings. For a new macOS IPPL build, use the toolchain settings in
[README.md](README.md#build), passing each CMake setting with `--cmake-arg`.
`FASTPM_CC`, `FASTPM_MPICC`, `FASTPM_FFTW_PREFIX`, `GADGET_CC` and
`GADGET_MPICC` select reference compiler/dependency settings.

## Run Instructions

Run the full local 128³ campaign:

```sh
MPLCONFIGDIR=/tmp/ippl-cosmology-mpl \
  /Users/adelmann/.venv-h6/bin/python -B \
  demos/cosmology/python/run_three_code_campaign.py --ranks 4
```

Executables are built only if missing. The three solver runs execute
sequentially, with one OpenMP thread per rank. Output goes into a fresh
`demos/cosmology/results/a11-local-128-*` directory. Use `--output NEW_DIRECTORY`
for a specific destination; an existing destination is rejected.
Use `--mpi-arg=--oversubscribe` if required by a small Open MPI machine.
`--timeout` sets the per-solver timeout in seconds (default 86,400).
The disk preflight includes a configurable `--reserve-gib` (default 2 GiB).

Inspect the resolved settings without building or running:

```sh
/Users/adelmann/.venv-h6/bin/python -B \
  demos/cosmology/python/run_three_code_campaign.py --plan
```

For an engineering smoke check with 16³ particles/meshes, cutoff 6 and 24 PM steps:

```sh
/Users/adelmann/.venv-h6/bin/python -B \
  demos/cosmology/python/run_three_code_campaign.py --smoke --ranks 2 --timeout 300
```

The smoke figure is labelled accordingly and is not a 128³ science result.
To retain all 25 original PM checkpoints, add `--checkpoints 24` after checking
available disk space. `--grid`, `--cutoff` and `--steps` can run smaller local
experiments; the local runner currently supports grids up to 128.

Each campaign retains shared ICs and their conversion audit, GADGET parameters,
solver logs, particle outputs, timings, configuration, hashes, and completion
state in `campaign.json`. A zero solver exit alone is insufficient: analysis
checks final particle IDs/counts, finite state and the z=0 endpoint.

Successful output includes:

- `analysis.json`: three spectra and input provenance.
- `figures/figure-A11.png` and `.svg`: the publication-quality two-panel figure.
- `figures/plotted_values.csv`: actual powers and offsets plotted.
- `figures/manifest.json`: input/source/output hashes and estimator conventions.

Verify and regenerate figures from a completed/evolved campaign without rerunning
any solver:

```sh
/Users/adelmann/.venv-h6/bin/python -B \
  demos/cosmology/python/run_three_code_campaign.py \
  --analyze-only /ABSOLUTE/CAMPAIGN_DIRECTORY
```

Strict source/input hashes must still match. Failures are recorded and return a
nonzero exit code. Partial solver outputs are retained; automatic simulation
restart is not implemented. Timeout cleanup applies only to the process group
started by this runner.

## Merlin6 A100 execution

Omitting `--cluster` runs locally. `--rank` defaults to `1`; `--ranks` remains
an alias. A comma-separated `--rank 1,4` is supported for Merlin6 submissions.

From the local repository, reuse the completed baseline's exact IC CSV:

```sh
~/.venv-h6/bin/python -B demos/cosmology/python/run_three_code_campaign.py \
  --cluster merlin6 --rank 1,4 \
  --shared-ic /ABS/BASELINE/ics/shared-z99.csv \
  --output /data/user/adelmann/NEW_A11_CAMPAIGN
```

The CSV requires its sibling `ic-manifest.json`. The controller copies both
through SSH/SCP to Merlin6 and invokes the same runner in `~/git/ippl`.
Use `--remote-root` or `--remote-python` to override the remote checkout and
Python environment. The remote checkout must contain this version of the
runner and `merlin/a11_job.sh`; no automatic Git pull or push is performed.
The same command can also be run directly on Merlin6 with a cluster-local IC.
`--plan` prints settings without SSH, copying, compiling or submitting jobs.

The user-facing cluster name is `merlin6`; its Slurm GPU cluster is
`gmerlin6`, partition/account `gwendolen`. The controller submits a one-A100,
four-CPU build job and dependency-linked single-node GPU runs with one MPI
rank per full A100 and one host thread per rank. CUDA AMPERE80, GCC 14.3,
CUDA 12.9.1, OpenMPI 5.0.10 and pinned Kokkos/heFFTe are selected.
C++ compilation uses the SHA256-recorded Kokkos `nvcc_wrapper` already cached
in `build_a100/_deps/kokkos-src/bin` with GCC as its host compiler. Builds live
in `build_a11_a100`. Cached binaries are reused only when their recorded native compilation
inputs and artifact hashes match. Python/Slurm launcher and tuning-table
hashes are frozen independently for each run; changes to those do not
misrepresent the original binary build or force unnecessary recompilation. A changed cache requires a fresh build
rather than silently accepting a stale executable. Select a fresh cluster-local
build path with `--ippl-build /ABS/NEW_BUILD`; `--nvcc-wrapper` selects another
existing cluster-local Kokkos wrapper. Compilation and simulation
run inside Slurm allocations, never on a login host.

These supplementary jobs run **IPPL only**. The original FastPM and GADGET-2
results remain the reference curves. Spectral self-tests run on each requested
rank count. Each rank's CUDA wrapper narrows the scheduler-provided GPU mask;
The launcher explicitly selects OpenMPI CUDA-aware UCX (`--mca pml ucx`),
with intra-node shared-memory, CUDA-copy and CUDA-IPC transports; it avoids
the site default BTL/UCT host-copy path that failed during four-rank particle
migration. See [OpenMPI CUDA transport documentation](https://docs.open-mpi.org/en/v5.0.x/tuning-apps/networking/cuda.html).
Analysis verifies distinct physical PCI identities, full A100 allocation,
CUDA execution/memory metadata, exact IC hashes, import roundtrip, particle
IDs/counts, finite phase space, mass conservation and the final epoch. This
changes execution backend and floating-point reduction order; no bitwise
CPU/GPU equality or continuum convergence is assumed.

Monitor job IDs from `submission.json` with `squeue -M gmerlin6` and
`sacct -M gmerlin6`. Each completed `gpu1`/`gpu4` directory includes
`gpu-analysis.json`, original rank snapshots, logs and scheduler evidence.
Failed jobs retain their outputs and require an explicit new campaign path;
there is no automatic resubmission.

Download the complete `gpu1` and `gpu4` directories and extend Figure A11:

```sh
~/.venv-h6/bin/python -B demos/cosmology/python/run_three_code_campaign.py \
  --extend-figure /ABS/BASELINE \
  --gpu-result /ABS/DOWNLOADED/gpu1 --gpu-result /ABS/DOWNLOADED/gpu4 \
  --output /ABS/NEW_COMPARISON
```

The merger checks the baseline's recorded analysis/output hashes, exact IC
identity and all downloaded GPU artifact hashes. It writes a new five-curve
Figure A11 and comparison provenance, preserving the original campaign and
figures. All ratios continue to use the original FastPM spectrum.
