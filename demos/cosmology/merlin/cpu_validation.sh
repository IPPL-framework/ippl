#!/bin/bash -l
# Default: run once inside a four-CPU, single-node Slurm allocation.
# Explicit --login: user-authorized Merlin login-host execution, capped at four CPUs.
# Usage: bash -l cpu_validation.sh [--login] ABSOLUTE_NEW_EVIDENCE_DIRECTORY
# Source is the checkout containing this script. Outputs must be on cluster storage.
set -euo pipefail

executionMode=slurm
if [[ ${1:-} == --login ]]; then
    executionMode=login
    shift
fi
if [[ $# != 1 || $1 != /* ]]; then
    printf 'Usage: bash -l %s [--login] ABSOLUTE_NEW_EVIDENCE_DIRECTORY\n' "$0" >&2
    exit 2
fi
if [[ "$executionMode" == login ]]; then
    if [[ -n ${SLURM_JOB_ID:-} ]]; then
        printf 'Refusing --login with an inherited Slurm job allocation.\n' >&2
        exit 2
    fi
    case $(hostname -s) in merlin-l-*) ;; *)
        printf 'Refusing --login outside a Merlin login host.\n' >&2; exit 2;; esac
else
    test -n "${SLURM_JOB_ID:-}"
    test "${SLURM_JOB_NUM_NODES:-0}" -eq 1
    test "${SLURM_NTASKS:-0}" -eq 4
    test "${SLURM_CPUS_PER_TASK:-0}" -eq 1
    test "${SLURM_CPUS_ON_NODE:-999}" -eq 4
    case $(hostname -s) in merlin-c-*|merlin-g-*) ;; *)
        printf 'Refusing a non-compute host.\n' >&2; exit 2;; esac
fi

scriptDir=$(cd "$(dirname "$0")" && pwd -P)
sourceRoot=$(cd "$scriptDir/../../.." && pwd -P)
git -C "$sourceRoot" diff --quiet HEAD --
# Exclusive mkdir prevents accidental reuse or overwrite of an earlier campaign.
mkdir "$1"
evidenceRoot=$(cd "$1" && pwd -P)
exec > >(tee "$evidenceRoot/controller.log") 2>&1
phase=setup
trap 'status=$?; printf "phase=%s exit=%s\n" "$phase" "$status" > "$evidenceRoot/controller-exit.txt"' EXIT

module unload Python/3.14.4
module load gcc/14.3.0 openmpi/5.0.10_slurm cmake/4.4.0 Python/3.11.11
python3 -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$evidenceRoot/matplotlib-cache"
export PIP_CACHE_DIR="$evidenceRoot/pip-cache"
export UCX_TLS=sm,self
buildDir="$evidenceRoot/ippl"
referenceDir="$evidenceRoot/fastpm"
pythonExe="$evidenceRoot/python/bin/python"
mkdir "$MPLCONFIGDIR"
{
    date -u '+%Y-%m-%dT%H:%M:%SZ'
    hostname
    uname -a
    printf 'execution_mode=%s ranks_max=4 build_jobs=4 threads_per_rank=1 backend=OPENMP\n' "$executionMode"
    if [[ "$executionMode" == slurm ]]; then
        printf 'slurm_job=%s cpu_budget=4 source=Slurm_allocation\n' "$SLURM_JOB_ID"
    else
        printf 'slurm_job=none cpu_budget=4 source=explicit_user_authorized_login_mode\n'
    fi
    printf 'linear_ctest_exception=one_rank_two_threads (within four-CPU budget)\n'
    printf 'source_root=%s\nevidence_root=%s\nUCX_TLS=%s\n' "$sourceRoot" "$evidenceRoot" "$UCX_TLS"
    git -C "$sourceRoot" rev-parse HEAD
    sha256sum "$scriptDir/cpu_validation.sh"
    module list 2>&1
    gcc --version
    mpiexec --version
    mpicc --showme
    cmake --version
    python3 --version
    if [[ "$executionMode" == slurm ]]; then
        scontrol -M "$SLURM_CLUSTER_NAME" show job "$SLURM_JOB_ID"
    fi
} > "$evidenceRoot/environment.txt"
(
    cd "$sourceRoot"
    while IFS= read -r -d '' file; do sha256sum "$file"; done < <(git ls-files -z)
) > "$evidenceRoot/source.sha256"

phase=python_environment
python3 -m venv "$evidenceRoot/python"
"$pythonExe" -m pip install --only-binary=:all: numpy==2.4.6 pandas==3.0.3 matplotlib==3.10.9
"$pythonExe" -m pip freeze > "$evidenceRoot/python-packages.txt"

phase=native_reference_build
export FASTPM_CC=gcc FASTPM_MPICC=mpicc FASTPM_JOBS=4
bash "$sourceRoot/demos/cosmology/reference/build_fastpm.sh" "$referenceDir"
bash "$sourceRoot/demos/cosmology/reference/build_fastpm_evolution.sh" "$referenceDir"

phase=ippl_build
cmake -S "$sourceRoot" -B "$buildDir" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DMPI_CXX_COMPILER=mpicxx \
    -DIPPL_PLATFORMS=OPENMP -DIPPL_ENABLE_COSMOLOGY=ON -DIPPL_ENABLE_FFT=ON \
    -DIPPL_ENABLE_SOLVERS=ON -DBUILD_TESTING=ON -DIPPL_ENABLE_TESTS=OFF \
    -DIPPL_ENABLE_UNIT_TESTS=OFF -DIPPL_ENABLE_KOKKOS_KERNELS=OFF \
    -DKokkos_VERSION=git.5.2.0 -DHeffte_VERSION=git.v2.4.1 \
    -DHeffte_ENABLE_FFTW=OFF -DHeffte_ENABLE_MKL=OFF -DHeffte_ENABLE_GPU_AWARE_MPI=OFF \
    -DIPPL_COSMOLOGY_PYTHON_VALIDATION=ON -DPython3_EXECUTABLE="$pythonExe" \
    -DMPIEXEC_EXECUTABLE="$(command -v mpiexec)" -DMPIEXEC_NUMPROC_FLAG=-n \
    '-DMPIEXEC_PREFLAGS=--bind-to;none;--map-by;slot' \
    -DIPPL_COSMOLOGY_FASTPM_EXECUTABLE="$referenceDir/FastPMForce" \
    -DIPPL_COSMOLOGY_FASTPM_MANIFEST="$referenceDir/build-manifest.txt" \
    -DIPPL_COSMOLOGY_FASTPM_EVOLUTION_EXECUTABLE="$referenceDir/evolution/FastPMEvolution" \
    -DIPPL_COSMOLOGY_FASTPM_EVOLUTION_MANIFEST="$referenceDir/evolution/build-manifest.txt"
cmake --build "$buildDir" -j4 --target \
    Cosmology TestCosmologyPhysics CompareCosmologyForce CompareCosmologyEvolution
for dependency in kokkos heffte; do
    dependencySource="$buildDir/_deps/$dependency-src"
    printf '%s\n' "$dependencySource"
    git -C "$dependencySource" rev-parse HEAD
    git -C "$dependencySource" diff --exit-code HEAD --
done > "$evidenceRoot/dependencies.txt"
sha256sum "$buildDir/demos/cosmology/"{Cosmology,TestCosmologyPhysics,CompareCosmologyForce,CompareCosmologyEvolution} \
    "$buildDir/CMakeCache.txt" > "$evidenceRoot/ippl-build.sha256"

phase=regression_tests
# One controller, serial CTest; individual tests launch at most four MPI ranks.
ctest --test-dir "$buildDir" -L cosmology -j1 --output-on-failure
studyCommand=("$pythonExe" -B "$sourceRoot/demos/cosmology/validate_resolution_study.py"
    --ippl-exe "$buildDir/demos/cosmology/CompareCosmologyEvolution"
    --fastpm-exe "$referenceDir/evolution/FastPMEvolution"
    --fastpm-manifest "$referenceDir/evolution/build-manifest.txt"
    --mpiexec "$(command -v mpiexec)" --numproc-flag=-n
    --mpi-arg=--bind-to --mpi-arg=none --mpi-arg=--map-by --mpi-arg=slot
    --reserve-gib 4 --timeout 3600)
phase=pipeline_smoke
"${studyCommand[@]}" --smoke --stage all --output-dir "$evidenceRoot/smoke"
phase=gaussian_study
# A fresh 18-run Linux CPU campaign, not an invalid cross-platform --resume.
# Exit 1 can mean completed failed checks OR an execution exception. Record the
# report state explicitly below; never infer scientific acceptance from exit alone.
set +e
"${studyCommand[@]}" --stage gaussian --output-dir "$evidenceRoot/gaussian"
studyStatus=$?
set -e
if [[ -f "$evidenceRoot/gaussian/results.json" ]]; then
    "$pythonExe" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps({k:r.get(k) for k in ("state","complete","passed","all_checks_passed","integrity_passed","execution_error","disk_message")}, indent=2))' \
        "$evidenceRoot/gaussian/results.json" > "$evidenceRoot/gaussian-status.json"
else
    printf '{"state":"missing_report","complete":false,"passed":false}\n' > "$evidenceRoot/gaussian-status.json"
fi
cat "$evidenceRoot/gaussian-status.json"
phase=final_integrity
(cd "$sourceRoot"; sha256sum -c "$evidenceRoot/source.sha256")
sha256sum -c "$evidenceRoot/ippl-build.sha256"
phase=gaussian_study_finished
exit "$studyStatus"
