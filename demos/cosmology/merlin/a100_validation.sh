#!/bin/bash -l
# Two separate Slurm jobs, always after CPU validation has completed.
# build: four CPUs, one A100. run: four CPUs, four full A100 GPUs, one node.
# Usage: bash -l a100_validation.sh build|run CPU_EVIDENCE GPU_EVIDENCE
set -euo pipefail
if [[ $# != 3 || ($1 != build && $1 != run) || $2 != /* || $3 != /* ]]; then
    printf 'Usage: bash -l %s build|run ABS_CPU_EVIDENCE ABS_GPU_EVIDENCE\n' "$0" >&2
    exit 2
fi
test -n "${SLURM_JOB_ID:-}"
test "${SLURM_JOB_NUM_NODES:-0}" -eq 1
test "${SLURM_NTASKS:-0}" -eq 4
test "${SLURM_CPUS_PER_TASK:-0}" -eq 1
test "${SLURM_CPUS_ON_NODE:-999}" -eq 4
case $(hostname -s) in merlin-g-*) ;; *)
    printf 'Refusing a non-GPU-compute host.\n' >&2; exit 2;; esac

action=$1
cpuRoot=$(cd "$2" && pwd -P)
scriptDir=$(cd "$(dirname "$0")" && pwd -P)
sourceRoot=$(cd "$scriptDir/../../.." && pwd -P)
pythonExe="$cpuRoot/python/bin/python"
referenceDir="$cpuRoot/fastpm"
module unload Python/3.14.4
module load gcc/14.3.0 openmpi/5.0.10_slurm cuda/12.9.1 cmake/4.4.0 Python/3.11.11
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export UCX_TLS=sm,self,cuda_copy,cuda_ipc
export IPPL_CUDART_LIBRARY=/opt/psi/Programming/cuda/12.9.1/lib64/libcudart.so
"$pythonExe" - "$cpuRoot/gaussian/results.json" "$action" <<'PY'
import json, os, sys
report = json.load(open(sys.argv[1]))
assert report['state'] == 'complete' and report['complete'] and report['integrity_passed']
assert report['configuration']['stage'] == 'gaussian' and not report['configuration']['smoke']
assert report['expected_runs'] == len(report['runs']) == 18
assert report['completed_stages'] == ['gaussian']
assert all(run['metadata']['execution_space'] == 'OpenMP'
           for run in report['runs'] if run['code'] == 'ippl')
tokens = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
expected = 1 if sys.argv[2] == 'build' else 4
assert len(tokens) == len(set(tokens)) == expected and all(t.strip() for t in tokens)
assert all(not t.startswith('MIG-') for t in tokens), 'Full A100 devices required, not MIG'
print('CPU execution complete; retained scientific failures:', len(report['failed_checks']))
PY
git -C "$sourceRoot" diff --quiet HEAD --
if [[ "$action" == build ]]; then
    mkdir "$3"
else
    test -f "$3/build-complete.sha256"
    # A run is not automatically retried after interruption or scientific failure.
    mkdir "$3/runtime"
fi
gpuRoot=$(cd "$3" && pwd -P)
buildDir="$gpuRoot/ippl"
config="$gpuRoot/launcher.json"
export MPLCONFIGDIR="$gpuRoot/matplotlib-cache"
mkdir -p "$MPLCONFIGDIR"
exec > >(tee "$gpuRoot/$action-controller.log") 2>&1
phase="$action-setup"
trap 'status=$?; printf "phase=%s exit=%s\n" "$phase" "$status" > "$gpuRoot/$action-exit.txt"' EXIT
{
    date -u '+%Y-%m-%dT%H:%M:%SZ'
    hostname
    uname -a
    module list 2>&1
    printf 'slurm_job=%s action=%s cpu_root=%s\n' "$SLURM_JOB_ID" "$action" "$cpuRoot"
    printf 'CUDA_VISIBLE_DEVICES=%s\nUCX_TLS=%s\n' "$CUDA_VISIBLE_DEVICES" "$UCX_TLS"
    scontrol -M "$SLURM_CLUSTER_NAME" show job "$SLURM_JOB_ID"
    git -C "$sourceRoot" rev-parse HEAD
    sha256sum "$scriptDir/a100_validation.sh" "$cpuRoot/gaussian/results.json"
    gcc --version
    nvcc --version
    mpiexec --version
    "$pythonExe" -m pip freeze
} > "$gpuRoot/$action-environment.txt"
# Query only scheduler-listed devices; record model and disabled MIG mode.
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=name,uuid,pci.bus_id,mig.mode.current \
    --format=csv,noheader > "$gpuRoot/$action-allocated-gpus.csv"
"$pythonExe" - "$gpuRoot/$action-allocated-gpus.csv" "$action" <<'PY'
import csv, sys
rows = list(csv.reader(open(sys.argv[1]), skipinitialspace=True))
assert len(rows) == (1 if sys.argv[2] == 'build' else 4)
assert all(len(row) == 4 and 'A100' in row[0] and row[3].strip() == 'Disabled' for row in rows)
assert len({row[1] for row in rows}) == len(rows)
assert len({row[2].lower() for row in rows}) == len(rows)
PY

if [[ "$action" == build ]]; then
    phase=cuda_build
    (
        cd "$sourceRoot"
        while IFS= read -r -d '' file; do sha256sum "$file"; done < <(git ls-files -z)
    ) > "$gpuRoot/source.sha256"
    # Fresh dependency clones: no borrowed modified OPAL source or build caches.
    cmake -S "$sourceRoot" -B "$buildDir" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DMPI_CXX_COMPILER=mpicxx \
        '-DIPPL_PLATFORMS=OPENMP;CUDA' -DKokkos_ARCH_AMPERE80=ON -DCMAKE_CUDA_ARCHITECTURES=80 \
        -DCUDAToolkit_rt_LIBRARY=/usr/lib64/librt.so.1 \
        -DIPPL_ENABLE_COSMOLOGY=ON -DIPPL_ENABLE_FFT=ON -DIPPL_ENABLE_SOLVERS=ON \
        -DBUILD_TESTING=ON -DIPPL_ENABLE_TESTS=OFF -DIPPL_ENABLE_UNIT_TESTS=OFF \
        -DIPPL_ENABLE_KOKKOS_KERNELS=OFF -DKokkos_VERSION=git.5.2.0 -DHeffte_VERSION=git.v2.4.1 \
        -DHeffte_ENABLE_FFTW=OFF -DHeffte_ENABLE_MKL=OFF -DHeffte_ENABLE_GPU_AWARE_MPI=OFF \
        -DIPPL_COSMOLOGY_PYTHON_VALIDATION=ON -DPython3_EXECUTABLE="$pythonExe" \
        -DMPIEXEC_EXECUTABLE="$scriptDir/gpu_mpiexec.py" -DMPIEXEC_NUMPROC_FLAG=-n \
        "-DMPIEXEC_PREFLAGS=--config;$config" \
        -DIPPL_COSMOLOGY_FASTPM_EXECUTABLE="$referenceDir/FastPMForce" \
        -DIPPL_COSMOLOGY_FASTPM_MANIFEST="$referenceDir/build-manifest.txt" \
        -DIPPL_COSMOLOGY_FASTPM_EVOLUTION_EXECUTABLE="$referenceDir/evolution/FastPMEvolution" \
        -DIPPL_COSMOLOGY_FASTPM_EVOLUTION_MANIFEST="$referenceDir/evolution/build-manifest.txt"
    cmake --build "$buildDir" -j4 --target \
        Cosmology TestCosmologyPhysics CompareCosmologyForce CompareCosmologyEvolution
    for dependency in kokkos heffte; do
        printf '%s\n' "$dependency"
        git -C "$buildDir/_deps/$dependency-src" rev-parse HEAD
        git -C "$buildDir/_deps/$dependency-src" diff --exit-code HEAD --
    done > "$gpuRoot/dependencies.txt"
    phase=launcher_configuration
    "$pythonExe" - "$gpuRoot" "$scriptDir" "$referenceDir" "$pythonExe" "$(command -v mpiexec)" <<'PY'
import hashlib, json, pathlib, sys
root, scripts, native, python, mpi = map(pathlib.Path, sys.argv[1:])
gpu = [root/'ippl/demos/cosmology'/name for name in
       ('Cosmology', 'CompareCosmologyForce', 'CompareCosmologyEvolution')]
cpu = [native/'FastPMForce', native/'evolution/FastPMEvolution']
helper = scripts/'gpu_rank.py'
artifacts = gpu + cpu + [python, mpi, helper, scripts/'gpu_mpiexec.py']
config = dict(mpiexec=str(mpi), python=str(python), gpu_helper=str(helper),
              evidence_dir=str(root/'runtime/launches'), gpu_executables=list(map(str, gpu)),
              cpu_executables=list(map(str, cpu)),
              allocation_evidence=str(root/'run-allocated-gpus.csv'),
              sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts})
with (root/'launcher.json').open('x') as output:
    json.dump(config, output, indent=2)
PY
    (cd "$sourceRoot"; sha256sum -c "$gpuRoot/source.sha256")
    sha256sum "$config" "$buildDir/CMakeCache.txt" "$gpuRoot/source.sha256" \
        "$cpuRoot/gaussian/results.json" "$referenceDir/evolution/build-manifest.txt" \
        "$buildDir/demos/cosmology/"{Cosmology,TestCosmologyPhysics,CompareCosmologyForce,CompareCosmologyEvolution} \
        > "$gpuRoot/build-complete.sha256"
    phase=build_complete_runtime_not_tested
    exit 0
fi

sha256sum -c "$gpuRoot/build-complete.sha256"
(cd "$sourceRoot"; sha256sum -c "$gpuRoot/source.sha256")
mkdir "$gpuRoot/runtime/launches"
phase=cuda_regressions
ctest --test-dir "$buildDir" -L cosmology -j1 --output-on-failure
studyCommand=("$pythonExe" -B "$sourceRoot/demos/cosmology/validate_resolution_study.py"
    --ippl-exe "$buildDir/demos/cosmology/CompareCosmologyEvolution"
    --fastpm-exe "$referenceDir/evolution/FastPMEvolution"
    --fastpm-manifest "$referenceDir/evolution/build-manifest.txt"
    --mpiexec "$scriptDir/gpu_mpiexec.py" --numproc-flag=-n "--mpi-arg=--config=$config"
    --reserve-gib 4 --timeout 3600)
phase=cuda_pipeline_smoke
"${studyCommand[@]}" --smoke --stage all --output-dir "$gpuRoot/runtime/smoke"
phase=cuda_full_study
# Both requested points: all34 spatial and all18 Gaussian runs, no reduced matrix.
set +e
"${studyCommand[@]}" --stage all --output-dir "$gpuRoot/runtime/full"
studyStatus=$?
set -e
phase=cuda_completion_audit
"$pythonExe" "$scriptDir/audit_a100_study.py" "$gpuRoot/runtime/full/results.json" \
    --cpu-report "$cpuRoot/gaussian/results.json" --launch-evidence "$gpuRoot/runtime/launches" \
    --output "$gpuRoot/runtime/completion-audit.json"
phase=cuda_figures
"$pythonExe" -B "$sourceRoot/demos/cosmology/plot_resolution_study.py" \
    --report "$gpuRoot/runtime/full/results.json" --output-dir "$gpuRoot/runtime/figures"
sha256sum -c "$gpuRoot/build-complete.sha256"
(cd "$sourceRoot"; sha256sum -c "$gpuRoot/source.sha256")
phase=full_study_complete_acceptance_in_report
exit "$studyStatus"
