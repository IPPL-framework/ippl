#!/bin/bash -l
# Slurm-only A100 build/run entry point for run_three_code_campaign.py.
# Usage: a11_job.sh build|run ABS_CAMPAIGN_ROOT RANKS
set -euo pipefail
[[ $# == 3 && ($1 == build || $1 == run) && $2 == /* && ($3 == 1 || $3 == 4) ]]
[[ -n ${SLURM_JOB_ID:-} && ${SLURM_JOB_NUM_NODES:-0} == 1 ]]
case $(hostname -s) in merlin-g-*) ;; *) exit 2;; esac
action=$1
root=$2
ranks=$3
module unload Python/3.14.4
module load gcc/14.3.0 openmpi/5.0.10_slurm cuda/12.9.1 cmake/4.4.0 Python/3.11.11
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg UCX_TLS=sm,self,cuda_copy,cuda_ipc
export IPPL_CUDART_LIBRARY=/opt/psi/Programming/cuda/12.9.1/lib64/libcudart.so
readarray -t settings < <(python3 - "$root/configuration.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]));print(c['python']);print(c['build_dir']);print(c['nvcc_wrapper']);print(c['source_root'])
PY
)
pythonExe=${settings[0]}
buildDir=${settings[1]}
nvccWrapper=${settings[2]}
sourceRoot=${settings[3]}
export KOKKOS_NVCC_WRAPPER_DEFAULT_COMPILER=g++
export MPLCONFIGDIR="$root/matplotlib-cache"
mkdir -p "$MPLCONFIGDIR"
"$pythonExe" - "$root/configuration.json" "$action" "$ranks" <<'PY'
import hashlib,json,os,sys
c=json.load(open(sys.argv[1]));expected=1 if sys.argv[2]=='build' else int(sys.argv[3])
tokens=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
assert len(tokens)==len(set(tokens))==expected and all(tokens)
assert all(not t.startswith('MIG-') for t in tokens)
assert int(os.environ['SLURM_NTASKS'])==(1 if sys.argv[2]=='build' else expected)
assert int(os.environ['SLURM_CPUS_PER_TASK'])==(4 if sys.argv[2]=='build' else 1)
for p,d in c['source_sha256'].items():assert hashlib.sha256(open(p,'rb').read()).hexdigest()==d,p
assert hashlib.sha256(open(c['shared_ic'],'rb').read()).hexdigest()==c['shared_ic_sha256']
PY
if [[ $action == build ]]; then
    # Serialize cached-build validation and compilation when several controllers coexist.
    mkdir -p "$buildDir"
    exec 9>"$buildDir/a11-build.lock"
    flock 9
    if [[ -x "$buildDir/demos/cosmology/CompareCosmologyEvolution" && -f "$buildDir/a11-build-manifest.json" ]]; then
        "$pythonExe" - "$root/configuration.json" "$buildDir/a11-build-manifest.json" "$root/build-manifest.json" <<'PY'
import hashlib,json,sys
c=json.load(open(sys.argv[1]));m=json.load(open(sys.argv[2]))
sys.path.insert(0,c['source_root']+'/demos/cosmology/python')
from run_three_code_campaign import native_source_hashes
assert native_source_hashes(m['source_sha256'])==native_source_hashes(c['source_sha256']),'Cached native CUDA source differs; select a fresh build directory'
m['reused_for_configuration_sha256']=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()
m['launch_source_sha256']=c['source_sha256']
for p,d in m['artifacts'].items():assert hashlib.sha256(open(p,'rb').read()).hexdigest()==d,p
json.dump(m,open(sys.argv[3],'w'),indent=2)
PY
        exit 0
    fi
    cmake -S "$sourceRoot" -B "$buildDir" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER="$nvccWrapper" -DMPI_CXX_COMPILER=mpicxx \
        '-DIPPL_PLATFORMS=OPENMP;CUDA' -DKokkos_ARCH_AMPERE80=ON -DCMAKE_CUDA_ARCHITECTURES=80 \
        -DCUDAToolkit_rt_LIBRARY=/usr/lib64/librt.so.1 \
        -DIPPL_ENABLE_COSMOLOGY=ON -DIPPL_ENABLE_FFT=ON -DIPPL_ENABLE_SOLVERS=ON \
        -DBUILD_TESTING=ON -DIPPL_ENABLE_TESTS=OFF -DIPPL_ENABLE_UNIT_TESTS=OFF \
        -DIPPL_ENABLE_KOKKOS_KERNELS=OFF -DKokkos_VERSION=git.5.2.0 -DHeffte_VERSION=git.v2.4.1 \
        -DHeffte_ENABLE_FFTW=OFF -DHeffte_ENABLE_MKL=OFF -DHeffte_ENABLE_GPU_AWARE_MPI=OFF \
        -DIPPL_COSMOLOGY_PYTHON_VALIDATION=ON -DPython3_EXECUTABLE="$pythonExe"
    cmake --build "$buildDir" -j4 --target Cosmology CompareCosmologyEvolution CompareCosmologyForce TestCosmologyPhysics
    "$buildDir/demos/cosmology/TestCosmologyPhysics"
    "$pythonExe" - "$root" "$buildDir" <<'PY'
import hashlib,json,pathlib,subprocess,sys
r,b=map(pathlib.Path,sys.argv[1:]);c=json.load(open(r/'configuration.json'))
for p,d in c['source_sha256'].items():assert hashlib.sha256(open(p,'rb').read()).hexdigest()==d,p
artifacts=[b/'demos/cosmology'/n for n in ('Cosmology','CompareCosmologyEvolution','CompareCosmologyForce','TestCosmologyPhysics')]+[b/'CMakeCache.txt']
m={'source_sha256':c['source_sha256'],'artifacts':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts},'source_commit':subprocess.check_output(['git','-C',c['source_root'],'rev-parse','HEAD'],text=True).strip(),'dependencies':{n:subprocess.check_output(['git','-C',str(b/'_deps'/f'{n}-src'),'rev-parse','HEAD'],text=True).strip() for n in ('kokkos','heffte')}}
for p in (r/'build-manifest.json',b/'a11-build-manifest.json'):json.dump(m,open(p,'w'),indent=2)
PY
    exit 0
fi
result="$root/gpu$ranks"
mkdir "$result"
phase=setup
trap 'status=$?; printf "phase=%s exit=%s\n" "$phase" "$status" > "$result/exit.txt"' EXIT
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=name,uuid,pci.bus_id,mig.mode.current --format=csv,noheader > "$result/allocated-gpus.csv"
{
    date -u
    hostname
    module list 2>&1
    scontrol -M gmerlin6 show job "$SLURM_JOB_ID"
    printf 'CUDA_VISIBLE_DEVICES=%s\nUCX_TLS=%s\n' "$CUDA_VISIBLE_DEVICES" "$UCX_TLS"
} > "$result/environment.txt"
"$pythonExe" - "$root" "$result" "$ranks" <<'PY'
import csv,hashlib,json,os,pathlib,sys
r,d=map(pathlib.Path,sys.argv[1:3]);rank=int(sys.argv[3]);c=json.load(open(r/'configuration.json'))
rows=list(csv.reader(open(d/'allocated-gpus.csv'),skipinitialspace=True))
assert len(rows)==rank and all(len(x)==4 and 'A100' in x[0] and x[3]=='Disabled' for x in rows)
assert len({x[1] for x in rows})==len({x[2] for x in rows})==rank
m=json.load(open(r/'build-manifest.json'))
sys.path.insert(0,c['source_root']+'/demos/cosmology/python')
from run_three_code_campaign import native_source_hashes
assert native_source_hashes(m['source_sha256'])==native_source_hashes(c['source_sha256'])
for p,h in m['artifacts'].items():assert hashlib.sha256(open(p,'rb').read()).hexdigest()==h,p
c.update(ranks=rank,result_root=str(d),job_id=os.environ['SLURM_JOB_ID'])
json.dump(c,open(d/'configuration.json','w'),indent=2)
PY
# Every rank inherits the node allocation, then gpu_rank.py narrows it to one GPU.
launcher=(mpiexec --mca pml ucx --mca pml_ucx_tls any --mca pml_ucx_devices any --bind-to none --map-by slot -n "$ranks" "$pythonExe" -B "$sourceRoot/demos/cosmology/python/merlin/gpu_rank.py")
phase=cuda_spectral_sanity
"${launcher[@]}" "$buildDir/demos/cosmology/Cosmology" --self-test > "$result/sanity.log" 2>&1
readarray -t runSettings < <("$pythonExe" - "$root/configuration.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]));print(c['grid']);print(c['steps']);print(c['checkpoints']);print(c['shared_ic']);print(c['timeout'])
PY
)
phase=evolution
start=$(date +%s)
timeout --signal=TERM "${runSettings[4]}" "${launcher[@]}" "$buildDir/demos/cosmology/CompareCosmologyEvolution" \
    "${runSettings[0]}" "${runSettings[0]}" 168.75 0.31 0.01 1 "${runSettings[1]}" "${runSettings[2]}" \
    "${runSettings[3]}" "$result/run" > "$result/solver.log" 2>&1
printf '%s\n' "$(( $(date +%s) - start ))" > "$result/wall-seconds.txt"
phase=analysis
"$pythonExe" -B - "$sourceRoot/demos/cosmology/python" "$result" <<'PY'
import sys
sys.path.insert(0,sys.argv[1])
from run_three_code_campaign import analyze_gpu
analyze_gpu(sys.argv[2])
PY
phase=complete
