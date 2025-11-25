#!/bin/bash
# ---
# sbatch settings for gh200 alps
# ---
#SBATCH --job-name=landau__n_
#SBATCH --account=@IPPL_JOB_SUBMISSION_ACCOUNT@
#SBATCH --time=00:30:00

#SBATCH --nodes=_n_
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive

# currently hardcoded uenv vars, but should be configurable
#SBATCH --uenv=/capstor/store/cscs/cscs/public/uenvs/opal-x-gh200-mpich-gcc-2025-09-28.squashfs
#SBATCH --view=develop

#SBATCH --output=landau__n_.out
#SBATCH --error=landau__n_.error

# variables of the form @VARIABLE@ will have their values set by cmake during configure_file
# and the completed version of this script will be placed in the build dir scripts folder

# ====================
# we output messages to stdout and also stderr to simplify debugging since logs are usually in
# separate files and we want to see where errors are happening
# ====================
function debug_output() {
    {
        echo
        echo -------------------------------------
        echo -- $1
        echo -------------------------------------
    } | tee >(cat >&2)
}

SPACK_CMD='spack -c config:install_tree:/user-environment'

export BASE_DIR=base_dir
export JOB_DIR=job_dir
export WRAPPER=/capstor/scratch/cscs/biddisco/build-santis/wrapper-mpi-ipc1.sh
export WRAPPER=/capstor/scratch/cscs/biddisco/build-santis/wrapper-mpi-ipc0.sh
export EXE=@LANDAU_BINARY@

mkdir -p data
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=72

#export MPICH_DBG_OUTPUT=VERBOSE
#export MPICH_DBG_CLASS=ALL
#export MPICH_DBG_FILENAME="dbg-%w-%d.log"

#
# Profiler preamble
#
export HPCRUN_MEMLEAK_TRACKING=1
export HPCRUN_PROCESS_FRACTION=2.0
export HPCTOOLKIT=$($SPACK_CMD location -i hpctoolkit)
export HPCRUN_OUT_PATH=./hpctoolkit
export HPCTOOLKIT_HPCSTRUCT_CACHE=$BASE_DIR/hpcrun-cache

unset PROFILE_HPCTOOLKIT
unset PROFILE_NSYS
# export PROFILE_HPCTOOLKIT=1
# export PROFILE_NSYS=1

if [ -n "$PROFILE_HPCTOOLKIT" ]; then
  debug_output "Running with hpctoolkit profiler"
  HPCRUN_CMDS="hpcrun -e CYCLES -tt -e gpu=nvidia --process-fraction $HPCRUN_PROCESS_FRACTION --output $HPCRUN_OUT_PATH"
  mkdir -p $HPCRUN_OUT_PATH
  mkdir -p $HPCTOOLKIT_HPCSTRUCT_CACHE
fi
if [ -n "$PROFILE_NSYS" ]; then
  debug_output "Running with nsys profiler"
  NSYS_CMD="nsys profile --trace=cuda,mpi --cuda-memory-usage=true --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true"
fi

# --------------------
iterations=100
cubesize=1024
particles_per_cell=8
nparticles=$(( $cubesize * $cubesize * $cubesize * $particles_per_cell ))
echo "Cube Size: $cubesize"
echo "N Particles: $nparticles"
echo "Iterations: $iterations"
# --------------------

srun $WRAPPER $HPCRUN_CMDS $NSYS_CMD $EXE $cubesize $cubesize $cubesize $nparticles $iterations FFT 1.0 LeapFrog --overallocate 2.0 --info 10

if [ -n "$PROFILE_HPCTOOLKIT" ]; then
    debug_output "Running hpcstruct on results"
    srun -u  -n 1  --cpu-bind=core -c 1  -n 1 hpcstruct --time -j $OMP_NUM_THREADS $HPCRUN_OUT_PATH

    debug_output "Running hpcprof on results"
    srun -u  -n 1  --cpu-bind=core -c 1  -n 1 hpcprof -j $OMP_NUM_THREADS $HPCRUN_OUT_PATH
fi
