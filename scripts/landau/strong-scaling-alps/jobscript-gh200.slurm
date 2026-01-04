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

#SBATCH --uenv=/capstor/store/cscs/cscs/public/uenvs/opal-x-gh200-mpich-gcc-2025-09-28.squashfs
#SBATCH --view=develop

#SBATCH --output=landau__n_.out
#SBATCH --error=landau__n_.error

# variables of the form @VARIABLE@ will have their values set by cmake during configure_file
# and the completed version of this script will be placed in the build dir scripts folder

# ====================
# we output messages to stdout and also stderr to simplify debugging since logs are usually in
# separate files and we want to see where errors are happening in the script
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
export EXE=@LANDAU_BINARY@
export WRAPPER=@PROJECT_BINARY_DIR@/@JOB_SCRIPTS_PATH@/@JOB_WRAPPER_SCRIPT@

mkdir -p data
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=72

# --------------------
# cray-mpich debug putput
#export MPICH_DBG_OUTPUT=VERBOSE
#export MPICH_DBG_CLASS=ALL
#export MPICH_DBG_FILENAME="dbg-%w-%d.log"

# --------------------
iterations=100
cubesize=1024
particles_per_cell=8
nparticles=$(( $cubesize * $cubesize * $cubesize * $particles_per_cell ))
echo "Cube Size: $cubesize"
echo "N Particles: $nparticles"
echo "Iterations: $iterations"
# --------------------

srun $WRAPPER $EXE $cubesize $cubesize $cubesize $nparticles $iterations FFT 1.0 LeapFrog --overallocate 2.0 --info 10
