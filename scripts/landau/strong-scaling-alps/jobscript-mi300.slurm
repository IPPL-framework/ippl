#!/bin/bash
# ---
# sbatch settings for mi300 alps
# ---
#SBATCH --job-name=landau__n_
#SBATCH --account=@IPPL_JOB_SUBMISSION_ACCOUNT@
#SBATCH --time=00:30:00
#SBATCH --partition=mi300

#SBATCH --nodes=_n_
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive

#SBATCH --uenv=/capstor/scratch/cscs/biddisco/opal-x-mi300-mpich-gcc-2025-10-27.squashfs
#SBATCH --view=default

#SBATCH --output=landau__n_.out
#SBATCH --error=landau__n_.error

# variables of the form  will have their values set by cmake during configure_file
# and the completed version of this script will be placed in the build dir scripts folder

export EXE=@LANDAU_BINARY@
export WRAPPER=@PROJECT_BINARY_DIR@/@JOB_SCRIPTS_PATH@/@JOB_WRAPPER_SCRIPT@

mkdir -p data
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=8

#BIND="--cpu-bind=mask_cpu:fffe000000000000,0000fffe00000000,00000000fffe0000,000000000000fffe",

# --------------------
iterations=100
cubesize=1024
particles_per_cell=8
nparticles=$(( $cubesize * $cubesize * $cubesize * $particles_per_cell ))
echo "Cube Size: $cubesize"
echo "N Particles: $nparticles"
echo "Iterations: $iterations"
# --------------------

srun $BIND $WRAPPER $EXE $cubesize $cubesize $cubesize $nparticles $iterations FFT 1.0 LeapFrog --overallocate 2.0 --info 10
