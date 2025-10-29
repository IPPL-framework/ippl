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

export EXE=@LANDAU_BINARY@
export WRAPPER=/user-environment/wrapper-mpi.sh

mkdir -p data
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=72

srun $WRAPPER $EXE 1024 1024 1024 8589934592 100 FFT 1.0 LeapFrog --overallocate 2.0 --info 10

