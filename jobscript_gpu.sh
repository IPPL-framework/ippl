#!/bin/bash -x
#SBATCH --account=cstma
#SBATCH --nodes=1                # number of compute nodes
#SBATCH --ntasks-per-node=1      # MPI ranks per node (total ranks = nodes * ntasks-per-node)
#SBATCH --output=vic.out
#SBATCH --error=vic.error
#SBATCH --time=00:10:00          # wall time (hh:mm:ss)
#SBATCH --partition=develbooster
#SBATCH --gres=gpu:1             # GPUs per node (must match ntasks-per-node if each rank uses one GPU)

srun --cpu-bind=sockets compute-sanitizer --tool memcheck \
     ./build_gpu/alvine/VortexInCell 128 128 10000 100 FFT 1 --overallocate 1.0 --info 5
