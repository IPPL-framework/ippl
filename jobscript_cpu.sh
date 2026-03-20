#!/bin/bash -x
#SBATCH --account=cstma
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=vic.out
#SBATCH --error=vic.error
#SBATCH --time=00:10:00
#SBATCH --partition=develbooster
#SBATCH --gres=gpu:2

srun --cpu-bind=sockets --export=ALL,OMP_NUM_THREADS=16  ./build/alvine/VortexInCell 128 128 10000 100 FFT 1 --overallocate 1.0 --info 5
