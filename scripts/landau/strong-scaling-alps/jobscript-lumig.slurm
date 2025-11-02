#!/bin/bash
# ---
# sbatch settings for LUMI-G
# ---
#SBATCH --job-name=landau__n_
#SBATCH --partition=standard-g
#SBATCH --account=@IPPL_JOB_SUBMISSION_ACCOUNT@
#SBATCH --time=00:30:00

#SBATCH --nodes=_n_
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
##SBATCH --cpus-per-task=7
#SBATCH --exclusive

#SBATCH --output=landau__n_.out
#SBATCH --error=landau__n_.error

module load  LUMI/24.03 partition/G cpeAMD rocm buildtools/24.03

cat << EOF > select_gpu
#!/bin/bash

export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec \$*
EOF

chmod +x ./select_gpu

# variables of the form @VARIABLE@ will have their values set by cmake during configure_file
# and the completed version of this script will be placed in the build dir scripts folder

export EXE=@LANDAU_BINARY@

mkdir -p data
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=6
export MPICH_GPU_SUPPORT_ENABLED=1

#CPU_BIND="mask_cpu:fe000000000000,fe00000000000000"
#CPU_BIND="${CPU_BIND},fe0000,fe000000"
#CPU_BIND="${CPU_BIND},fe,fe00"
#CPU_BIND="${CPU_BIND},fe00000000,fe0000000000"

CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"

srun --cpu-bind=${CPU_BIND} ./select_gpu $EXE 1024 1024 1024 8589934592 100 FFT 1.0 LeapFrog --overallocate 2.0 --info 10

rm -rf ./select_gpu
