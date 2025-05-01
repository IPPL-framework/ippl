#!/bin/bash -l
#
#SBATCH --job-name=cos1
#SBATCH --error=cos-%j.error
#SBATCH --output=cos-%j.out
#SBATCH --time=00:05:00
#SBATCH --partition=dev-g
#SBATCH --nodes 32
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --account=project_465001705 
#SBATCH --hint=nomultithread
#SBATCH --hint=exclusive

CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"

export MPICH_GPU_SUPPORT_ENABLED=1
export Data_DIR=/users/adelmann/runStructureFormation/lsf_32/

#export Data_DIR=/flash/project_465001705/
ulimit -s unlimited

export EXE_DIR=/users/adelmann/sandbox/ippl/build/cosmology-new/

cat << EOF > select_gpu
#!/bin/bash

export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec \$*
EOF

chmod +x ./select_gpu
srun ./select_gpu ${EXE_DIR}/StructureFormation input.par cmb.tf out.data ${Data_DIR} FFT 1.0 LeapFrog --overallocate 1.0 --info 5
rm -rf ./select_gpu


