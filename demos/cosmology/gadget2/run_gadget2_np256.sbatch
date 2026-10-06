#!/bin/bash
#SBATCH --job-name=gadget2-np256
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=/data/user/adelmann/gadget2-zeldovich-np256-20261006/logs/slurm-%j.out
#SBATCH --error=/data/user/adelmann/gadget2-zeldovich-np256-20261006/logs/slurm-%j.err
set -euo pipefail

campaign_root=/data/user/adelmann/gadget2-zeldovich-np256-20261006
python_runtime=/data/user/adelmann/cosmology-cpu-login-20261004/python/bin/python
module load gcc/14.3.0 openmpi/5.0.10_slurm
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$campaign_root"
mkdir -p "$campaign_root/analysis/matplotlib-cache"
export MPLCONFIGDIR="$campaign_root/analysis/matplotlib-cache"
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
module list
python3 "$campaign_root/source/run_gadget2_np256.py"
"$python_runtime" -B "$campaign_root/source/analyze_gadget2_np256.py"
"$python_runtime" -B "$campaign_root/source/plot_three_code_np256.py"
