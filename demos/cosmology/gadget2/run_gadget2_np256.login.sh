#!/usr/bin/env bash
set -euo pipefail

campaign_root=/data/user/adelmann/gadget2-zeldovich-np256-20261006
python_runtime=/data/user/adelmann/cosmology-cpu-login-20261004/python/bin/python
module load gcc/14.3.0 openmpi/5.0.10_slurm
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$campaign_root/analysis/matplotlib-cache"
export MPLBACKEND=Agg
mkdir -p "$MPLCONFIGDIR"
cd "$campaign_root"
hostname
module list
"$python_runtime" -B "$campaign_root/source/run_gadget2_np256.py" --login
"$python_runtime" -B "$campaign_root/source/analyze_gadget2_np256.py"
"$python_runtime" -B "$campaign_root/source/plot_three_code_np256.py"
