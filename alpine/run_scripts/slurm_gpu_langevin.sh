#!/bin/bash
#SBATCH --partition=hourly        # Using 'hourly' will grant higher priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --clusters=gmerlin6
#SBATCH --partition=gwendolen
#SBATCH --account=gwendolen
#SBATCH --gpus=1
#SBATCH --time=00:06:00            # Define max time job will run
#SBATCH --output=data/langevin_gpu.out   # Define your output file
#SBATCH --error=data/langevin_gpu.err    # Define your output file

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# General Solver Parameters
MPI_OVERALLOC=2.0
SOLVER_T=FFT            # Solver Type to solve for the electrostatic potential
LB_THRESHOLD=1.0        # Load Balancing Threshold
NR=256                  # Number of gridpoints on the spatial grid (along each dim.)
BEAM_RADIUS=0.001774    # [cm]
BOXL=0.01               # [cm]
NP=156055               # Number of particles
DT=2.15623e-13          # [s] Timestep
NT=1200                 # Number of timesteps
PARTICLE_CHARGE=-1      # [e]
PARTICLE_MASS=1         # [m_e]
FOCUS_FORCE=1.5         # Scaling factor of constant focusing force
EPS_INV=3.182609e9      # [\frac{cm^3 m_e}{s^2 q_e^2}] Inverse Vacuum Permittivity

# Collisional Parameters
NV=64                   # Number of gridpoints on the velocity grid (along each dim.)
VMAX=5e7                # [cm / s] Extent of velocity grid ([-VMAX, VMAX] in each dim.); $VMAX = 5\sigma_v$ of Boltzmann distribution
FRICTION_SOLVER=HOCKNEY # Solver for first Rosenbluth Potential (Options: [HOCKNEY, VICO])
HESSIAN_OPERATOR=SPECTRAL   # How to compute the hessian [SPECTRAL, FD]

# Frequency of computing statistics
DUMP_INTERVAL=1         # How often to dump beamstatistics to ${OUT_DIR}

# Take first User argument as foldername if provided
USER_OUT_DIR=$1
USER_OUT_DIR="${USER_OUT_DIR:=langevin}"
OUT_DIR=data/${USER_OUT_DIR}_$(date +%m%d_%H%M)

echo "Output directory: ${OUT_DIR}"

# Create directory to write output data and this script
mkdir -p ${OUT_DIR}
# Copy this script to the data directory (follows symlinks)
THIS_FILE="$(readlink -f "$0")"
cp ${THIS_FILE} ${OUT_DIR}/jobscript.sh

srun --cpus-per-task=${SLURM_CPUS_PER_TASK} ./Langevin \
    ${MPI_OVERALLOC} ${SOLVER_T} ${LB_THRESHOLD} ${NR} \
    ${BEAM_RADIUS} ${BOXL} ${NP} ${DT} \
    ${NT} ${PARTICLE_CHARGE} ${PARTICLE_MASS} ${FOCUS_FORCE} \
    ${EPS_INV} ${NV} ${VMAX} ${FRICTION_SOLVER} ${HESSIAN_OPERATOR} \
    ${DUMP_INTERVAL} ${OUT_DIR} \
    --info 5 --kokkos-num-devices=${SLURM_GPUS} 2>&1 | tee -a ${OUT_DIR}/langevin.out | tee -a ${OUT_DIR}/langevin.err >&2
