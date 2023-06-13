#!/bin/bash
#SBATCH --partition=hourly        # Using 'hourly' will grant higher priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:59:00            # Define max time job will run
#SBATCH --output=data/langevin_cpu.out   # Define your output file
#SBATCH --error=data/langevin_cpu.err    # Define your output file
##SBATCH --exclusive


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# General Solver Parameters
MPI_OVERALLOC=2.0
SOLVER_T=FFT            # Solver Type to solve for the electrostatic potential
LB_THRESHOLD=1.0        # Load Balancing Threshold
NR=64                   # Number of gridpoints on the spatial grid (along each dim.)
BOXL=0.01               # [cm]
NP=156055               # Number of particles
DT=2.15623e-13          # [s] Timestep
PARTICLE_CHARGE=-1      # [e]
PARTICLE_MASS=1         # [m_e]
EPS_INV=3.182609e9      # [\frac{cm^3 m_e}{s^2 q_e^2}] Inverse Vacuum Permittivity

# Collisional Parameters
NV_MAX=128              # Number of gridpoints on the velocity grid (along each dim.)
VMAX=5                  # Maximum vel.-domain size to test (runs tests over the interval of [8, NV_MAX], in powers of two)
FRICTION_SOLVER=VICO    # Solver for first Rosenbluth Potential (Options: [HOCKNEY, VICO])

# Frequency of computing statistics
DUMP_INTERVAL=1         # How often to dump beamstatistics to ${OUT_DIR}

# Take first User argument as foldername if provided
USER_OUT_DIR=$1
USER_OUT_DIR="${USER_OUT_DIR:=langevin}"
OUT_DIR=data/${USER_OUT_DIR}_$(date +%m%d_%H%M)

echo "Output directory: ${OUT_DIR}"

# Create directory to write output data and this script
mkdir -p ${OUT_DIR}
mkdir -p ${OUT_DIR}/convergenceStats

# Copy this script to the data directory (follows symlinks)
THIS_FILE="$(readlink -f "$0")"
cp ${THIS_FILE} ${OUT_DIR}/jobscript.sh

srun --cpus-per-task=${SLURM_CPUS_PER_TASK} ./TestLangevinPotentials  \
    ${MPI_OVERALLOC} ${SOLVER_T} ${LB_THRESHOLD} ${NR} \
    ${BOXL} ${NP} ${DT} ${PARTICLE_CHARGE} ${PARTICLE_MASS} \
    ${EPS_INV} ${NV_MAX} ${VMAX} ${FRICTION_SOLVER} ${OUT_DIR} \
    --info 5 1>&1 | tee ${OUT_DIR}/langevin.out 2>${OUT_DIR}/langevin.err 
