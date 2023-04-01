#!/bin/bash

#SBATCH --partition=hourly      # Using 'hourly' will grant higher priority, daily
#SBATCH --time=00:05:00         # Define max time job will run // cant be more than one hour??
#SBATCH --exclusive
#SBATCH --error=langevin.err    # Define your output file
#SBATCH --output=langevin.out   # Define your output file
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1              # Job will run 32 tasks
#SBATCH --ntasks-per-core=1     # Force no Hyper-Threading, will run 1 task per core

# Shared Memory Parallelism doesn't work on Severin's branch
#SBATCH --cpus-per-task=1
# set OMP_NUM_THREADS=
# set OMP_PROC_BIND=spread
# set OMP_PLACES=threads

# General Solver Parameters
MPI_OVERALLOC=2.0
LB_THRESHOLD=1.0        # Load Balancing Threshold
NR=256                  # Number of gridpoints on the spatial grid (along each dim.)
BEAM_RADIUS=0.001774    # [cm]
BOXL=0.01               # [cm]
NP=156055               # Number of particles
DT=2e-10                # [ms] Timestep
NT=2000                 # Number of timesteps
PARTICLE_CHARGE=-1      # [e]
PARTICLE_MASS=1         # [m_e]
FOCUS_FORCE=1.0         # Scaling factor of constant focusing force
EPS_INV=3.182609e3      # [\frac{cm^3 m_e}{ms^2 q_e^2}] Inverse Vacuum Permittivity
PRINT=10                # How often to print status to `langevin.out`
DUMP_INTERVAL=20       # How often to dump beamstatistics to ${OUT_DIR}

# Collisional Parameters
NV=32                   # Number of gridpoints on the velocity grid (along each dim.)
VMAX=9e4                # [cm / ms] Extent of velocity grid ([-VMAX, VMAX] in each dim.)
REL_BUFFER=1.03         # Relative allocated buffer zone for adaptive velocity
VMESH_ADAPT_B=0         # Adapt velocity mesh size dynamiccally (doesn't work yet)

COLLISION=0             # Boolean defining whether to introduce collisional terms in solver
FCT=1                   # Factor by which to scale the spatial density
DRAG_FCT_B=4e-7         # Scaling factor of Drag term
DIFF_FCT_B=16e4         # Scaling factor of Diffusion term
DRAG_B=1                # Boolean introducing Drag term
DIFF_B=1                # Boolean introducing Diffusion term 

# Take first User argument as foldername if provided
USER_OUT_DIR=$1
USER_OUT_DIR="${USER_OUT_DIR:=langevin}"
OUT_DIR=data/${USER_OUT_DIR}_$(date +%m%d_%H%M)

# Create directory to write output data and this script
mkdir -p ${OUT_DIR}
# Copy this script to the data directory (follows symlinks)
THIS_FILE="$(readlink -f "$0")"
cp ${THIS_FILE} ${OUT_DIR}/jobscript.sh

mkdir -p ${OUT_DIR}

srun --cpus-per-task=${SLURM_CPUS_PER_TASK} ./Langevin \
FFT ${LB_THRESHOLD} ${MPI_OVERALLOC} ${NR} ${BEAM_RADIUS} \
${BOXL} ${NP} ${DT} ${NT} ${PARTICLE_CHARGE} ${PARTICLE_MASS} \
${FOCUS_FORCE} ${DUMP_INTERVAL} ${EPS_INV} ${NV} ${VMAX} \
${REL_BUFFER} ${VMESH_ADAPT_B} ${SCATTER_PHASE_B} ${DRAG_B} \
${DIFF_B} ${FCT} ${DRAG_FCT_B} ${DIFF_FCT_B} ${PRINT} ${COLLISION} ${OUT_DIR} \
--info 5 \
#1>${OUT_DIR}/langevin.out 2>${OUT_DIR}/langevin.err  
