#!/bin/bash

#export OMP_NUM_THREADS=6
#export OMP_PROC_BIND=spread
#export OMP_PLACES=threads

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
EPS_INV=3.182609e9      # [\frac{cm^3 m_e}{s^2 q_e^2}] Inverse Vacuum Permittivity
FOCUS_FORCE=1.5         # Scaling factor of constant focusing force
PRINT=1                 # How often to print status to `langevin.out`
PRINT_INTERVAL=1        # How often to dump beamstatistics to ${OUT_DIR}

# Collisional Parameters
NV=4                   # Number of gridpoints on the velocity grid (along each dim.)
VMAX=9e4                # [cm / ms] Extent of velocity grid ([-VMAX, VMAX] in each dim.)
REL_BUFFER=1.03         # Relative allocated buffer zone for adaptive velocity
VMESH_ADAPT_B=0         # Adapt velocity mesh size dynamiccally (doesn't work yet)
SCATTER_PHASE_B=0       # Scatter full phasespace before computing collisions \
                        # (not sure why this is needed, as it is done anyways for each collisional type)
COLLISION=0             # Boolean defining whether to introduce collisional terms in solver
FCT=1                   # Factor by which to scale the spatial density
DRAG_FCT_B=4e-7         # Scaling factor of Drag term
DIFF_FCT_B=16e4         # Scaling factor of Diffusion term
DRAG_B=1                # Boolean introducing Drag term
DIFFUSION_B=1           # Boolean introducing Diffusion term 

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

./Langevin \
${MPI_OVERALLOC} ${SOLVER_T} ${LB_THRESHOLD} ${NR} \
${BEAM_RADIUS} ${BOXL} ${NP} ${DT} \
${NT} ${PARTICLE_CHARGE} ${PARTICLE_MASS} \
${FOCUS_FORCE} ${PRINT_INTERVAL} ${EPS_INV} ${NV} ${VMAX} ${REL_BUFFER} \
${VMESH_ADAPT_B} ${SCATTER_PHASE_B} ${FCT} ${DRAG_FCT_B} ${DIFF_FCT_B} \
${DRAG_B} ${DIFFUSION_B} ${PRINT} ${COLLISION} ${OUT_DIR} \
--info 5 1>&1 | tee ${OUT_DIR}/langevin.out 2>${OUT_DIR}/langevin.err 
