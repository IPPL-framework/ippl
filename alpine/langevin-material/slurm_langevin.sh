#!/bin/bash
#SBATCH --partition=hourly      # Using 'hourly' will grant higher priority, daily
#SBATCH --time=00:59:00         # Define max time job will run // cant be more than one hour??
#SBATCH --error=langevin.err    # Define your output file
#SBATCH --output=langevin.out   # Define your output file
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1              # Job will run 32 tasks
#SBATCH --ntasks-per-core=44     # Force no Hyper-Threading, will run 1 task per core
##SBATCH --exclusive

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

###########################
#NM=256
#NV=32
NM=64
NV=16

VMAX=9e4                 # cm / ms

###########################
OUT_DIR=data.${VER}
COLLISION=0
DRAG_B=4e-7
DIFF_B=16e4
FCT=1
DRAG=1
DIFF=1

###########################
GADAPT=0                 #dont adaptive velocity mesh size doesn't work yet...
REL_BUFFER=1.03
###########################
FOCUS=1                   #  scale constant focusing force...
###########################
epsinv=3.182609e3
DT=2e-10
TT=1200
NP=156055
PQ=-1 #in e
PM=1 #in me
BEAMRADIUS=0.001774 #cm
BOXL=0.01 #cm
PRINT=1

# Take first User argument as foldername if provided
USER_OUT_DIR=$1
USER_OUT_DIR="${USER_OUT_DIR:=langevin}"
OUT_DIR=data/${USER_OUT_DIR}_$(date +%m%d_%H%M)

# Create directory to write output data and this script
mkdir -p ${OUT_DIR}
# Copy this script to the data directory (follows symlinks)
THIS_FILE="$(readlink -f "$0")"
cp ${THIS_FILE} ${OUT_DIR}/jobscript.sh

###########################
srun Langevin --cpus-per-task=${SLURM_CPUS_PER_TASK} \
FFT 1.0 2.0 \
${NM} ${BEAMRADIUS} ${BOXL} ${NP} ${DT} ${TT} ${PQ} ${PM} ${FOCUS} 1 \
${epsinv} ${NV} ${VMAX} ${REL_BUFFER} \
${GADAPT} 1 ${DRAG_B} ${DIFF_B}  ${FCT}  ${DRAG} ${DIFF} ${PRINT} ${COLLISION} ${OUT_DIR} \
--info 5 \
1>${OUT_DIR}/langevin.out 2>${OUT_DIR}/langevin.err  
