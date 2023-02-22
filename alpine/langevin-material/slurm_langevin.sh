#!/bin/bash

#SBATCH --partition=hourly      # Using 'hourly' will grant higher priority, daily
#SBATCH --time=01:00:00         # Define max time job will run // cant be more than one hour??
#SBATCH --exclusive
#SBATCH --error=langevin.err    # Define your output file
#SBATCH --output=langevin.out   # Define your output file
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1              # Job will run 32 tasks
#SBATCH --ntasks-per-core=1     # Force no Hyper-Threading, will run 1 task per core


##SBATCH --cpus-per-task=1
# set OMP_PROC_BIND=spread
# set OMP_PLACES=threads

###########################
###########################

###########################
NM=256
NV=32
# NM=16
# NV=4

                 # 1=fixed spatial density factorC; 0=gather factor

VMAX=6e4                 # cm / ms



VER=VMAX:6e4_DIFF:1e-7_DRAG:16e4

###########################
OUT_DIR=data.${VER}
COLLISION=1
DRAG_B=4e-7
DIFF_B=16e4
FCT=1
DRAG=1
DIFF=1

###########################
# OUT_DIR=${VER}
# COLLISION=1
# DRAG=1
# DIFF=1
########################### 
###########################
# OUT_DIR=${VER}_dr
# COLLISION=1
# DRAG=1
# DIFF=0
########################### 
###########################
# OUT_DIR=${VER}_di
# COLLISION=1
# DRAG=0
# DIFF=1
########################### 
###########################
# OUT_DIR=${VER}_nc
# COLLISION=0
# DRAG=1
# DIFF=0
########################### 










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

mkdir ${OUT_DIR}
###########################
srun Langevin \
FFT 1.0 2.0 \
${NM} ${BEAMRADIUS} ${BOXL} ${NP} ${DT} ${TT} ${PQ} ${PM} ${FOCUS} 1 \
${epsinv} ${NV} ${VMAX} ${REL_BUFFER} \
${GADAPT} 1 ${DRAG_B} ${DIFF_B}  ${FCT}  ${DRAG} ${DIFF} ${PRINT} ${COLLISION} ${OUT_DIR} \
--info 5 \
1>${OUT_DIR}/langevin.out 2>${OUT_DIR}/langevin.err  

 