#!/bin/bash

export IPPL_DIR=/.../ippl-frk
export PENNINGTRAP_BINDIR=${IPPL_DIR}/build/alpine

PV_PREFIX=".../ParaView-5.12.0-MPI-Linux-Python3.10-x86_64"

export  CATALYST_PIPELINE_PATH=${IPPL_DIR}/src/Stream/InSitu/catalyst_scripts/pipeline_default.py

export  CATALYST_EXTRACTOR_SCRIPT_P=${IPPL_DIR}/src/Stream/InSitu/catalyst_scripts/catalyst_extractors/png_ext_particle.py
export    CATALYST_EXTRACTOR_SCRIPT_S=${IPPL_DIR}/src/Stream/InSitu/catalyst_scripts/catalyst_extractors/png_ext_sfield.py
export    CATALYST_EXTRACTOR_SCRIPT_V=${IPPL_DIR}/src/Stream/InSitu/catalyst_scripts/catalyst_extractors/png_ext_vfield.py


export CATALYST_IMPLEMENTATION_PATHS="${PV_PREFIX}/lib/catalyst"
export CATALYST_IMPLEMENTATION_NAME="paraview"

# export ASCENT_ACTIONS_PATH=${IPPL_DIR}/src/Stream/InSitu/ascent_scripts/ascent_actions_default.yaml
# Ascent Adaptor will try fetch from environment, else swap to default
# export ASCENT_ACTIONS_PATH=${IPPL_DIR}/src/Stream/InSitu/ascent_scripts/ascent_actions_default.yaml



cd ${PENNINGTRAP_BINDIR}

rm -rd data
mkdir data

# export MPIEXEC=/.../ParaView-5.12.0-MPI-Linux-Python3.10-x86_64/lib/mpiexec
# exec $MPIEXEC -np 1 

# ./PenningTrap 4 4 4 512 21 FFT 0.05 LeapFrog --overallocate 1.0  --info 5
./PenningTrap 8 8 8 4096 21 FFT 0.05 LeapFrog --overallocate 1.0  --info 5
cd $IPPL_DIR


