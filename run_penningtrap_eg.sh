#!/bin/bash

export PV_PREFIX=/.../ParaView-5.12.0-MPI-Linux-Python3.10-x86_64
export IPPL_DIR=/.../ippl
# or similar



export PENNINGTRAP_BINDIR=${IPPL_DIR}/build/alpine
export PVSCRIPT=${IPPL_DIR}/src/Stream/InSitu/catalyst_scripts/catalyst-script.py
export PVSCRIPT=${IPPL_DIR}/src/Stream/InSitu/catalyst_scripts/pipeline.py
export  PVPROXY=${IPPL_DIR}/src/Stream/InSitu/catalyst_scripts/proxy.xml
# Ascent Adaptor will try fetch from environment, else swap to default
export ASCENT_ACTIONS_PATH=${IPPL_DIR}/src/Stream/InSitu/ascent_scripts/ascent_actions_default.yaml
# Catalyst Adaptor will fetch from environment
export CATALYST_IMPLEMENTATION_PATHS="${PV_PREFIX}/lib/catalyst"
export CATALYST_IMPLEMENTATION_NAME="paraview"


cd ${PENNINGTRAP_BINDIR}
rm -rd data
mkdir data


# currently run with both ascent and catalyst 
./PenningTrap 4 4 4 512 10 FFT 0.05 LeapFrog --overallocate 1.0  --pvscript $PVSCRIPT --pvproxy  $PVPROXY --frequency 3 --info 5
cd $IPPL_DIR







