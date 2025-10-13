#!/bin/bash

# export IPPL_DIR=./ippl-frk
export PENNINGTRAP_BINDIR=./build/alpine

# These two needed env Variables will be automatically  be set when loading module on 
# julich system. When running everything locally they need to be set manually.
# PV_PREFIX="/.../ParaView-5.12.0-MPI-Linux-Python3.10-x86_64"
#export CATALYST_IMPLEMENTATION_PATHS="${PV_PREFIX}/lib/catalyst"
#export CATALYST_IMPLEMENTATION_NAME="paraview"

# echo $CATALYST_IMPLEMENTATION_PATHS on jureca eg. should be something simlar to: 
# /p/software/jurecadc/stages/2024/software/ParaView/5.12.0-RC2-gpsmpi-2023a/lib64/catalyst


# #####################################################################
#  CONFIGURE CATALYST OPTIONS
# #####################################################################
export IPPL_CATALYST_STEER=OFF
export IPPL_CATALYST_PNG=ON
export IPPL_CATALYST_VTK=OFF


# #####################################################################
# Catalyst Adaptor will try to fetch paths from environment else switch to
# harcoded preconfigured defaults inside IPPL src directory definde via cmake.
# #####################################################################

# export  CATALYST_PIPELINE_PATH=${IPPL_DIR}/src/Stream/InSitu/catalyst_scripts/pipeline_default.py
# export  CATALYST_EXTRACTOR_SCRIPT_P=${IPPL_DIR}/src/Stream/InSitu/catalyst_scripts/catalyst_extractors/png_ext_particle.py
# export  CATALYST_EXTRACTOR_SCRIPT_S=${IPPL_DIR}/src/Stream/InSitu/catalyst_scripts/catalyst_extractors/png_ext_sfield.py
# export  CATALYST_EXTRACTOR_SCRIPT_V=${IPPL_DIR}/src/Stream/InSitu/catalyst_scripts/catalyst_extractors/png_ext_vfield.py

# export  CATALYST_PROXY_PATH_M=${IPPL_DIR}/src/Stream/InSitu/catalyst_scripts/proxy_default_magnetic.xml
# export  CATALYST_PROXY_PATH_E=${IPPL_DIR}/src/Stream/InSitu/catalyst_scripts/proxy_default_electric.xml


# #####################################################################
# ASCENT
# Ascent Adaptor will try fetch from environment, else swap to default
# #####################################################################
# export ASCENT_ACTIONS_PATH=${IPPL_DIR}/src/Stream/InSitu/ascent_scripts/ascent_actions_default.yaml



cd ${PENNINGTRAP_BINDIR}

rm -rd data
mkdir data

#####################################################################################
# when running locally with MPI this might(?) be needed to guarantee compatibility (openMPI vs MPIch)
# export MPIEXEC=$PV_PREFIX/lib/mpiexec
# exec $MPIEXEC -np 1 ...
#####################################################################################
# slurm:
# srun  .... 
# ###################################################################################


# ./PenningTrap 4 4 4 512 20 FFT 0.05 LeapFrog --overallocate 1.0  --info 5
./PenningTrap 8 8 8 4096 20 FFT 0.05 LeapFrog --overallocate 1.0  --info 5

# ./BumponTailInstability 4 4 4 512  20 FFT 0.05 LeapFrog --overallocate 1.0  --info 5
# ./BumponTailInstability 8 8 8 4096 20 FFT 0.05 LeapFrog --overallocate 1.0  --info 5

# ./LandauDamping 4 4 4 512  20 FFT 0.05 LeapFrog --overallocate 1.0  --info 5
# ./LandauDamping 8 8 8 4096 20 FFT 0.05 LeapFrog --overallocate 1.0  --info 5



cd $IPPL_DIR