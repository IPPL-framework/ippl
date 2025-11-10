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
# any invalid input will switch to default case, check output during initialisation for parsed settings
# TODO overwrite proxy path:

# does nothing atm
# "ON":
#  any/"OFF": (default)
export IPPL_CATALYST_VIS=ON


# "ON":
#  any/"OFF": (default)
export IPPL_CATALYST_STEER=ON

#  activate/deactivate extractors (frequencies can be overwritten in scripts directly for now)

# "ON":
#  any/"OFF": (default)
export IPPL_CATALYST_PNG=OFF

# "ON":
#  any/"OFF": (default)
export IPPL_CATALYST_VTK=OFF

# any/"ON":(default)    writes catalyst_scripts/catalyst_proxy.xml and continues simulation
# "PRODUCE_ONLY":       writes and throws exception
# "OFF":                doesn't write but (still tries to access old catalyst_scripts/catalyst_proxy.xml 
#                       if CATALYST_PROXY_PATH is not set) and runs simulation
export IPPL_CATALYST_PROXY_OPTION=ON


#######################################################################
# DON'T CHANGE FOR NOW

# "ON":                 <-> masking GHOST_MASKS <-> not finally tested 
#  any/"OFF": (default) <-> cutting GHOST_MASKS <->  works
export IPPL_CATALYST_GHOST_MASKS=OFF

# any/"element" (default)
# "vertex"
export IPPL_CATALYST_ASSOCIATE="element"



#######################################################################
# DON'T USE FOR NOW:

# #####################################################################
# Catalyst Adaptor will try to fetch paths from environment else switch to
# harcoded preconfigured defaults inside IPPL src directory (helped via cmake)
# #####################################################################

# overwrites catalyst main script/pipeline:
# export CATALYST_PIPELINE_PATH=${IPPL_DIR}/src/Stream/InSitu/catalyst_scripts/pipeline_default.py

# overwrites catalyst script (png extraction) for arbitrary vis channels
# export CATALYST_EXTRACTOR_SCRIPT_" +label

# overwrite steering proxies completeley by referencing different file:
# export CATALYST_PROXYS_PATH = 

# change default ranges for stering channels:
# export IPPL_PROXY_CONFIG_YAML





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