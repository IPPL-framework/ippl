OUT_DIR=data
epsinv=3.14.15e3
NM=32
NV=8
NP=156055
VMAX=4e7
COLLISION=1
REL_BUFFER=1.03
# QE=1
# ME=1
srun ./Langevin FFT 1.0 2.0 \
${NM} 0.001774 0.01 ${NP} 0.0 1e6 2.15623e-13 1000 1 1 1.0 1  ${epsinv} ${NV} ${VMAX} ${REL_BUFFER} \
1 1 1 1 1 1 1 ${COLLISION}   --info 5   1>${OUT_DIR}/langevin.out 2>${OUT_DIR}/langevin.err 
