KE=3.05e9
NM=64
NV=32
NP=156055
vmax=1e7
COLLISION=0
# QE=1
# ME=1

srun ./Langevin FFT 1.0 2.0 \
${NM} 0.001774 0.01 ${NP} 0.0 1e6 2.15623e-13 1000 1 1 1.0 1  ${KE} ${NV} 4e7 \
0 1 1 1 1 1 1 ${COLLISION}   --info 26   1>data/langevin.out 2>data/langevin.err 


