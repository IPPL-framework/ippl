#!/bin/bash

NNODES=`wc -l < $PBS_NODEFILE` 
NRANKS=1
NDEPTH=32 
NTHREADS=32

NTOTRANKS=$(( NNODES * NRANKS ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS} THREADS_PER_RANK= ${NTHREADS}"

mpirun --np ${NTOTRANKS} -ppn ${NRANKS} -d ${NDEPTH} --cpu-bind depth -env OMP_NUM_THREADS=${NTHREADS} ./PenningTrap 32 32 32 655360 400 FFT 0.01 LeapFrog --overallocate 1.0 --info 5 --frequency 2