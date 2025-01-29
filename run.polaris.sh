#!/bin/bash

NNODES=`wc -l < $PBS_NODEFILE` 
NRANKS_PER_NODE=4
NDEPTH=8 
NTHREADS=8

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
mpirun --np ${NTOTRANKS} -ppn ${NRANKS} -d ${NDEPTH} -env OMP_NUM_THREADS=${NTHREADS} depth ./PenningTrap 32 32 32 655360 400 FFT 0.01 LeapFrog --overallocate 1.0 --info 5 --frequency 2
