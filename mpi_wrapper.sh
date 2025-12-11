#!/usr/bin/env bash

# last edit 2025-09-19 by JB
# Warning - this wrapper script won't work for N>1 GPUs per rank

# -----------------------------------------------------------------------------
# detect cray mpich or openmpi installed in uenv
# uenv view will set MPICC to an MPI install path
# -----------------------------------------------------------------------------
mpicc_path=$(command -v $MPICC)
if [ -n "$mpicc_path" ]; then
    mpicc_realpath=$(readlink -f "$mpicc_path")
    if [[ "$mpicc_realpath" == *cray* ]]; then
        # echo "Wrapper script using Cray MPICH at $mpicc_realpath"
        CRAY_MPICH=1
    fi
    if [[ "$mpicc_realpath" == *openmpi* ]]; then
        # echo "Wrapper script using OpenMPI at $mpicc_realpath"
        OPENMPI=1
    fi
else
    echo "Error: mpicc not found in PATH" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# osu benchmark sanity checks
# here are example commands to run osu benchmarks using either srun, or mpirun
#   when using cray-mpich: only srun is allowed
#   when using openmpi   : either srun or mpirun should work
# large message size expected BW outputs for openmpi runs are included for reference
# we use $NTHREADS cores per rank to illustrate openmpi binding syntax
# (add export OMP_NUM_THREADS=$NTHREADS (or whatever) when using openmp)
# --------------------------------------------

# --------------------------------------------
# example: cuda GPU -> GPU : 2 gpus on same node
#
# mpirun --map-by ppr:1:l3cache:PE=$NTHREADS --bind-to core --np 2         $WRAPPER osu_bw -d cuda D D
# srun -v -N 1 -n 2 --mpi=pmix --cpus-per-task=$NTHREADS --gpus-per-task=1 $WRAPPER osu_bw -d cuda D D
# ...
# 524288             115497.90
# 1048576            145525.76
# 2097152            165970.40
# 4194304            180612.88

# --------------------------------------------
# example: CPU -> CPU : 2 cpus on same node (different numa)
#
# mpirun --map-by ppr:1:l3cache:PE=$NTHREADS --bind-to core --np 2         $WRAPPER osu_bw
# srun -v -N 1 -n 2 --mpi=pmix --cpus-per-task=$NTHREADS --gpus-per-task=1 $WRAPPER osu_bw
# ...
# 524288               6213.87
# 1048576              6405.66
# 2097152              6128.29
# 4194304              5962.13

# --------------------------------------------
# example: GPU->CPU copy : GPU/CPU on same node (different numa)
#
# mpirun --map-by ppr:1:l3cache:PE=$NTHREADS --bind-to core --np 2           $WRAPPER osu_bw -d cuda D H
# srun -v -N 1 -n 2 --mpi=pmix --cpus-per-task=$NTHREADS --gpus-per-task=1   $WRAPPER osu_bw -d cuda D H
# ...
# 524288              38731.44
# 1048576             58549.66
# 2097152             80907.46
# 4194304            100265.34

# --------------------------------------------
# example: CPU->CPU : different nodes
#
# mpirun --map-by ppr:1:node:PE=$NTHREADS --bind-to core --np 2            $WRAPPER osu_bw
# srun -v -N 2 -n 2 --mpi=pmix --cpus-per-task=$NTHREADS --gpus-per-task=1 $WRAPPER osu_bw
# ...
# 524288              23749.11
# 1048576             23890.81
# 2097152             23953.50
# 4194304             23994.22

# --------------------------------------------
# example: GPU->GPU : 2 different nodes
#
# mpirun --map-by ppr:1:node:PE=$NTHREADS --bind-to core --np 2            $WRAPPER osu_bw -d cuda D D
# srun -v -N 2 -n 2 --mpi=pmix --cpus-per-task=$NTHREADS --gpus-per-task=1 $WRAPPER osu_bw -d cuda D D
# ...
# 524288              23596.17
# 1048576             23803.42
# 2097152             23916.79
# 4194304             23965.88

# --------------------------------------------
# these are the same examples as above but collected for easy cut'n'paste into terminal
# --------------------------------------------
# export WRAPPER=/user-environment/wrapper-mpi.sh
# NTHREADS=8
# mpirun --map-by ppr:1:l3cache:PE=$NTHREADS --bind-to core --np 2         $WRAPPER osu_bw -d cuda D D
# srun -v -N 1 -n 2 --mpi=pmix --cpus-per-task=$NTHREADS --gpus-per-task=1 $WRAPPER osu_bw -d cuda D D
# mpirun --map-by ppr:1:l3cache:PE=$NTHREADS --bind-to core --np 2         $WRAPPER osu_bw
# srun -v -N 1 -n 2 --mpi=pmix --cpus-per-task=$NTHREADS --gpus-per-task=1 $WRAPPER osu_bw
# mpirun --map-by ppr:1:l3cache:PE=$NTHREADS --bind-to core --np 2         $WRAPPER osu_bw -d cuda D H
# srun -v -N 1 -n 2 --mpi=pmix --cpus-per-task=$NTHREADS --gpus-per-task=1 $WRAPPER osu_bw -d cuda D H
# mpirun --map-by ppr:1:node:PE=$NTHREADS --bind-to core --np 2            $WRAPPER osu_bw
# srun -v -N 2 -n 2 --mpi=pmix --cpus-per-task=$NTHREADS --gpus-per-task=1 $WRAPPER osu_bw
# mpirun --map-by ppr:1:node:PE=$NTHREADS --bind-to core --np 2            $WRAPPER osu_bw -d cuda D D
# srun -v -N 2 -n 2 --mpi=pmix --cpus-per-task=$NTHREADS --gpus-per-task=1 $WRAPPER osu_bw -d cuda D D

# ---------------
# (Optionally) Disable core dumps
# ---------------
ulimit -c 0

# ---------------
# (Optionally) Kill any child processes when this script is terminated
# hopefully reduces the chance of orphaned processes on compute nodes
# this tends to be a problem when running ctest and something fails,
# then all remaining tests can fail since the node isn't cleaned up
# ---------------
trap "kill 0" SIGINT SIGTERM

# -----------------------------------------------------------------------------
# get cpu affinity of current process,
# use openmpi or slurm vars for compatibility with flavours of mpi invocation
# -----------------------------------------------------------------------------
cpus=$(taskset -pc $$ | awk '{print $6}')
# Identify the NUMA nodes intersected by CPU affinity
numa_nodes=$(hwloc-calc --physical --intersect NUMAnode $(taskset -p $$ | awk '{print "0x"$6}'))
# the GPU is also reported as a numa node, so strip off the first numa reported and use that
IFS=',' read -r first_numa other_nodes <<< "$numa_nodes"

# The first numa node (in the binding list) is the one we will use for GPU and NIC selection
gpu=$first_numa
nic="cxi${first_numa}"

lrank=0
grank=0
if [ -z ${OMPI_COMM_WORLD_LOCAL_RANK+x} ]
then
    let lrank=$SLURM_LOCALID
    let grank=$SLURM_PROCID
else
    let lrank=$OMPI_COMM_WORLD_LOCAL_RANK
    let grank=$OMPI_COMM_WORLD_RANK
fi

# Use SLURM_TASKS_PER_NODE to determine ranks per node if available
if [ -n "$SLURM_TASKS_PER_NODE" ]; then
    ranks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | cut -d'(' -f1)
else
    if [ -n "$SLURM_NTASKS" ] && [ -n "$SLURM_JOB_NUM_NODES" ]; then
        ranks_per_node=$((SLURM_NTASKS / SLURM_JOB_NUM_NODES))
    else
        ranks_per_node=1
    fi
fi

if [[ $grank == 0 ]]
then
    echo "Slurm Job Hostlist: $SLURM_JOB_NODELIST"
fi

# ---------------
# (optionally) print out helpful binding information to see what we extracted
# ---------------
printf "Hostname=%-12s, Rank=%-4d ,Local=%-3d ,RPN=%-3d ,CPUs=%-8s ,GPU=%-1s ,NIC=%-4s ,numa_nodes=%-5s ,first_numa=%-2s ,other_nodes=%-5s\n" \
    "$(hostname)" "$grank" "$lrank" "$ranks_per_node" "$cpus" "$gpu" "$nic" "$numa_nodes" "$first_numa" "$other_nodes"

# ---------------
# GPU selection env var used by nvidia boilerplate
# ---------------
# (paul) This causes CUDA aware MPI to fail.
#export CUDA_VISIBLE_DEVICES=$gpu

# ---------------
#  cray-mpich : see https://cpe.ext.hpe.com/docs/24.03/mpt/mpich/intro_mpi.html#general-mpich-environment-variables
# ---------------
# GPU support
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_IPC_ENABLED=1
export MPICH_GPU_IPC_CACHE_MAX_SIZE=128

export MPICH_GPU_IPC_THRESHOLD=1
# debug info

# ---------------
# OpenMPI mappings for MCA variables
# note that if we did not compile openmpi with ucx/tcp/infiniband/etc then turning these off isn't necessary
# ---------------
# set the address vector in libfabric to use table mappings
export OMPI_MCA_mtl_ofi_av=table
# disable these providers for the byte transport layer (just don't add support for them when compiling)
export OMPI_MCA_btl='^tcp,ofi,vader,openib'
# disable these providers for point-to-point messaging (don't compile ucx support in anyway)
export OMPI_MCA_pml='^ucx'
# enable libfabric OFI for the message transport layer
export OMPI_MCA_mtl=ofi
# tell libfabric that we will  be using LNX provider, valid values are "cxi", "lnx"
export OMPI_MCA_opal_common_ofi_provider_include=lnx
export OMPI_MCA_opal_common_ofi_provider_include=cxi
# Disable PMIx security (psec) component munge : should be fixed by building openmpi/pmix without munge support
export PMIX_MCA_psec=^munge

# ---------------
#  libfabric settings
# ---------------
export FI_LNX_OUTPUT_STATS_CSV=0

# linkx provider : for debugging - omit shm - only use cxi but with lnx layered over it.
# export FI_LNX_PROV_LINKS="cxi:cxi0|cxi:cxi1|cxi:cxi2|cxi:cxi3"

# linkx provider - this is the usual default : select cxi devices round robin from provided list
# export FI_LNX_PROV_LINKS="shm+cxi:cxi0|shm+cxi:cxi1|shm+cxi:cxi2|shm+cxi:cxi3"

# linkx provider - we will force one nic based on per rank numa/placemment that we computed above
export FI_LNX_PROV_LINKS="shm+cxi:$nic"

# If there are multiple cxi domains on a node, this sets the one to use per rank (for the cxi provider)
export FI_CXI_DEVICE_NAME=$nic

# don't disable SHM, we want it
export FI_OFI_RXM_ENABLE_SHM=1
export FI_SHM_USE_XPMEM=1
export FI_LNX_DISABLE_SHM=0

# make sure lnx shared receive queue is enabled
export FI_LNX_SRQ_SUPPORT=1

# values that can/should be tweaked depending on latest advice
export FI_CXI_RDZV_THRESHOLD=16384
export FI_CXI_RDZV_EAGER_SIZE=8192
export FI_CXI_OFLOW_BUF_SIZE=12582912
export FI_CXI_OFLOW_BUF_COUNT=3
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_REQ_BUF_MAX_CACHED=0
export FI_CXI_REQ_BUF_MIN_POSTED=6
export FI_CXI_REQ_BUF_SIZE=12582912

# when using openmpi with linkx provider, we need to set the RX tag matching mode
if [[ -n "$OPENMPI" ]]; then
    if [[ "$OMPI_MCA_opal_common_ofi_provider_include" == "lnx" ]]; then
        export FI_CXI_RX_MATCH_MODE=software
    else
        export FI_CXI_RX_MATCH_MODE=hardware
    fi
else
    export FI_CXI_RX_MATCH_MODE=hardware
fi

# MR_CACHE, recommended "userfaultfd" or "disabled"
export FI_MR_CACHE_MONITOR=userfaultfd
if [[ "$FI_MR_CACHE_MONITOR" == "disabled" ]]; then
    # set minimum (zero causes crashes - @TODO investigate)
    export FI_MR_CACHE_MAX_SIZE=1
    export FI_MR_CACHE_MAX_COUNT=1
else
    # set unlimited size and large storage count
    export FI_MR_CACHE_MAX_SIZE=-1
    export FI_MR_CACHE_MAX_COUNT=524288
fi

# ----------------------------------------------
# execute the real command
"$@"
