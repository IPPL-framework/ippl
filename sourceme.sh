module reset
module use /soft/modulefiles
module switch PrgEnv-nvhpc PrgEnv-gnu
module load spack-pe-base cmake
#module load kokkos/4.2.01
module load cudatoolkit-standalone
module load visualization/ascent

export CXX=CC
export CC=cc
