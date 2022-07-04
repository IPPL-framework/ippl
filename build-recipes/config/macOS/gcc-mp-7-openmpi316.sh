export OTB_TOOLSET=gcc
export OTB_COMPILER_VERSION=gcc-mp-7
export OTB_MPI=openmpi
export OTB_MPI_VERSION=3.1.6

declare -a OTB_RECIPES=(
	050-build-cmake
	060-build-openmpi
	070-build-zlib
	080-build-hdf5
	090-build-gsl
	100-build-h5hut
	110-build-boost
	200-build-parmetis
	210-build-openblas
	220-build-trilinos
	230-build-amrex
	300-build-gtest)

declare -A OTB_SYMLINKS
OTB_SYMLINKS[bin/cc]="/opt/local/bin/${OTB_COMPILER_VERSION}"
OTB_SYMLINKS[bin/gcc]="/opt/local/bin/${OTB_COMPILER_VERSION}"
OTB_SYMLINKS[bin/c++]="/opt/local/bin/${OTB_COMPILER_VERSION/gcc/g++}"
OTB_SYMLINKS[bin/g++]="/opt/local/bin/${OTB_COMPILER_VERSION/gcc/g++}"
OTB_SYMLINKS[bin/gfortran]="/opt/local/bin/${OTB_COMPILER_VERSION/gcc/gfortran}"

