export OTB_TOOLSET=clang
export OTB_COMPILER_VERSION=$(clang --version | head -1)
export OTB_MPI=mpich

declare -a OTB_RECIPES=(
	050-build-cmake
	061-build-mpich
	070-build-hdf5
	080-build-gsl
	090-build-h5hut
	100-build-zlib
	110-build-boost
	200-build-parmetis
	210-build-openblas
	220-build-trilinos
	300-build-gtest)

