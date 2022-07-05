export OTB_TOOLSET=clang
export OTB_COMPILER_VERSION=$(clang --version | head -1)
export OTB_MPI=mpich

declare -a OTB_RECIPES=(
    040-build-kokkos
    050-build-heffte
    400-build-ippl)

