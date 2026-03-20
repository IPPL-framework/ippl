
# the -e options means "quit on the first encountered error"
set -e

mkdir -p heffte_post_install_test
cd heffte_post_install_test

rm -f CMakeCache.txt

heffte_mpic_compiler=""
heffte_mpif_compiler=""
if [[ ! -z "" ]]; then
    heffte_mpic_compiler="-DMPI_C_COMPILER="
fi
if [[ ! -z "" ]]; then
    heffte_mpif_compiler="-DMPI_Fortran_COMPILER="
fi

/p/software/default/stages/2026/software/CMake/4.0.3-GCCcore-14.3.0/bin/cmake \
    -DCMAKE_CXX_COMPILER=/p/software/default/stages/2026/software/GCCcore/14.3.0/bin/c++ \
    -DCMAKE_CXX_FLAGS="" \
    -DHeffte_DIR=/p/home/jusers/mostafa3/juwels/ippl/build_gpu/lib64/cmake/Heffte \
    -DMPI_CXX_COMPILER="/p/software/default/stages/2026/software/OpenMPI/5.0.8-GCC-14.3.0/bin/mpicxx" \
    -DMPIEXEC_EXECUTABLE="/usr/bin/srun" \
    $heffte_mpic_compiler \
    $heffte_mpif_compiler \
    -DMPIEXEC_NUMPROC_FLAG="-n" \
    -DMPIEXEC_PREFLAGS="" \
    -DMPIEXEC_POSTFLAGS="" \
    /p/home/jusers/mostafa3/juwels/ippl/build_gpu/share/heffte/testing/

make -j4
ctest -V
