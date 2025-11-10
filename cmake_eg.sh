#!/bin/bash
set -euo pipefail

CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CXX_STANDARD=20 
  -DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON
#  -DIPPL_PLATFORMS=openmp
#  -DMPI_C_COMPILER="${MPICC}"
#  -DMPI_CXX_COMPILER="${MPICXX}"
#  -DMPIEXEC_EXECUTABLE="${MPIEXEC}"
)
CMAKE_ARGS+=(
  -DIPPL_ENABLE_FFT=ON 
  -DIPPL_ENABLE_SOLVERS=ON 
  -DIPPL_ENABLE_ALPINE=ON 
  -DIPPL_ENABLE_TESTS=OFF 
)


#####################################################################################
# When running on jülich systems these should not be necessary, when building locally
# depending on your install you might need to pass hints to cmake. These might follow
# the following form:
#####################################################################################
# CATALYST_CMAKE_PATH="/.../catalyst/install/lib/cmake/catalyst-2.0"
# ASCENT_CMAKE_PATH="/.../ascent/build/install/ascent-checkout/lib/cmake/ascent"
# echo " Passing CMake Catalyst Hint $CATALYST_CMAKE_PATH"
# echo " Passing CMake Ascent   Hint $ASCENT_CMAKE_PATH"
#####################################################################################
# When build fails on jülich, when including with ascent you can also try pass hints
# which will be something like the following (depending on loaded module).
# /p/software/default/stages/2024/software/Ascent/20240122-gpsmpi-2023a/lib64/cmake/ascent
#####################################################################################
# On Clusters or/and on some manual installs a full Catalyst 2.0 is included in 
# ParaView installation. but when only downloading binaries, one might have to 
# additionally install the basic Catalyst 2.0 (libcatalyst), with the same MPI (!!) 
# variant as the ParaView binary. If build fails at including Catalyst one might need 
# to pass adequat hint to which basic Catalyst Version should be linked.
#####################################################################################

CMAKE_ARGS+=(
    -DIPPL_ENABLE_CATALYST=ON
#    -DCATALYST_HINT_PATH=${CATALYST_CMAKE_PATH}
)
CMAKE_ARGS+=(
  -DIPPL_ENABLE_ASCENT=OFF
#  -DASCENT_HINT_PATH=${ASCENT_CMAKE_PATH}
)
  


# Fresh build
rm -rf build
mkdir build
cd build
cmake "${CMAKE_ARGS[@]}" ..
# cmake --build . -j
cd ..