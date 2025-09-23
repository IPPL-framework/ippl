#!/bin/bash
set -euo pipefail


CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CXX_STANDARD=20 
  -DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON
  -DMPI_C_COMPILER="${MPICC}"
  -DMPI_CXX_COMPILER="${MPICXX}"
  -DMPIEXEC_EXECUTABLE="${MPIEXEC}"
)
CMAKE_ARGS+=(
  -DIPPL_ENABLE_FFT=ON 
  -DIPPL_ENABLE_SOLVERS=ON 
  -DIPPL_ENABLE_ALPINE=ON 
  -DIPPL_ENABLE_TESTS=OFF 
)


# #   example for ascent on jülich, maybe want to use newer /maybe no hint needed
# #   /p/software/default/stages/2024/software/Ascent/20240122-gpsmpi-2023a/lib64/cmake/ascent
CATALYST_CMAKE_PATH="/.../catalyst/install/lib/cmake/catalyst-2.0"
ASCENT_CMAKE_PATH="/.../alpine_ascent/ascent/build/install/ascent-checkout/lib/cmake/ascent"
echo " Passing CMake Catalyst Hint $CATALYST_CMAKE_PATH"
echo " Passing CMake Ascent   Hint $ASCENT_CMAKE_PATH"



CMAKE_ARGS+=(
    -DIPPL_ENABLE_CATALYST=ON
    -DCATALYST_HINT_PATH=${CATALYST_CMAKE_PATH}
)
CMAKE_ARGS+=(
  -DIPPL_ENABLE_ASCENT=ON 
  -DASCENT_HINT_PATH=${ASCENT_CMAKE_PATH}
)
  


# Fresh build
rm -rf build
mkdir build
cd build
cmake "${CMAKE_ARGS[@]}" ..
cd ..