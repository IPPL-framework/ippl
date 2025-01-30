#!/bin/bash

export CXX=CC
export CC=cc

cmake -B build-openmp -S . \
  -DCMAKE_INSTALL_PREFIX=install-ippl \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=CC \
  -DCMAKE_C_COMPILER=cc \
  -DKokkos_ENABLE_CUDA=OFF \
  -DKokkos_ENABLE_OPENMP=ON \
  -DCMAKE_CXX_STANDARD=20 \
  -DENABLE_FFT=ON \
  -DENABLE_SOLVERS=ON \
  -DENABLE_ALPINE=True \
  -DENABLE_TESTS=ON \
  -DUSE_ALTERNATIVE_VARIANT=ON \
  -DIPPL_PLATFORMS=openmp \
  -DAscent_DIR=`pwd`/../ascent/scripts/build_ascent/install/ascent-checkout/lib/cmake/ascent 


cmake --build build-openmp -j8
