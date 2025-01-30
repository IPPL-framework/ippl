#!/bin/bash

export CXX=CC
export CC=cc

cmake -B build-cuda -S . \
  -DCMAKE_INSTALL_PREFIX=install-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=CC \
  -DCMAKE_C_COMPILER=cc \
  -DCMAKE_CUDA_HOST_COMPILER=cc \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_AMPERE80=ON \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF \
  -DCMAKE_CXX_STANDARD=20 \
  -DENABLE_FFT=ON \
  -DENABLE_SOLVERS=ON \
  -DENABLE_ALPINE=True \
  -DENABLE_TESTS=ON \
  -DUSE_ALTERNATIVE_VARIANT=ON \
  -DIPPL_PLATFORMS=cuda \
  -DAscent_DIR=`pwd`/../ascent/scripts/build_ascent/install/ascent-checkout/lib/cmake/ascent 


cmake --build build-cuda --target install -j8
