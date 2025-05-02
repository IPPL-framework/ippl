#!/bin/bash

export CXX=mpiCC
export CC=mpicc

cmake -B build-cuda-sophia -S . \
  -DCMAKE_INSTALL_PREFIX=install-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=mpiCC \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CUDA_HOST_COMPILER=mpiCC \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_AMPERE80=ON \
  -DKokkos_ENABLE_OPENMP=ON \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DENABLE_FFT=ON \
  -DENABLE_SOLVERS=ON \
  -DENABLE_ALPINE=True \
  -DENABLE_TESTS=ON \
  -DUSE_ALTERNATIVE_VARIANT=ON \
  -DIPPL_PLATFORMS=cuda \
  -DAscent_DIR=`pwd`/../../ascent/scripts/build_ascent/install/ascent-checkout/lib/cmake/ascent 


cmake --build build-cuda-sophia --target install -j8
