mkdir -p build && cd build && rm -rf *
export MPICH_GPU_SUPPORT_ENABLED=1
cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=20 \
      -DCMAKE_CXX_COMPILER=hipcc \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_HIP_FLAGS=--offload-arch=gfx90a \
      -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON \
      -DKokkos_ENABLE_DEBUG=OFF \
      -DKokkos_ARCH_ZEN3=ON \
      -DKokkos_ARCH_AMD_GFX90A=ON \
      -DKokkos_ENABLE_HIP=ON \
      -DIPPL_PLATFORMS="HIP;OPENMP" \
      -DIPPL_ENABLE_HIP_PROFILER=OFF \
      -DENABLE_TESTS=ON \
      -DENABLE_FFT=ON  \
      -DENABLE_SOLVERS=ON \
      -DENABLE_ALPINE=OFF \
      -DHeffte_ENABLE_ROCM=ON\
      -DHeffte_ENABLE_GPU_AWARE_MPI=ON \
      -DCMAKE_EXE_LINKER_FLAGS="-L/opt/cray/pe/mpich/8.1.28/ofi/amd/5.0/lib -L/opt/cray/pe/mpich/8.1.28/gtl/lib -L/opt/cray/pe/libsci/24.03.0/AMD/5.0/x86_64/lib -L/opt/cray/pe/dsmml/0.3.0/dsmml/lib -L/opt/cray/xpmem/2.8.2-1.0_5.1__g84a27a5.shasta/lib64 -lsci_amd_mpi -lsci_amd -ldl -lmpi_amd -lmpi_gtl_hsa -ldsmml -lxpmem -L/opt/rocm-6.0.3/lib/lib -L/opt/rocm-6.0.3/lib/lib64 -L/opt/rocm-6.0.3/lib/llvm/lib"





# In case -DIPPL_ENABLE_HIP_PROFILER=ON check 
# https://hackmd.io/@sfantao/lumi-training-sto-2025#LUMI-Training---Stockholm-Sweden-March-2025

