cmake -S . -B build_gpu -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_STANDARD=20 \
         -DIPPL_PLATFORMS=CUDA \
         -DKokkos_ARCH_AMPERE80=ON \
         -DIPPL_ENABLE_FFT=ON \
         -DIPPL_ENABLE_TESTS=ON \
         -DIPPL_ENABLE_SOLVERS=ON \
         -DIPPL_ENABLE_ALPINE=ON \
         -DIPPL_USE_ALTERNATIVE_VARIANT=ON \
         -DHeffte_DIR=$Heffte_DIR ;

cmake --build build_gpu -j4
