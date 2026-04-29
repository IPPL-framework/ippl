#ifndef IPPL_FFT_BACKEND_H
#define IPPL_FFT_BACKEND_H

#include "FFT/Backend/Heffte.h"

#ifdef IPPL_ENABLE_CUFFTMP
#include "FFT/Backend/CuFFTMp.h"
#endif

#ifdef KOKKOS_ENABLE_CUDA
#include "FFT/Backend/CuFFT.h"
#endif

#endif  // IPPL_FFT_BACKEND_H
