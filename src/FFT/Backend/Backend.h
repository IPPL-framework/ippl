/*!
 * @file Backend.h
 * @brief Aggregate include for all FFT backends supported by IPPL.
 *
 * Pulls in the heFFTe wrappers unconditionally, plus optional CUDA-only
 * (CuFFT) and CUDA+MPI (CuFFTMp) backends gated by the corresponding
 * Kokkos / IPPL configuration macros.
 */
#ifndef IPPL_FFT_BACKEND_H
#define IPPL_FFT_BACKEND_H

#include "FFT/Backend/Heffte.h"

#if defined(IPPL_ENABLE_CUFFTMP) && defined(KOKKOS_ENABLE_CUDA)
#include "FFT/Backend/CuFFTMp.h"
#endif

#ifdef KOKKOS_ENABLE_CUDA
#include "FFT/Backend/CuFFT.h"
#endif

#endif  // IPPL_FFT_BACKEND_H
