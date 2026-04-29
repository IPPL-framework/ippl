#ifndef IPPL_FFT_TRAITS_H
#define IPPL_FFT_TRAITS_H

#include <Kokkos_Core.hpp>

#ifdef IPPL_ENABLE_CUFFTMP
#include <cufftMp.h>
#endif

#include <heffte_fft3d.h>
#include <heffte_fft3d_r2c.h>
#include <type_traits>

namespace ippl {
    // Transform tags
    struct CCTransform {};
    struct RCTransform {};
    struct SineTransform {};
    struct CosTransform {};
    struct Cos1Transform {};
    struct NUFFTransform {};
    struct PrunedCCTransform {};
    struct PrunedRCTransform {};

    // Direction
    enum TransformDirection {
        FORWARD,
        BACKWARD
    };

    // Communication strategy
    enum FFTComm {
        a2a    = 0,
        a2av   = 1,
        p2p    = 2,
        p2p_pl = 3
    };

    // Pruning parameters
    template <unsigned Dim>
    struct PruningParams {
        Vector<std::size_t, Dim> n_modes{};

        PruningParams() = default;

        template <typename Vec>
        explicit PruningParams(const Vec& modes) {
            for (unsigned d = 0; d < Dim; ++d) {
                n_modes[d] = modes[d];
            }
        }
    };

    // Primary template - specialized per transform type
    template <typename Transform, typename Field>
    class FFT;

    namespace fft {

        //=============================================================================
        // Feature Tags
        //=============================================================================

        struct FFTW {};
        struct MKL {};
        struct CuFFT {};
        struct RocFFT {};
        struct HeffteGPU {};
        struct CuFFTMp {};
        struct Finufft {};
        struct GPUFinufft {};

        //=============================================================================
        // Unified Feature Detection: is_available<Feature>
        //=============================================================================

        template <typename Feature>
        struct is_available : std::false_type {};

#ifdef Heffte_ENABLE_FFTW
        template <>
        struct is_available<FFTW> : std::true_type {};
#endif

#ifdef Heffte_ENABLE_MKL
        template <>
        struct is_available<MKL> : std::true_type {};
#endif

#ifdef Heffte_ENABLE_CUDA
        template <>
        struct is_available<CuFFT> : std::true_type {};
#endif

#ifdef Heffte_ENABLE_ROCM
        template <>
        struct is_available<RocFFT> : std::true_type {};
#endif

#ifdef Heffte_ENABLE_GPU
        template <>
        struct is_available<HeffteGPU> : std::true_type {};
#endif

#ifdef IPPL_ENABLE_CUFFTMP
        template <>
        struct is_available<CuFFTMp> : std::true_type {};
#endif

#ifdef ENABLE_FINUFFT
        template <>
        struct is_available<Finufft> : std::true_type {};
#endif

#if defined(ENABLE_FINUFFT) && defined(ENABLE_GPU_NUFFT)
        template <>
        struct is_available<GPUFinufft> : std::true_type {};
#endif

        template <typename Feature>
        inline constexpr bool is_available_v = is_available<Feature>::value;

        //=============================================================================
        // heFFTe Backend Selection by Memory Space
        //=============================================================================

        template <typename MemSpace>
        struct HeffteBackend {
            // Default: stock backend
            using c2c  = heffte::backend::stock;
            using sin  = heffte::backend::stock_sin;
            using cos  = heffte::backend::stock_cos;
            using cos1 = heffte::backend::stock_cos1;
        };

        // Host: FFTW > MKL > Stock
#if defined(Heffte_ENABLE_FFTW)
        template <>
        struct HeffteBackend<Kokkos::HostSpace> {
            using c2c  = heffte::backend::fftw;
            using sin  = heffte::backend::fftw_sin;
            using cos  = heffte::backend::fftw_cos;
            using cos1 = heffte::backend::fftw_cos1;
        };
#elif defined(Heffte_ENABLE_MKL)
        template <>
        struct HeffteBackend<Kokkos::HostSpace> {
            using c2c  = heffte::backend::mkl;
            using sin  = heffte::backend::mkl_sin;
            using cos  = heffte::backend::mkl_cos;
            using cos1 = heffte::backend::mkl_cos1;
        };
#endif

#ifdef KOKKOS_ENABLE_CUDA
        template <>
        struct HeffteBackend<Kokkos::CudaSpace> {
            using c2c  = heffte::backend::cufft;
            using sin  = heffte::backend::cufft_sin;
            using cos  = heffte::backend::cufft_cos;
            using cos1 = heffte::backend::cufft_cos1;
        };
#endif

#ifdef KOKKOS_ENABLE_HIP
        template <>
        struct HeffteBackend<Kokkos::HIPSpace> {
#ifdef Heffte_ENABLE_ROCM
            using c2c  = heffte::backend::rocfft;
            using sin  = heffte::backend::rocfft_sin;
            using cos  = heffte::backend::rocfft_cos;
            using cos1 = heffte::backend::rocfft_cos1;
#else
            using c2c  = heffte::backend::stock;
            using sin  = heffte::backend::stock_sin;
            using cos  = heffte::backend::stock_cos;
            using cos1 = heffte::backend::stock_cos1;
#endif
        };
#endif

        //=============================================================================
        // GPU Stream Support
        //=============================================================================

        template <typename MemSpace>
        struct Stream {
            using stream_type = int;  // Dummy
            using exec_space  = Kokkos::DefaultExecutionSpace;

            static void create(stream_type&) {}
            static void destroy(stream_type&) {}
            static void sync(stream_type&) {}
            static exec_space instance(stream_type&) { return exec_space(); }
        };

#ifdef KOKKOS_ENABLE_CUDA
        template <>
        struct Stream<Kokkos::CudaSpace> {
            using stream_type = cudaStream_t;
            using exec_space  = Kokkos::Cuda;

            static void create(stream_type& s) { cudaStreamCreate(&s); }
            static void destroy(stream_type& s) { cudaStreamDestroy(s); }
            static void sync(stream_type& s) { cudaStreamSynchronize(s); }
            static exec_space instance(stream_type& s) { return exec_space(s); }
        };
#endif

#ifdef KOKKOS_ENABLE_HIP
        template <>
        struct Stream<Kokkos::HIPSpace> {
            using stream_type = hipStream_t;
            using exec_space  = Kokkos::HIP;

            static void create(stream_type& s) { hipStreamCreate(&s); }
            static void destroy(stream_type& s) { hipStreamDestroy(s); }
            static void sync(stream_type& s) { hipStreamSynchronize(s); }
            static exec_space instance(stream_type& s) { return exec_space(s); }
        };
#endif

        //=============================================================================
        // FFTW Trig Scaling
        //=============================================================================

        inline constexpr double fftw_trig_scale() {
            return is_available_v<FFTW> ? 8.0 : 1.0;
        }
    }  // namespace fft

}  // namespace ippl

// Register Kokkos complex with heFFTe
namespace heffte {
    template <>
    struct is_ccomplex<Kokkos::complex<float>> : std::true_type {};
    template <>
    struct is_zcomplex<Kokkos::complex<double>> : std::true_type {};
}  // namespace heffte

#endif