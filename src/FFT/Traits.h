/*!
 * @file Traits.h
 * @brief Compile-time traits, tags and dispatch helpers for IPPL's FFT layer.
 *
 * Provides:
 *   - Transform tag types (CCTransform, RCTransform, ...) used to select an
 *     FFT specialization.
 *   - Backend feature tags (FFTW, MKL, CuFFT, RocFFT, HeffteGPU, CuFFTMp,
 *     Finufft, GPUFinufft) along with @c is_available_v<Feature> compile-time
 *     queries.
 *   - HeffteBackend selects the heFFTe backend type for a given Kokkos
 *     memory space (host vs CUDA vs HIP vs SYCL).
 *   - Stream provides minimal RAII-style helpers for execution-space streams.
 */
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
    //! @name FFT transform tag types
    //! Empty tag types selecting an FFT specialization at compile time.
    //! @{
    struct CCTransform {};         //!< Complex-to-complex.
    struct RCTransform {};         //!< Real-to-complex (and inverse).
    struct SineTransform {};       //!< Discrete sine transform.
    struct CosTransform {};        //!< Discrete cosine transform (Type II).
    struct Cos1Transform {};       //!< Discrete cosine transform variant (Type I).
    struct NUFFTransform {};       //!< Non-uniform FFT (Type 1 / Type 2).
    struct PrunedCCTransform {};   //!< Pruned C2C (low-mode Fourier truncation).
    struct PrunedRCTransform {};   //!< Pruned R2C.
    //! @}

    //! Direction of a forward / backward transform.
    enum TransformDirection {
        FORWARD,
        BACKWARD
    };

    //! Communication algorithm for distributed transforms (used by makeHeffteOptions).
    enum FFTComm {
        a2a    = 0, //!< MPI_Alltoall.
        a2av   = 1, //!< MPI_Alltoallv.
        p2p    = 2, //!< Point-to-point.
        p2p_pl = 3  //!< Point-to-point pipelined.
    };

    /*!
     * @struct PruningParams
     * @brief Per-axis kept-mode counts for pruned FFTs.
     * @tparam Dim Spatial dimension.
     */
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

    //! Primary FFT template; specialized per transform tag in FFT/Transform/*.
    template <typename Transform, typename Field>
    class FFT;

    namespace fft {

        //=============================================================================
        // Feature Tags
        //=============================================================================

        //! @name FFT backend / library feature tags.
        //! Used as template arguments to is_available to compile-time test for
        //! optional FFT libraries.
        //! @{
        struct FFTW {};        //!< FFTW host library.
        struct MKL {};         //!< Intel MKL host FFT.
        struct CuFFT {};       //!< NVIDIA cuFFT (single-GPU).
        struct RocFFT {};      //!< AMD rocFFT.
        struct HeffteGPU {};   //!< heFFTe with any GPU backend enabled.
        struct CuFFTMp {};     //!< NVIDIA cuFFTMp (multi-GPU/-node).
        struct Finufft {};     //!< Host finufft library.
        struct GPUFinufft {};  //!< GPU finufft library.
        //! @}

        //=============================================================================
        // Unified Feature Detection: is_available<Feature>
        //=============================================================================

        //! Generic `false_type`; specializations below set `value = true` when
        //! the corresponding feature is enabled at configure time.
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

        //! Convenience alias: `is_available_v<F>` is true iff `F` is enabled.
        template <typename Feature>
        inline constexpr bool is_available_v = is_available<Feature>::value;

        //=============================================================================
        // heFFTe Backend Selection by Memory Space
        //=============================================================================

        /*!
         * @struct HeffteBackend
         * @brief Pick the appropriate heFFTe backend types for a memory space.
         *
         * Provides nested aliases @c c2c, @c sin, @c cos, @c cos1 for the
         * complex / sine / cosine / cosine-Type-I transforms. Specializations
         * select FFTW / MKL on the host, cuFFT on CUDA, rocFFT on HIP, and
         * fall back to the heFFTe stock backend everywhere else.
         *
         * @tparam MemSpace Kokkos memory space.
         */
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

#ifdef KOKKOS_ENABLE_SYCL
        // No SYCL-specific heFFTe backend wired up yet. Heffte's oneMKL
        // backend (Heffte_ENABLE_ONEAPI) would go here once IPPL's
        // Dependencies.cmake plumbs it through. For now fall back to the
        // stock CPU backend so SYCL builds at least compile (FFTs run on
        // the host).
        template <>
        struct HeffteBackend<Kokkos::SYCLDeviceUSMSpace> {
            using c2c  = heffte::backend::stock;
            using sin  = heffte::backend::stock_sin;
            using cos  = heffte::backend::stock_cos;
            using cos1 = heffte::backend::stock_cos1;
        };
#endif

        //=============================================================================
        // GPU Stream Support
        //=============================================================================

        /*!
         * @struct Stream
         * @brief Minimal stream wrapper used to launch FFT-related kernels.
         *
         * The primary template is a no-op for memory spaces without a native
         * stream concept; specializations below provide CUDA and HIP streams.
         *
         * @tparam MemSpace Kokkos memory space.
         */
        template <typename MemSpace>
        struct Stream {
            using stream_type = int;  //!< Dummy handle for backends without streams.
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

            static void create(stream_type& s) { (void)hipStreamCreate(&s); }
            static void destroy(stream_type& s) { (void)hipStreamDestroy(s); }
            static void sync(stream_type& s) { (void)hipStreamSynchronize(s); }
            static exec_space instance(stream_type& s) { return exec_space(s); }
        };
#endif

        //=============================================================================
        // FFTW Trig Scaling
        //=============================================================================

        /*!
         * @brief Extra normalization factor applied by FFTW's trig transforms.
         *
         * FFTW's sine / cosine transforms include an implicit factor of 8 in
         * 3D (2 per axis); other backends do not. The trig wrappers multiply
         * by this to keep the normalization consistent across backends.
         */
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
