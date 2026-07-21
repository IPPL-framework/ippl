/*!
 * @file NUFFT.h
 * @brief Non-uniform FFT specialization (NUFFTransform tag).
 *
 * Provides a single FFT class that can dispatch to the native IPPL NUFFT
 * (Kokkos-based ES kernel + heFFTe) or to FINUFFT / cuFINUFFT when those
 * libraries are enabled at configure time.
 */
#ifndef IPPL_FFT_TRANSFORM_NUFFT_H
#define IPPL_FFT_TRANSFORM_NUFFT_H

#include <array>
#include <cmath>
#include <memory>

#include "Utility/IpplException.h"
#include "Utility/ParameterList.h"

#include "Communicate/Communicator.h"
#include "FFT/NUFFT/NativeNUFFT.h"
#include "FFT/Traits.h"
#include "FFT/Transform/Common.h"

#ifdef ENABLE_FINUFFT
#include <finufft.h>
#ifdef ENABLE_GPU_NUFFT
#include <cufinufft.h>
#endif
#endif

namespace ippl {

    // Forward declaration
    template <typename T, class... Properties>
    class ParticleAttrib;

    namespace detail {

#ifdef ENABLE_FINUFFT
        /*!
         * @struct FinufftTraits
         * @brief Type traits for FINUFFT backend selection.
         *
         * Provides a unified surface (ComplexType, PlanType, OptsType,
         * CountType, plus static makeplan/setpts/execute/destroy wrappers)
         * over the CPU @c finufft library and the GPU @c cufinufft library
         * so the rest of the code can be written backend-agnostic.
         *
         * @tparam T Real precision (float or double).
         */
        template <typename T>
        struct FinufftTraits;

#ifdef ENABLE_GPU_NUFFT
        template <>
        struct FinufftTraits<float> {
            using ComplexType = cuFloatComplex;
            using PlanType    = cufinufftf_plan;
            using OptsType    = cufinufft_opts;
            using CountType   = int;  // cufinufft uses int for point counts

            static void defaultOpts(OptsType* opts) { cufinufft_default_opts(opts); }

            static int makeplan(int type, int dim, int64_t* nmodes, int iflag, int ntransf,
                                float tol, PlanType* plan, OptsType* opts) {
                return cufinufftf_makeplan(type, dim, nmodes, iflag, ntransf, tol, plan, opts);
            }

            static int setpts(PlanType plan, CountType M, float* x, float* y, float* z,
                              CountType N, float* s, float* t, float* u) {
                return cufinufftf_setpts(plan, M, x, y, z, N, s, t, u);
            }

            static int execute(PlanType plan, ComplexType* c, ComplexType* f) {
                return cufinufftf_execute(plan, c, f);
            }

            static int destroy(PlanType plan) { return cufinufftf_destroy(plan); }
        };

        template <>
        struct FinufftTraits<double> {
            using ComplexType = cuDoubleComplex;
            using PlanType    = cufinufft_plan;
            using OptsType    = cufinufft_opts;
            using CountType   = int;

            static void defaultOpts(OptsType* opts) { cufinufft_default_opts(opts); }

            static int makeplan(int type, int dim, int64_t* nmodes, int iflag, int ntransf,
                                double tol, PlanType* plan, OptsType* opts) {
                return cufinufft_makeplan(type, dim, nmodes, iflag, ntransf, tol, plan, opts);
            }

            static int setpts(PlanType plan, CountType M, double* x, double* y, double* z,
                              CountType N, double* s, double* t, double* u) {
                return cufinufft_setpts(plan, M, x, y, z, N, s, t, u);
            }

            static int execute(PlanType plan, ComplexType* c, ComplexType* f) {
                return cufinufft_execute(plan, c, f);
            }

            static int destroy(PlanType plan) { return cufinufft_destroy(plan); }
        };

#else  // CPU FINUFFT

        template <>
        struct FinufftTraits<float> {
            using ComplexType = std::complex<float>;
            using PlanType    = finufftf_plan;
            using OptsType    = finufft_opts;
            using CountType   = int64_t;

            static void defaultOpts(OptsType* opts) { finufft_default_opts(opts); }

            static int makeplan(int type, int dim, int64_t* nmodes, int iflag, int ntransf,
                                float tol, PlanType* plan, OptsType* opts) {
                return finufftf_makeplan(type, dim, nmodes, iflag, ntransf, tol, plan, opts);
            }

            static int setpts(PlanType plan, CountType M, float* x, float* y, float* z,
                              CountType N, float* s, float* t, float* u) {
                return finufftf_setpts(plan, M, x, y, z, N, s, t, u);
            }

            static int execute(PlanType plan, ComplexType* c, ComplexType* f) {
                return finufftf_execute(plan, c, f);
            }

            static int destroy(PlanType plan) { return finufftf_destroy(plan); }
        };

        template <>
        struct FinufftTraits<double> {
            using ComplexType = std::complex<double>;
            using PlanType    = finufft_plan;
            using OptsType    = finufft_opts;
            using CountType   = int64_t;

            static void defaultOpts(OptsType* opts) { finufft_default_opts(opts); }

            static int makeplan(int type, int dim, int64_t* nmodes, int iflag, int ntransf,
                                double tol, PlanType* plan, OptsType* opts) {
                return finufft_makeplan(type, dim, nmodes, iflag, ntransf, tol, plan, opts);
            }

            static int setpts(PlanType plan, CountType M, double* x, double* y, double* z,
                              CountType N, double* s, double* t, double* u) {
                return finufft_setpts(plan, M, x, y, z, N, s, t, u);
            }

            static int execute(PlanType plan, ComplexType* c, ComplexType* f) {
                return finufft_execute(plan, c, f);
            }

            static int destroy(PlanType plan) { return finufft_destroy(plan); }
        };

#endif  // ENABLE_GPU_NUFFT
#endif  // ENABLE_FINUFFT

    }  // namespace detail

    /*!
     * @class FFT<NUFFTransform, RealField>
     * @brief Non-uniform FFT for IPPL particles + uniform Fourier modes.
     *
     * Supports both the native IPPL implementation and the FINUFFT
     * (CPU / GPU) backend. The runtime backend choice is controlled by the
     * @c "useFinufft" / @c "lockMethod" parameter keys plus the configure
     * macros @c ENABLE_FINUFFT and @c ENABLE_GPU_NUFFT.
     *
     * Type 1: non-uniform points -> uniform grid (spreading / adjoint).
     * Type 2: uniform grid -> non-uniform points (interpolation).
     *
     * @tparam RealField IPPL Field of real values used for grid sizing.
     */
    template <typename RealField>
    class FFT<NUFFTransform, RealField> {
    public:
        static constexpr unsigned Dim = RealField::dim;

        using T         = typename RealField::value_type;
        using Complex_t = Kokkos::complex<T>;
        using MemSpace  = typename RealField::memory_space;
        using ExecSpace = typename RealField::execution_space;
        using Layout_t  = FieldLayout<Dim>;

        using ComplexField =
            typename Field<Complex_t, Dim, typename RealField::Mesh_t,
                           typename RealField::Centering_t, ExecSpace>::uniform_type;

        using NativeNUFFT_t = nufft::NativeNUFFT<Dim, T, ExecSpace>;

#ifdef ENABLE_FINUFFT
        using Traits_t          = detail::FinufftTraits<T>;
        using FinufftComplex_t  = typename Traits_t::ComplexType;
        using FinufftPlan_t     = typename Traits_t::PlanType;
        using FinufftOpts_t     = typename Traits_t::OptsType;
        using FinufftCount_t    = typename Traits_t::CountType;
#endif

    private:
        // Configuration
        int type_m;
        T tol_m;
        bool useFinufft_m;
        bool useUpsampledInputs_m;
        bool useR2C_m;
        int r2cDir_m;
        bool lockMethod_m;

        std::array<int64_t, 3> nModes_m{1, 1, 1};

        // Native NUFFT backend
        std::unique_ptr<NativeNUFFT_t> nativeNufft_m;

#ifdef ENABLE_FINUFFT
        // FINUFFT backend
        FinufftPlan_t finufftPlan_m{};

        // Temporary buffers for FINUFFT
        using FieldViewType    = Kokkos::View<FinufftComplex_t***, Kokkos::LayoutLeft, MemSpace>;
        using ParticleRealView = Kokkos::View<T*, Kokkos::LayoutLeft, MemSpace>;
        using ParticleCplxView = Kokkos::View<FinufftComplex_t*, Kokkos::LayoutLeft, MemSpace>;

        FieldViewType tempField_m;
        std::array<ParticleRealView, 3> tempR_m;
        ParticleCplxView tempQ_m;
#endif

    public:
        /**
         * @brief Construct NUFFT transform
         *
         * @param layout Field layout
         * @param localNp Local number of particles
         * @param type Transform type (1 or 2)
         * @param params Configuration parameters
         */
        FFT(const Layout_t& layout, detail::size_type localNp, int type,
            const ParameterList& params);

        ~FFT();

        // Non-copyable, non-movable (due to FINUFFT plan)
        FFT(const FFT&)            = delete;
        FFT& operator=(const FFT&) = delete;
        FFT(FFT&&)                 = delete;
        FFT& operator=(FFT&&)      = delete;

        /*!
         * @brief Execute NUFFT transform.
         *
         * Type 1: Spreads particle data @p Q at positions @p R onto field @p f.
         * Type 2: Interpolates field @p f to positions @p R, storing results in @p Q.
         *
         * @param R Particle positions.
         * @param Q Particle scalar values (input on Type 1, output on Type 2).
         * @param f Complex field on a uniform grid (output on Type 1, input on Type 2).
         */
        template <class... Properties>
        void transform(const ParticleAttrib<Vector<T, Dim>, Properties...>& R,
                       ParticleAttrib<T, Properties...>& Q, ComplexField& f);

        //! Native (Kokkos / heFFTe) NUFFT path.
        //! @note Public because NVCC's extended lambda support disallows
        //!       lambdas inside private templated member functions.
        template <class... Properties>
        void transformNative(const ParticleAttrib<Vector<T, Dim>, Properties...>& R,
                             ParticleAttrib<T, Properties...>& Q, ComplexField& f);

        //! FINUFFT / cuFINUFFT NUFFT path. Same NVCC public-visibility caveat.
        template <class... Properties>
        void transformFinufft(const ParticleAttrib<Vector<T, Dim>, Properties...>& R,
                              ParticleAttrib<T, Properties...>& Q, ComplexField& f);
    private:
        //! Pick a backend (native vs. finufft) based on @p params and lock_method.
        void initBackend(const Layout_t& layout, const ParameterList& params);
        //! Build the native NUFFT engine.
        void initNative(const Layout_t& layout, const ParameterList& params);
        //! Tear down whichever backend is currently allocated.
        void cleanupBackend();

        //! Build the FINUFFT plan and configure tolerances/options.
        void initFinufft(const ParameterList& params);
        //! Allocate the LayoutLeft scratch views FINUFFT consumes.
        void allocateFinufftBuffers(const Layout_t& layout, detail::size_type localNp);
    };

}  // namespace ippl

#include "FFT/Transform/NUFFT.hpp"

#endif  // IPPL_FFT_TRANSFORM_NUFFT_H
