//
// Native NUFFT Implementation
//   NUFFT using kernel-based scatter/gather and heFFTe FFT.
//   Does not depend on external NUFFT libraries.
//
#ifndef IPPL_NATIVE_NUFFT_H
#define IPPL_NATIVE_NUFFT_H

#include <Kokkos_Core.hpp>

#include <Kokkos_Complex.hpp>
#include <array>
#include <bit>
#include <cmath>

#include "Types/Vector.h"

#include "Utility/IpplTimings.h"

#include "Field/Field.h"

#include "FFT/FFT.h"
#include "FFT/NUFFT/Correction.h"
#include "FFT/NUFFT/ESKernel.h"
#include "FFT/NUFFT/NUFFTUtilities.h"
#include "Particle/ParticleAttrib.h"
#include "Interpolation/ScatterConfig.h"

namespace ippl {
    namespace NUFFT {

        /**
         * @brief NUFFT implementation.
         *
         * Type 1: Spread from nonuniform points to uniform Fourier modes
         *         (scatter -> FFT -> deconvolution)
         *
         * Type 2: Interpolate uniform Fourier modes at nonuniform points
         *         (correction -> FFT -> gather)
         *
         * @tparam Dim Number of dimensions
         * @tparam T Floating point type
         * @tparam ExecSpace Kokkos execution space
         */
        template <unsigned Dim, typename T = double,
                  typename ExecSpace = Kokkos::DefaultExecutionSpace>
        class NativeNUFFT {
        public:
            using execution_space = ExecSpace;
            using memory_space    = typename ExecSpace::memory_space;
            using complex_type    = Kokkos::complex<T>;
            using size_type       = size_t;

            // View types
            using complex_view_1d = Kokkos::View<complex_type*, memory_space>;
            using real_view_1d    = Kokkos::View<T*, memory_space>;

            // Field types
            using Mesh_t       = UniformCartesian<T, Dim>;
            using Centering_t  = Cell;
            using ComplexField = Field<complex_type, Dim, Mesh_t, Centering_t, ExecSpace>;
            using Layout_t     = FieldLayout<Dim>;

            struct Config {
                T tol     = T(1e-6);  // Error tolerance
                T sigma   = T(2.0);   // Upsampling factor
                Interpolation::ScatterConfig spread;  // Spread/gather configuration (chooses impl/tiling/..)
            };

            struct TimingInfo {
                T spread     = 0;
                T fft        = 0;
                T correct    = 0;
                T total      = 0;
                T precompute = 0;
            };

        private:
            Config cfg_;
            ESKernel<T> kernel_;
            Vector<size_t, Dim> n_modes_;
            Vector<size_t, Dim> n_grid_;

            // Deconvolution factors per dimension
            std::array<complex_view_1d, Dim> factors_;

            // Upsampled grid for FFT
            std::unique_ptr<ComplexField> grid_field_;
            std::unique_ptr<Layout_t> grid_layout_;
            std::unique_ptr<Mesh_t> grid_mesh_;

            // heFFTe FFT object
            std::unique_ptr<FFT<CCTransform, ComplexField>> heffte_fft_;

            TimingInfo timing_;
            bool initialized_ = false;

        public:
            /**
             * @brief Construct NativeNUFFT with given mode counts.
             *
             * @param n_modes Number of Fourier modes per dimension
             * @param cfg Configuration parameters
             */
            NativeNUFFT(const Vector<size_t, Dim>& n_modes, Config cfg = {})
                : cfg_(cfg)
                , kernel_(cfg.tol)
                , n_modes_(n_modes) {
                // Compute upsampled grid sizes
                for (unsigned d = 0; d < Dim; ++d) {
                    n_grid_[d] = std::bit_ceil<size_t>(
                        std::max<size_t>(cfg_.sigma * n_modes_[d], 2 * kernel_.width()));
                }
                initialized_ = false;
            }

            /**
             * @brief Initialize the NUFFT with a layout.
             *
             * Must be called before transform operations.
             *
             * @param comm MPI communicator
             */
            void initialize(const MPI_Comm& comm = MPI_COMM_WORLD) {
                if (initialized_)
                    return;

                static IpplTimings::TimerRef initTimer = IpplTimings::getTimer("NativeNUFFT::init");
                IpplTimings::startTimer(initTimer);

                // Create index domain for upsampled grid
                NDIndex<Dim> domain;
                for (unsigned d = 0; d < Dim; ++d) {
                    domain[d] = Index(n_grid_[d]);
                }

                // Create decomposition
                std::array<bool, Dim> isParallel;
                isParallel.fill(false);

                // Enable periodic BCs for ghost cell exchange
                grid_layout_ = std::make_unique<Layout_t>(comm, domain, isParallel, true);

                // Create mesh for upsampled grid
                Vector<T, Dim> origin, hx;
                for (unsigned d = 0; d < Dim; ++d) {
                    origin[d] = -M_PI;
                    T extent = T(2.0) * M_PI;
                    hx[d] = extent / n_grid_[d];
                }

                grid_mesh_ = std::make_unique<Mesh_t>(domain, hx, origin);

                // Create upsampled grid field with sufficient ghost cells for kernel width
                // Native NUFFT uses field ghosts directly instead of creating extended grid
                const int hw = kernel_.width() / 2;
                const int nghost = hw;  // Need nghost >= hw for kernel width w
                grid_field_ = std::make_unique<ComplexField>(*grid_mesh_, *grid_layout_, nghost);

                // Initialize heFFTe FFT
                ParameterList fftParams;
                fftParams.add("use_heffte_defaults", true);
                heffte_fft_ =
                    std::make_unique<FFT<CCTransform, ComplexField>>(*grid_layout_, fftParams);

                // Precompute deconvolution factors
                auto t0 = std::chrono::high_resolution_clock::now();

                ESKernel<T> nufft_kernel(cfg_.tol);
                for (unsigned d = 0; d < Dim; ++d) {
                    factors_[d] = complex_view_1d("deconv_factors", n_modes_[d]);

                    ippl::nufft::compute_deconvolution_factors<ExecSpace, T>(
                        factors_[d],
                        static_cast<int64_t>(n_modes_[d]),
                        static_cast<int64_t>(n_grid_[d]),
                        nufft_kernel);
                }

                Kokkos::fence();
                timing_.precompute =
                    std::chrono::duration<T>(std::chrono::high_resolution_clock::now() - t0)
                        .count();

                initialized_ = true;
                IpplTimings::stopTimer(initTimer);
            }

            /**
             * @brief Type 1 NUFFT: Spread from nonuniform points to uniform Fourier modes.
             *
             * Computes f_k = sum_j c_j * exp(i * k * x_j) for uniform k.
             *
             * @tparam Properties ParticleAttrib properties
             * @tparam OutField Output field type
             * @param R Particle positions in [0, 2*pi)^Dim
             * @param Q Particle values (input)
             * @param f Output Fourier modes field
             */
            template <class... Properties, typename OutField>
            void type1(const ParticleAttrib<Vector<T, Dim>, Properties...>& R,
                       const ParticleAttrib<T, Properties...>& Q, OutField& f) {
                if (!initialized_) {
                    throw IpplException("NativeNUFFT::type1",
                                        "NUFFT not initialized. Call initialize() first.");
                }

                resetTimings();
                auto ttot = std::chrono::high_resolution_clock::now();

                // Step 1: Spread particles to upsampled grid
                auto t0      = std::chrono::high_resolution_clock::now();
                *grid_field_ = complex_type(0, 0);  // Zero the grid
                Q.scatter(*grid_field_, R, kernel_, cfg_.spread);
                Kokkos::fence();

                timing_.spread =
                    std::chrono::duration<T>(std::chrono::high_resolution_clock::now() - t0)
                        .count();

                // Step 1.5: Accumulate ghost cells from scatter
                grid_field_->accumulateHalo();

                // Step 2: Inverse FFT
                performFFT(-1);

                // Step 3: Deconvolution and truncation to output modes
                t0 = std::chrono::high_resolution_clock::now();
                applyDeconvolutionType1<Dim, ExecSpace, T>(
                    grid_field_->getView(), factors_,
                    f.getView(), n_modes_, n_grid_,
                    grid_field_->getNghost(), f.getNghost());
                timing_.correct =
                    std::chrono::duration<T>(std::chrono::high_resolution_clock::now() - t0)
                        .count();

                timing_.total =
                    std::chrono::duration<T>(std::chrono::high_resolution_clock::now() - ttot)
                        .count()
                    + timing_.precompute;
            }

            /**
             * @brief Type 2 NUFFT: Interpolate uniform Fourier modes at nonuniform points.
             *
             * Computes c_j = sum_k f_k * exp(i * k * x_j) at nonuniform x_j.
             *
             * @tparam InField Input field type
             * @tparam Properties ParticleAttrib properties
             * @param f Input Fourier modes field
             * @param R Particle positions in [0, 2*pi)^Dim
             * @param Q Output particle values
             */
            template <typename InField, class... Properties>
            void type2(const InField& f, const ParticleAttrib<Vector<T, Dim>, Properties...>& R,
                       ParticleAttrib<T, Properties...>& Q) {
                if (!initialized_) {
                    throw IpplException("NativeNUFFT::type2",
                                        "NUFFT not initialized. Call initialize() first.");
                }

                resetTimings();
                auto ttot = std::chrono::high_resolution_clock::now();

                // Step 1: Apply pre-correction
                auto t0 = std::chrono::high_resolution_clock::now();

                applyPreCorrectionType2<Dim, ExecSpace, T>(
                    f.getView(), factors_,
                    grid_field_->getView(),
                    n_modes_, n_grid_,
                    f.getNghost(), grid_field_->getNghost());

                timing_.correct =
                    std::chrono::duration<T>(std::chrono::high_resolution_clock::now() - t0)
                        .count();

                // Step 2: Inverse FFT
                t0 = std::chrono::high_resolution_clock::now();
                performFFT(-1);

                timing_.fft =
                    std::chrono::duration<T>(std::chrono::high_resolution_clock::now() - t0)
                        .count();

                // Step 2.5: Fill ghost cells for gather
                grid_field_->fillHalo();

                // Step 3: Gather/interpolate at particle positions
                t0 = std::chrono::high_resolution_clock::now();
                Q.gather(*grid_field_, R, kernel_, false, cfg_.spread);
                Kokkos::fence();
                timing_.spread =
                    std::chrono::duration<T>(std::chrono::high_resolution_clock::now() - t0)
                        .count();

                timing_.total =
                    std::chrono::duration<T>(std::chrono::high_resolution_clock::now() - ttot)
                        .count()
                    + timing_.precompute;
            }

            // Accessors
            const TimingInfo& timing() const { return timing_; }
            const ESKernel<T>& kernel() const { return kernel_; }
            Vector<size_t, Dim> gridSize() const { return n_grid_; }
            Vector<size_t, Dim> numModes() const { return n_modes_; }

            void resetTimings() {
                timing_.spread  = 0;
                timing_.fft     = 0;
                timing_.correct = 0;
            }

            void performFFT(int sign) {
                TransformDirection direction = (sign < 0) ? BACKWARD : FORWARD;
                heffte_fft_->transform(direction, *grid_field_);
            }
        };

    }  // namespace NUFFT
}  // namespace ippl

#endif  // IPPL_NATIVE_NUFFT_H
