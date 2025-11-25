//
// Native NUFFT Implementation
//   NUFFT using scatterES/gatherES and heFFTe FFT.
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
#include "Particle/ParticleAttrib.h"

// TODO(paul) this is for testing, remove at some point
#include <KokkosFFT.hpp>

#include "correction.h"
#include "es_kernel.h"
#include "nufft_types.h"

namespace ippl {
    namespace NUFFT {

        /**
         * @brief Native NUFFT implementation using IPPL components.
         *
         * Type 1: Spread from nonuniform points to uniform Fourier modes
         *         (scatterES -> FFT -> deconvolution)
         *
         * Type 2: Interpolate uniform Fourier modes at nonuniform points
         *         (correction -> FFT -> gatherES)
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

            // Field types (using uniform Cartesian mesh)
            using Mesh_t       = UniformCartesian<T, Dim>;
            using Centering_t  = Cell;
            using ComplexField = Field<complex_type, Dim, Mesh_t, Centering_t, ExecSpace>;
            using Layout_t     = FieldLayout<Dim>;

            struct Config {
                T tol     = T(1e-6);  // Error tolerance
                T sigma   = T(2.0);   // Upsampling factor
                bool sort = true;     // Sort particles for cache efficiency
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

                // Create decomposition (single rank for now)
                std::array<bool, Dim> isParallel;
                isParallel.fill(false);

                grid_layout_ = std::make_unique<Layout_t>(comm, domain, isParallel);

                // Create mesh for upsampled grid
                // Use origin at -π to match typical NUFFT convention of [-π, π]^Dim
                Vector<T, Dim> origin, hx;
                for (unsigned d = 0; d < Dim; ++d) {
                    origin[d] = -M_PI;
                    T extent = T(2.0) * M_PI;
                    hx[d] = extent / n_grid_[d];  // Mesh spacing, not extent!
                }

                grid_mesh_ = std::make_unique<Mesh_t>(domain, hx, origin);

                // Create upsampled grid field
                grid_field_ = std::make_unique<ComplexField>(*grid_mesh_, *grid_layout_);

                // Initialize heFFTe FFT
                ParameterList fftParams;
                fftParams.add("use_heffte_defaults", true);
                heffte_fft_ =
                    std::make_unique<FFT<CCTransform, ComplexField>>(*grid_layout_, fftParams);

                // Precompute deconvolution factors
                // auto t0 = std::chrono::high_resolution_clock::now();
                // for (unsigned d = 0; d < Dim; ++d) {
                //     factors_[d] = complex_view_1d("deconv_factors", n_modes_[d]);
                //     computeDeconvolutionFactors<ExecSpace, T>(factors_[d], n_modes_[d],
                //     n_grid_[d],
                //                                               kernel_);
                // }
                // Kokkos::fence();
                // timing_.precompute =
                //     std::chrono::duration<T>(std::chrono::high_resolution_clock::now() - t0)
                //         .count();

                // Precompute deconvolution factors using kokkos-nufft’s ES_Kernel
                auto t0 = std::chrono::high_resolution_clock::now();

                // Build a kokkos-nufft kernel with same tolerance as your IPPL kernel/config
                nufft::ES_Kernel<ExecSpace, T> nufft_kernel(cfg_.tol);

                for (unsigned d = 0; d < Dim; ++d) {
                    factors_[d] = complex_view_1d("deconv_factors", n_modes_[d]);

                    nufft::compute_deconvolution_factors<ExecSpace, T>(
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

                std::cout << "[NativeNUFFT] Starting type1 transform" << std::endl;
                std::cout << "[NativeNUFFT] n_grid = (" << n_grid_[0] << ", " << n_grid_[1] << ", " << n_grid_[2] << ")" << std::endl;
                std::cout << "[NativeNUFFT] grid origin = (" << grid_mesh_->getOrigin()[0] << ", " << grid_mesh_->getOrigin()[1] << ", " << grid_mesh_->getOrigin()[2] << ")" << std::endl;
                std::cout << "[NativeNUFFT] grid spacing = (" << grid_mesh_->getMeshSpacing()[0] << ", " << grid_mesh_->getMeshSpacing()[1] << ", " << grid_mesh_->getMeshSpacing()[2] << ")" << std::endl;
                std::cout << "[NativeNUFFT] kernel width = " << kernel_.width() << ", beta = " << kernel_.beta() << std::endl;
                std::cout << "[NativeNUFFT] kernel(0) = " << kernel_(T(0)) << std::endl;
                std::cout << "[NativeNUFFT] kernel(0.5) = " << kernel_(T(0.5)) << std::endl;

                // DEBUG: Compute kernel normalization
                {
                    int w = kernel_.width();
                    T inv_hw = T(2.0) / w;
                    T sum_1d = 0;
                    for (int i = 0; i < w; ++i) {
                        T x = (T(0.5) - T(i)) * inv_hw;  // Particle at center of cell
                        sum_1d += kernel_(x);
                    }
                    T sum_3d = sum_1d * sum_1d * sum_1d;
                    std::cout << "[NativeNUFFT] Kernel 1D sum = " << sum_1d << ", 3D sum = " << sum_3d << std::endl;
                }

                // DEBUG: Print first 3 particles and sum of charges
                {
                    auto R_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), R.getView());
                    auto Q_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Q.getView());
                    size_t npart = R.getParticleCount();
                    std::cout << "[NativeNUFFT] Type 1 - First 3 particles:" << std::endl;
                    for (size_t i = 0; i < std::min(size_t(3), npart); ++i) {
                        std::cout << "  x[" << i << "] = (" << R_host(i)[0] << ", " << R_host(i)[1] << ", " << R_host(i)[2]
                                  << "), c[" << i << "] = (" << Q_host(i) << ", 0)" << std::endl;
                    }

                    // Compute sum of charges
                    T sum_q = 0;
                    for (size_t i = 0; i < npart; ++i) {
                        sum_q += Q_host(i);
                    }
                    std::cout << "[NativeNUFFT] Sum of input charges = " << sum_q << std::endl;
                }

                // Use scatterES for spreading
                Q.scatterES(*grid_field_, R, kernel_, cfg_.sort);
                Kokkos::fence();

                std::cout << "[NativeNUFFT] Spreading complete" << std::endl;

                // DEBUG: Check a single value after spreading and compute sum
                {
                    auto grid_host = grid_field_->getHostMirror();
                    Kokkos::deep_copy(grid_host, grid_field_->getView());
                    int ng = grid_field_->getNghost();
                    int mid = n_grid_[0]/2 + ng;
                    std::cout << "[NativeNUFFT] grid[mid,mid,mid] after spread = "
                             << grid_host(mid, mid, mid) << std::endl;

                    // Compute sum of grid
                    complex_type sum = 0;
                    for (size_t i = ng; i < n_grid_[0] + ng; ++i) {
                        for (size_t j = ng; j < n_grid_[1] + ng; ++j) {
                            for (size_t k = ng; k < n_grid_[2] + ng; ++k) {
                                sum += grid_host(i, j, k);
                            }
                        }
                    }
                    std::cout << "[NativeNUFFT] grid sum after spread = " << sum << std::endl;
                }

                timing_.spread =
                    std::chrono::duration<T>(std::chrono::high_resolution_clock::now() - t0)
                        .count();

                // Step 2: Inverse FFT
                std::cout << "[NativeNUFFT] Performing inverse FFT..." << std::endl;
                performFFT(-1);
                std::cout << "[NativeNUFFT] FFT complete (norm_factor = " << (n_grid_[0] * n_grid_[1] * n_grid_[2]) << ")" << std::endl;

                // DEBUG: Check a single value after FFT and look for imaginary parts
                {
                    auto grid_host = grid_field_->getHostMirror();
                    Kokkos::deep_copy(grid_host, grid_field_->getView());
                    int ng = grid_field_->getNghost();
                    int mid = n_grid_[0]/2 + ng;
                    std::cout << "[NativeNUFFT] grid[mid,mid,mid] after FFT = "
                             << grid_host(mid, mid, mid) << std::endl;
                    // Check a few other points for imaginary parts
                    std::cout << "[NativeNUFFT] grid[mid+1,mid,mid] after FFT = "
                             << grid_host(mid+1, mid, mid) << std::endl;
                    std::cout << "[NativeNUFFT] grid[mid,mid+1,mid] after FFT = "
                             << grid_host(mid, mid+1, mid) << std::endl;
                }

                t0 = std::chrono::high_resolution_clock::now();

                // Build the same shape arrays the original NUFFT uses
                using kokkos_size_type = typename memory_space::size_type;
                Kokkos::Array<kokkos_size_type, Dim> nmodes{};
                Kokkos::Array<kokkos_size_type, Dim> ngrid{};

                for (unsigned d = 0; d < Dim; ++d) {
                    nmodes[d] = static_cast<kokkos_size_type>(n_modes_[d]);
                    ngrid[d]  = static_cast<kokkos_size_type>(n_grid_[d]);
                }

                // Create temporary input/output views with LayoutRight for apply_correction
                auto grid_view_temp = nufft::make_view<complex_type, Dim, memory_space>("grid_temp", ngrid);
                auto output_view_temp = nufft::make_view<complex_type, Dim, memory_space>("output_temp", nmodes);

                // Copy grid data to temporary view WITHOUT normalization
                // heFFTe uses its own normalization convention
                T norm_factor = T(1);  // No normalization
                copyGridToTempNormalized(grid_view_temp, ngrid, norm_factor);

                // Convert factors std::array to Kokkos::Array
                Kokkos::Array<complex_view_1d, Dim> factors_kokkos;
                for (unsigned d = 0; d < Dim; ++d) {
                    factors_kokkos[d] = factors_[d];
                }

                nufft::apply_correction<ExecSpace, T, Dim>(
                    grid_view_temp,     // upsampled grid (LayoutRight)
                    factors_kokkos,     // deconvolution factors
                    output_view_temp,   // output modes field view (LayoutRight)
                    nmodes,             // number of modes (output extents)
                    ngrid,              // upsampled grid extents
                    nmodes,             // "target" modes (same as nmodes here)
                    timing_.correct     // timing (will be written by apply_correction)
                );

                // Copy result back to output field
                copyOutputToField(output_view_temp, f, nmodes);

                // DEBUG: Check output field
                {
                    auto f_host = f.getHostMirror();
                    Kokkos::deep_copy(f_host, f.getView());
                    int ng = f.getNghost();
                    int mid = n_modes_[0]/2 + ng;
                    std::cout << "[NativeNUFFT] output[0,0,0] after copy = "
                             << f_host(ng, ng, ng) << std::endl;
                    std::cout << "[NativeNUFFT] output[mid,mid,mid] after copy = "
                             << f_host(mid, mid, mid) << std::endl;
                    // Also check test index (k = (0.37*16, 0.26*16, 0.13*16) = (5,4,2))
                    int test_i = n_modes_[0]/2 + 5 + ng;
                    int test_j = n_modes_[1]/2 + 4 + ng;
                    int test_k = n_modes_[2]/2 + 2 + ng;
                    std::cout << "[NativeNUFFT] output[" << test_i << "," << test_j << "," << test_k << "] = "
                             << f_host(test_i, test_j, test_k) << std::endl;
                }

                timing_.correct =
                    std::chrono::duration<T>(std::chrono::high_resolution_clock::now() - t0)
                        .count();

                // // Step 3: Deconvolution and truncation to output modes
                // t0 = std::chrono::high_resolution_clock::now();
                // applyDeconvolutionType1<Dim, ExecSpace, T>(grid_field_->getView(), factors_,
                //                                            f.getView(), n_modes_, n_grid_);
                // timing_.correct =
                //     std::chrono::duration<T>(std::chrono::high_resolution_clock::now() - t0)
                //         .count();

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

                // Step 1: Pre-correction and zero-padding
                auto t0 = std::chrono::high_resolution_clock::now();
                applyPreCorrectionType2<Dim, ExecSpace, T>(
                    f.getView(), factors_, grid_field_->getView(), n_modes_, n_grid_);
                timing_.correct =
                    std::chrono::duration<T>(std::chrono::high_resolution_clock::now() - t0)
                        .count();

                // Step 2: FFT (forward direction for type 2)
                t0 = std::chrono::high_resolution_clock::now();
                heffte_fft_->transform(FORWARD, *grid_field_);
                timing_.fft =
                    std::chrono::duration<T>(std::chrono::high_resolution_clock::now() - t0)
                        .count();

                // Step 3: Gather/interpolate at particle positions
                t0 = std::chrono::high_resolution_clock::now();
                Q.gatherES(*grid_field_, R, kernel_, false, cfg_.sort);
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

            // Helper to copy grid data from IPPL field to temporary view
            template <typename TempView, typename SizeType>
            void copyGridToTemp(TempView& dst, const Kokkos::Array<SizeType, Dim>& ngrid) {
                auto src_view = grid_field_->getView();
                const int nghost = grid_field_->getNghost();

                using policy_type = Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<Dim>>;

                if constexpr (Dim == 3) {
                    policy_type policy({0, 0, 0}, {static_cast<int>(ngrid[0]),
                                                    static_cast<int>(ngrid[1]),
                                                    static_cast<int>(ngrid[2])});
                    Kokkos::parallel_for("copyGridToTemp", policy,
                        KOKKOS_LAMBDA(int i, int j, int k) {
                            dst(i, j, k) = src_view(i + nghost, j + nghost, k + nghost);
                        });
                } else if constexpr (Dim == 2) {
                    policy_type policy({0, 0}, {static_cast<int>(ngrid[0]),
                                                 static_cast<int>(ngrid[1])});
                    Kokkos::parallel_for("copyGridToTemp", policy,
                        KOKKOS_LAMBDA(int i, int j) {
                            dst(i, j) = src_view(i + nghost, j + nghost);
                        });
                } else {
                    Kokkos::parallel_for("copyGridToTemp",
                        Kokkos::RangePolicy<execution_space>(0, ngrid[0]),
                        KOKKOS_LAMBDA(int i) {
                            dst(i) = src_view(i + nghost);
                        });
                }
                Kokkos::fence();
            }

            // Helper to copy grid data from IPPL field to temporary view with normalization
            template <typename TempView, typename SizeType>
            void copyGridToTempNormalized(TempView& dst, const Kokkos::Array<SizeType, Dim>& ngrid, T norm_factor) {
                auto src_view = grid_field_->getView();
                const int nghost = grid_field_->getNghost();

                using policy_type = Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<Dim>>;

                if constexpr (Dim == 3) {
                    policy_type policy({0, 0, 0}, {static_cast<int>(ngrid[0]),
                                                    static_cast<int>(ngrid[1]),
                                                    static_cast<int>(ngrid[2])});
                    Kokkos::parallel_for("copyGridToTemp", policy,
                        KOKKOS_LAMBDA(int i, int j, int k) {
                            dst(i, j, k) = src_view(i + nghost, j + nghost, k + nghost) / norm_factor;
                        });
                } else if constexpr (Dim == 2) {
                    policy_type policy({0, 0}, {static_cast<int>(ngrid[0]),
                                                 static_cast<int>(ngrid[1])});
                    Kokkos::parallel_for("copyGridToTemp", policy,
                        KOKKOS_LAMBDA(int i, int j) {
                            dst(i, j) = src_view(i + nghost, j + nghost) / norm_factor;
                        });
                } else {
                    Kokkos::parallel_for("copyGridToTemp",
                        Kokkos::RangePolicy<execution_space>(0, ngrid[0]),
                        KOKKOS_LAMBDA(int i) {
                            dst(i) = src_view(i + nghost) / norm_factor;
                        });
                }
                Kokkos::fence();
            }

            // Helper to copy from nufft output view to IPPL field
            template <typename OutputView, typename OutField, typename SizeType>
            void copyOutputToField(const OutputView& src, OutField& dst,
                                   const Kokkos::Array<SizeType, Dim>& nmodes) {
                auto dst_view = dst.getView();
                const int nghost = dst.getNghost();

                using policy_type = Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<Dim>>;

                if constexpr (Dim == 3) {
                    const int nx = static_cast<int>(nmodes[0]);
                    const int ny = static_cast<int>(nmodes[1]);
                    const int nz = static_cast<int>(nmodes[2]);

                    policy_type policy({0, 0, 0}, {nx, ny, nz});
                    Kokkos::parallel_for("copyOutputToField", policy,
                        KOKKOS_LAMBDA(int i, int j, int k) {
                            // apply_correction outputs in corner-DC format
                            // Apply FFT-shift to convert to IPPL's centered format
                            // and conjugate to match kokkos_nufft convention
                            const int ii_shift = (i + nx/2) % nx;
                            const int jj_shift = (j + ny/2) % ny;
                            const int kk_shift = (k + nz/2) % nz;
                            dst_view(i + nghost, j + nghost, k + nghost) =
                                Kokkos::conj(src(ii_shift, jj_shift, kk_shift));
                        });
                } else if constexpr (Dim == 2) {
                    const int nx = static_cast<int>(nmodes[0]);
                    const int ny = static_cast<int>(nmodes[1]);

                    policy_type policy({0, 0}, {nx, ny});
                    Kokkos::parallel_for("copyOutputToField", policy,
                        KOKKOS_LAMBDA(int i, int j) {
                            const int ii_shift = (i + nx/2) % nx;
                            const int jj_shift = (j + ny/2) % ny;
                            dst_view(i + nghost, j + nghost) =
                                Kokkos::conj(src(ii_shift, jj_shift));
                        });
                } else {
                    const int nx = static_cast<int>(nmodes[0]);
                    Kokkos::parallel_for("copyOutputToField",
                        Kokkos::RangePolicy<execution_space>(0, nx),
                        KOKKOS_LAMBDA(int i) {
                            const int ii_shift = (i + nx/2) % nx;
                            dst_view(i + nghost) = Kokkos::conj(src(ii_shift));
                        });
                }
                Kokkos::fence();
            }

            void performFFT(int sign) {
                using exec_space = execution_space;

                // Ghost-free logical grid (LayoutStride)
                auto grid_ng = this->gridViewNoGhosts();

                const auto t0 = std::chrono::high_resolution_clock::now();

                if constexpr (Dim == 1) {
                    const auto n0 = static_cast<int>(n_grid_[0]);

                    // 1D scratch view with LayoutRight
                    Kokkos::View<complex_type*, Kokkos::LayoutRight, memory_space> fft_view(
                        "fft_view", n0);

                    // Copy from subview to scratch (manual copy for layout compatibility)
                    Kokkos::parallel_for(
                        "copy_to_fft_view_1d",
                        Kokkos::RangePolicy<exec_space>(0, n0),
                        KOKKOS_LAMBDA(int i) { fft_view(i) = grid_ng(i); });

                    Kokkos::fence();

                    if (sign < 0) {
                        KokkosFFT::ifft(exec_space{}, fft_view, fft_view,
                                        KokkosFFT::Normalization::none);
                    } else {
                        KokkosFFT::fft(exec_space{}, fft_view, fft_view,
                                       KokkosFFT::Normalization::none);
                    }

                    // Copy back into subview
                    Kokkos::parallel_for(
                        "copy_from_fft_view_1d",
                        Kokkos::RangePolicy<exec_space>(0, n0),
                        KOKKOS_LAMBDA(int i) { grid_ng(i) = fft_view(i); });

                    Kokkos::fence();

                } else if constexpr (Dim == 2) {
                    const auto n0 = static_cast<int>(n_grid_[0]);
                    const auto n1 = static_cast<int>(n_grid_[1]);

                    Kokkos::View<complex_type**, Kokkos::LayoutRight, memory_space> fft_view(
                        "fft_view", n0, n1);

                    // Copy from subview to scratch
                    Kokkos::parallel_for(
                        "copy_to_fft_view_2d",
                        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({0, 0}, {n0, n1}),
                        KOKKOS_LAMBDA(int i, int j) { fft_view(i, j) = grid_ng(i, j); });

                    Kokkos::fence();

                    // Axes 0,1 (row-major on LayoutRight)
                    std::array<int, 2> axes = {0, 1};

                    if (sign < 0) {
                        KokkosFFT::ifftn(exec_space{}, fft_view, fft_view, axes,
                                         KokkosFFT::Normalization::none);
                    } else {
                        KokkosFFT::fftn(exec_space{}, fft_view, fft_view, axes,
                                        KokkosFFT::Normalization::none);
                    }

                    // Copy back into subview
                    Kokkos::parallel_for(
                        "copy_from_fft_view_2d",
                        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({0, 0}, {n0, n1}),
                        KOKKOS_LAMBDA(int i, int j) { grid_ng(i, j) = fft_view(i, j); });

                    Kokkos::fence();

                } else if constexpr (Dim == 3) {
                    const auto n0 = static_cast<int>(n_grid_[0]);
                    const auto n1 = static_cast<int>(n_grid_[1]);
                    const auto n2 = static_cast<int>(n_grid_[2]);

                    Kokkos::View<complex_type***, Kokkos::LayoutRight, memory_space> fft_view(
                        "fft_view", n0, n1, n2);

                    // Copy from subview to scratch
                    Kokkos::parallel_for(
                        "copy_to_fft_view_3d",
                        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>({0, 0, 0}, {n0, n1, n2}),
                        KOKKOS_LAMBDA(int i, int j, int k) {
                            fft_view(i, j, k) = grid_ng(i, j, k);
                        });

                    Kokkos::fence();

                    std::array<int, 3> axes = {0, 1, 2};

                    if (sign < 0) {
                        KokkosFFT::ifftn(exec_space{}, fft_view, fft_view, axes,
                                         KokkosFFT::Normalization::none);
                    } else {
                        KokkosFFT::fftn(exec_space{}, fft_view, fft_view, axes,
                                        KokkosFFT::Normalization::none);
                    }

                    // Copy back into subview
                    Kokkos::parallel_for(
                        "copy_from_fft_view_3d",
                        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>({0, 0, 0}, {n0, n1, n2}),
                        KOKKOS_LAMBDA(int i, int j, int k) {
                            grid_ng(i, j, k) = fft_view(i, j, k);
                        });

                    Kokkos::fence();
                }

                timing_.fft =
                    std::chrono::duration<T>(std::chrono::high_resolution_clock::now() - t0)
                        .count();
            }

            // TODO(paul) this is for testing
            auto gridViewNoGhosts() {
                using view_type = typename ComplexField::view_type;
                view_type full  = grid_field_->getView();

                const int nghost = grid_field_->getNghost();

                if constexpr (Dim == 1) {
                    return Kokkos::subview(
                        full, Kokkos::make_pair(static_cast<int>(nghost),
                                                static_cast<int>(nghost + n_grid_[0])));
                } else if constexpr (Dim == 2) {
                    return Kokkos::subview(
                        full,
                        Kokkos::make_pair(static_cast<int>(nghost),
                                          static_cast<int>(nghost + n_grid_[0])),
                        Kokkos::make_pair(static_cast<int>(nghost),
                                          static_cast<int>(nghost + n_grid_[1])));
                } else {  // Dim == 3
                    return Kokkos::subview(
                        full,
                        Kokkos::make_pair(static_cast<int>(nghost),
                                          static_cast<int>(nghost + n_grid_[0])),
                        Kokkos::make_pair(static_cast<int>(nghost),
                                          static_cast<int>(nghost + n_grid_[1])),
                        Kokkos::make_pair(static_cast<int>(nghost),
                                          static_cast<int>(nghost + n_grid_[2])));
                }
            }
        };

    }  // namespace NUFFT
}  // namespace ippl

#endif  // IPPL_NATIVE_NUFFT_H
