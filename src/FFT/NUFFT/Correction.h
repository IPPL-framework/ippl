#ifndef IPPL_NUFFT_CORRECTION_H
#define IPPL_NUFFT_CORRECTION_H

#include <Kokkos_Core.hpp>

#include <Kokkos_Complex.hpp>
#include <array>

#include "Types/ViewTypes.h"

#include "FFT/NUFFT/ESKernel.h"
#include "FFT/NUFFT/Quadrature.h"

namespace ippl {
    namespace NUFFT {

        /**
         * @brief Computes deconvolution factors for NUFFT.
         *
         * For each frequency k, computes: factor[k] = 1 / phi_hat(k)
         * where phi_hat is the Fourier transform of the ES kernel.
         *
         * @tparam ExecSpace Kokkos execution space
         * @tparam T Floating point type
         * @param factors Output view for deconvolution factors [n_modes]
         * @param n_modes Number of output Fourier modes
         * @param n_grid Upsampled grid size
         * @param kernel ES kernel
         */
        template <typename ExecSpace, typename T = double>
        void computeDeconvolutionFactors(
            Kokkos::View<Kokkos::complex<T>*, typename ExecSpace::memory_space> factors,
            int64_t n_modes, int64_t n_grid, const ESKernel<T>& kernel) {
            using complex_type = Kokkos::complex<T>;
            using memory_space = typename ExecSpace::memory_space;

            constexpr int q = 100;
            Kokkos::View<T*, memory_space> nodes("quad_nodes", q);
            Kokkos::View<T*, memory_space> weights("quad_weights", q);

            gaussLegendre<memory_space, T>(q, nodes, weights);

            const T alpha = M_PI * kernel.width() / n_grid;
            const T beta  = kernel.beta();
            const int w   = kernel.width();

            Kokkos::parallel_for(
                "compute_deconv_factors", Kokkos::RangePolicy<ExecSpace>(0, n_modes),
                KOKKOS_LAMBDA(const int64_t k) {
                    int freq = (k < n_modes / 2) ? k : k - n_modes;
                    T ft     = 0.0;

                    for (int i = 0; i < q; ++i) {
                        const T x   = nodes(i);
                        const T wt  = weights(i);
                        const T ker = Kokkos::exp(beta * (Kokkos::sqrt(T(1) - x * x) - T(1)));
                        ft += wt * ker * Kokkos::cos(freq * alpha * x);
                    }

                    factors(k) = complex_type(T(2) / (w * ft), T(0));
                });

            Kokkos::fence();
        }

        /**
         * @brief Apply deconvolution correction for Type 1 NUFFT (post-FFT) on IPPL Fields.
         *
         * Works locally on distributed fields. Input and output have the same size.
         * Applies correction factors based on global frequency indices, copies input to output
         * with the correction applied.
         *
         * Type 1: grid -> modes (post-FFT correction with conjugation)
         *
         * @tparam Field IPPL Field type
         * @tparam ExecSpace Kokkos execution space
         * @tparam T Floating point type
         * @param input Input field (FFT output on upsampled grid)
         * @param factors Deconvolution factors for each dimension
         * @param output Output field (corrected modes)
         * @param n_modes Global number of modes in each dimension
         * @param n_grid Global upsampled grid size in each dimension
         */
        template <typename FieldIn, typename FieldOut, typename ExecSpace, typename T>
        void applyDeconvolutionType1(
            FieldIn& input,
            const std::array<Kokkos::View<Kokkos::complex<T>*, typename ExecSpace::memory_space>,
                             3>& factors,
            FieldOut& output, const Vector<size_t, 3>& n_modes, const Vector<size_t, 3>& n_grid) {
            using complex_type = Kokkos::complex<T>;

            constexpr unsigned Dim = 3;

            // Get field views
            auto input_view  = input.getView();
            auto output_view = output.getView();

            // Get layout information
            const auto& layout = input.getLayout();
            const auto& lDom   = layout.getLocalNDIndex();
            const int nghost   = input.getNghost();

            // Local domain bounds (global indices)
            Vector<int, Dim> local_first, local_last;
            for (unsigned d = 0; d < Dim; ++d) {
                local_first[d] = lDom[d].first();
                local_last[d]  = lDom[d].last();
            }

            // Capture factors by value
            auto f0 = factors[0];
            auto f1 = factors[1];
            auto f2 = factors[2];

            const int nx = static_cast<int>(n_modes[0]);
            const int ny = static_cast<int>(n_modes[1]);
            const int nz = static_cast<int>(n_modes[2]);

            // Iterate over local domain
            Kokkos::parallel_for(
                "deconv_type1_3d_local",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                    {local_first[0], local_first[1], local_first[2]},
                    {local_last[0] + 1, local_last[1] + 1, local_last[2] + 1}),
                KOKKOS_LAMBDA(int gi, int gj, int gk) {
                    auto in_bounds = [&](double g, double n_modes) {
                        return (g >= 0 && g < n_modes / 2)
                               || (g >= n_modes && g < n_modes + n_modes / 2);
                    };

                    int li = gi - local_first[0] + nghost;
                    int lj = gj - local_first[1] + nghost;
                    int lk = gk - local_first[2] + nghost;

                    if (in_bounds(gi, nx) && in_bounds(gj, ny) && in_bounds(gk, nz)) {
                        // Apply FFT-shift to get the shifted index for factor lookup
                        const int ii_shift = (gi + nx / 2) % nx;
                        const int jj_shift = (gj + ny / 2) % ny;
                        const int kk_shift = (gk + nz / 2) % nz;

                        // Compute factor using shifted indices
                        complex_type factor     = f0(ii_shift) * f1(jj_shift) * f2(kk_shift);
                        output_view(li, lj, lk) = Kokkos::conj(input_view(li, lj, lk) * factor);
                    } else {
                        output_view(li, lj, lk) = 0.0;
                    }
                });

            Kokkos::fence();
        }

        /**
         * @brief Apply pre-correction for Type 2 NUFFT (pre-FFT) on IPPL Fields.
         *
         * Works locally on distributed fields. Input and output have the same size.
         * Applies correction factors based on global frequency indices.
         *
         * Type 2: modes -> grid (pre-FFT correction)
         *
         * @tparam Field IPPL Field type
         * @tparam ExecSpace Kokkos execution space
         * @tparam T Floating point type
         * @param input Input field (Fourier modes)
         * @param factors Correction factors for each dimension
         * @param output Output field (corrected and ready for inverse FFT)
         * @param n_modes Global number of modes in each dimension
         * @param n_grid Global upsampled grid size in each dimension
         */
        template <typename FieldIn, typename FieldOut, typename ExecSpace, typename T>
        void applyPreCorrectionType2(
            FieldIn& input,
            const std::array<Kokkos::View<Kokkos::complex<T>*, typename ExecSpace::memory_space>,
                             3>& factors,
            FieldOut& output, const Vector<size_t, 3>& n_modes, const Vector<size_t, 3>& n_grid) {
            using complex_type = Kokkos::complex<T>;

            constexpr unsigned Dim = 3;

            // Get field views
            auto input_view  = input.getView();
            auto output_view = output.getView();

            // Get layout information
            const auto& layout = input.getLayout();
            const auto& lDom   = layout.getLocalNDIndex();
            const int nghost   = input.getNghost();

            // Local domain bounds (global indices)
            Vector<int, Dim> local_first, local_last;
            for (unsigned d = 0; d < Dim; ++d) {
                local_first[d] = lDom[d].first();
                local_last[d]  = lDom[d].last();
            }

            // Zero-initialize output (in case some regions are not covered)
            Kokkos::deep_copy(output_view, complex_type(0, 0));

            // Capture factors by value
            auto f0 = factors[0];
            auto f1 = factors[1];
            auto f2 = factors[2];

            const int nx = static_cast<int>(n_modes[0]);
            const int ny = static_cast<int>(n_modes[1]);
            const int nz = static_cast<int>(n_modes[2]);

            // Iterate over local domain
            Kokkos::parallel_for(
                "precorr_type2_3d_local",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                    {local_first[0], local_first[1], local_first[2]},
                    {local_last[0] + 1, local_last[1] + 1, local_last[2] + 1}),
                KOKKOS_LAMBDA(int gi, int gj, int gk) {
                    auto in_bounds = [&](double g, double n_modes) {
                        return (g >= 0 && g < n_modes / 2)
                               || (g >= n_modes && g < n_modes + n_modes / 2);
                    };
                    int li = gi - local_first[0] + nghost;
                    int lj = gj - local_first[1] + nghost;
                    int lk = gk - local_first[2] + nghost;

                    if (in_bounds(gi, nx) && in_bounds(gj, ny) && in_bounds(gk, nz)) {
                        const int ii_shift = (gi + nx / 2) % nx;
                        const int jj_shift = (gj + ny / 2) % ny;
                        const int kk_shift = (gk + nz / 2) % nz;

                        // Compute factor using shifted indices
                        complex_type factor = f0(ii_shift) * f1(jj_shift) * f2(kk_shift);

                        output_view(li, lj, lk) = input_view(li, lj, lk) * factor;
                    } else {
                        output_view(li, lj, lk) = 0.0;
                    }
                });

            Kokkos::fence();
        }

        /**
         * @brief Apply deconvolution correction for Type 1 NUFFT (post-FFT) - legacy view
         * interface.
         */
        template <unsigned Dim, typename ExecSpace, typename T>
        void applyDeconvolutionType1(
            typename detail::ViewType<Kokkos::complex<T>, Dim>::view_type input,
            const std::array<Kokkos::View<Kokkos::complex<T>*, typename ExecSpace::memory_space>,
                             Dim>& factors,
            typename detail::ViewType<Kokkos::complex<T>, Dim>::view_type output,
            const Vector<size_t, Dim>& n_modes, const Vector<size_t, Dim>& n_grid,
            int input_nghost = 0, int output_nghost = 0) {
            using complex_type = Kokkos::complex<T>;

            const int in_ghost  = input_nghost;
            const int out_ghost = output_nghost;

            if constexpr (Dim == 3) {
                auto f0 = factors[0];
                auto f1 = factors[1];
                auto f2 = factors[2];

                Kokkos::parallel_for(
                    "deconv_type1_3d",
                    Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                        {0, 0, 0},
                        {static_cast<int64_t>(n_modes[0]), static_cast<int64_t>(n_modes[1]),
                         static_cast<int64_t>(n_modes[2])}),
                    KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                        const int nx = static_cast<int>(n_modes[0]);
                        const int ny = static_cast<int>(n_modes[1]);
                        const int nz = static_cast<int>(n_modes[2]);

                        const int ii_shift = (i + nx / 2) % nx;
                        const int jj_shift = (j + ny / 2) % ny;
                        const int kk_shift = (k + nz / 2) % nz;

                        int64_t in_idx0 =
                            (ii_shift < nx / 2) ? ii_shift : n_grid[0] - (nx - ii_shift);
                        int64_t in_idx1 =
                            (jj_shift < ny / 2) ? jj_shift : n_grid[1] - (ny - jj_shift);
                        int64_t in_idx2 =
                            (kk_shift < nz / 2) ? kk_shift : n_grid[2] - (nz - kk_shift);

                        complex_type factor = f0(ii_shift) * f1(jj_shift) * f2(kk_shift);
                        output(i + out_ghost, j + out_ghost, k + out_ghost) = Kokkos::conj(
                            input(in_idx0 + in_ghost, in_idx1 + in_ghost, in_idx2 + in_ghost)
                            * factor);
                    });
            }

            Kokkos::fence();
        }

        /**
         * @brief Apply pre-correction for Type 2 NUFFT (pre-FFT) - legacy view interface.
         */
        template <unsigned Dim, typename ExecSpace, typename T>
        void applyPreCorrectionType2(
            typename detail::ViewType<Kokkos::complex<T>, Dim>::view_type input,
            const std::array<Kokkos::View<Kokkos::complex<T>*, typename ExecSpace::memory_space>,
                             Dim>& factors,
            typename detail::ViewType<Kokkos::complex<T>, Dim>::view_type output,
            const Vector<size_t, Dim>& n_modes, const Vector<size_t, Dim>& n_grid,
            int input_nghost = 0, int output_nghost = 0) {
            using complex_type = Kokkos::complex<T>;

            Kokkos::deep_copy(output, complex_type(0, 0));

            const int in_ghost  = input_nghost;
            const int out_ghost = output_nghost;

            if constexpr (Dim == 3) {
                const int nx = static_cast<int>(n_modes[0]);
                const int ny = static_cast<int>(n_modes[1]);
                const int nz = static_cast<int>(n_modes[2]);

                auto f0 = factors[0];
                auto f1 = factors[1];
                auto f2 = factors[2];

                Kokkos::parallel_for(
                    "precorr_type2_3d",
                    Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {nx, ny, nz}),
                    KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                        const int ii_shift = (i + nx / 2) % nx;
                        const int jj_shift = (j + ny / 2) % ny;
                        const int kk_shift = (k + nz / 2) % nz;

                        int64_t out_idx0 =
                            (ii_shift < nx / 2) ? ii_shift : n_grid[0] - (nx - ii_shift);
                        int64_t out_idx1 =
                            (jj_shift < ny / 2) ? jj_shift : n_grid[1] - (ny - jj_shift);
                        int64_t out_idx2 =
                            (kk_shift < nz / 2) ? kk_shift : n_grid[2] - (nz - kk_shift);

                        complex_type factor = f0(ii_shift) * f1(jj_shift) * f2(kk_shift);
                        output(out_idx0 + out_ghost, out_idx1 + out_ghost, out_idx2 + out_ghost) =
                            input(i + in_ghost, j + in_ghost, k + in_ghost) * factor;
                    });

                Kokkos::fence();
            } else if constexpr (Dim == 2) {
                const int nx = static_cast<int>(n_modes[0]);
                const int ny = static_cast<int>(n_modes[1]);

                auto f0 = factors[0];
                auto f1 = factors[1];

                Kokkos::parallel_for(
                    "precorr_type2_2d",
                    Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {nx, ny}),
                    KOKKOS_LAMBDA(int64_t k0, int64_t k1) {
                        int64_t i0 = (k0 < nx / 2) ? k0 : n_grid[0] - (nx - k0);
                        int64_t i1 = (k1 < ny / 2) ? k1 : n_grid[1] - (ny - k1);

                        int64_t j0 = (k0 + nx / 2) % nx;
                        int64_t j1 = (k1 + ny / 2) % ny;

                        complex_type factor = f0(k0) * f1(k1);
                        output(i0 + out_ghost, i1 + out_ghost) =
                            Kokkos::conj(input(j0 + in_ghost, j1 + in_ghost)) * factor;
                    });
            } else {
                const int nx = static_cast<int>(n_modes[0]);

                auto f0 = factors[0];

                Kokkos::parallel_for(
                    "precorr_type2_1d", Kokkos::RangePolicy<ExecSpace>(0, nx),
                    KOKKOS_LAMBDA(int64_t k0) {
                        int64_t i0             = (k0 < nx / 2) ? k0 : n_grid[0] - (nx - k0);
                        int64_t j0             = (k0 + nx / 2) % nx;
                        output(i0 + out_ghost) = Kokkos::conj(input(j0 + in_ghost)) * f0(k0);
                    });
            }

            Kokkos::fence();
        }

    }  // namespace NUFFT
}  // namespace ippl

#endif  // IPPL_NUFFT_CORRECTION_H