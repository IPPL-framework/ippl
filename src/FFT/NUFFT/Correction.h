#ifndef IPPL_NUFFT_CORRECTION_H
#define IPPL_NUFFT_CORRECTION_H

#include <Kokkos_Core.hpp>

#include <Kokkos_Complex.hpp>
#include <array>

#include "Types/ViewTypes.h"

#include "FFT/NUFFT/ESKernel.h"
#include "FFT/NUFFT/Quadrature.h"
#include "FFT/NUFFT/NUFFTUtilities.h"

namespace ippl {
    namespace nufft {

        // ====================================================================
        // Cell-centered phase convention
        // ====================================================================
        //
        // Scatter/gather use cell-centered DOFs at x_j = (j + 1/2) * h.
        // The DFT of the scattered field is:
        //
        //   G_hat_k = phi_hat_k * exp(+i pi k / N) * conj(f_k)
        //
        // where the exp(+i pi k / N) arises because the kernel is evaluated
        // at (x_p / h - j - 1/2) instead of (x_p / h - j).
        //
        // Type 1: f_k = conj(G_hat_k * factor)
        //   requires  factor = deconv * exp(-i pi k / N)   [negative phase]
        //
        // Type 2: G_hat_k = N * f_k * conj(factor)
        //   requires  conj(factor) = deconv * exp(+i pi k / N)  [positive phase]
        //
        // The factors array always stores the Type-1 factor (negative phase).
        // Type-2 functions apply conj(factor) explicitly.
        // ====================================================================

        /**
         * @brief Computes deconvolution factors for NUFFT.
         */
        template <class ExecSpace, typename RealType = double>
        void compute_deconvolution_factors(
            Kokkos::View<Kokkos::complex<RealType>*, typename ExecSpace::memory_space> factors,
            int64_t n_modes, int64_t n_grid, const ESKernel<RealType>& kernel) {
            using complex_type = Kokkos::complex<RealType>;

            // Set up quadrature
            constexpr int q = 100;
            auto nodes      = Kokkos::View<RealType*, typename ExecSpace::memory_space>("nodes", q);
            auto weights = Kokkos::View<RealType*, typename ExecSpace::memory_space>("weights", q);

            gauss_legendre<ExecSpace>(q, nodes, weights);

            const RealType alpha = Kokkos::numbers::pi_v<RealType> * kernel.width() / n_grid;
            const RealType beta  = kernel.beta();

            const int64_t l_n_modes = n_modes;
            constexpr int l_q       = q;

            Kokkos::parallel_for(
                "compute_deconv_factors", Kokkos::RangePolicy<ExecSpace>(0, n_modes),
                KOKKOS_LAMBDA(const int64_t k) {
                    int freq    = (k < l_n_modes / 2) ? k : k - l_n_modes;
                    RealType ft = 0.0;

                    for (int i = 0; i < l_q; ++i) {
                        const RealType x   = nodes(i);
                        const RealType w   = weights(i);
                        const RealType ker = Kokkos::exp(beta * (Kokkos::sqrt(1.0 - x * x) - 1.0));
                        ft += w * ker * Kokkos::cos(freq * alpha * x);
                    }

                    const RealType deconv = 2.0 / (kernel.width() * ft);

                    // Phase correction for cell-centered DOFs: exp(-i pi freq / N_grid)
                    // Type-1 uses this factor directly; Type-2 applies conj(factor).
                    const RealType phase = Kokkos::numbers::pi_v<RealType>
                                           * static_cast<RealType>(freq)
                                           / static_cast<RealType>(n_grid);
                    factors(k) =
                        complex_type(deconv * Kokkos::cos(phase), deconv * Kokkos::sin(phase));
                });
        }

        /**
         * @brief Apply deconvolution correction for Type 1 NUFFT (post-FFT).
         *
         * Type 1: nonuniform points -> uniform Fourier modes.
         * After spreading (cell-centered) and FFT, the result satisfies:
         *
         *   G_hat_k = phi_hat_k * exp(+i pi k / N) * conj(f_k)
         *
         * This function recovers f_k via:
         *
         *   output_k = conj(input_k * factor_k)
         *
         * where factor_k = deconv_k * exp(-i pi k / N)  (stored in `factors`).
         *
         * Operates locally; input lives on the upsampled grid, output on the
         * mode grid. Entries outside the mode band are set to zero.
         */
        template <typename FieldIn, typename FieldOut, typename ExecSpace, typename T>
        void applyDeconvolutionType1(
            FieldIn& input,
            const std::array<Kokkos::View<Kokkos::complex<T>*, typename ExecSpace::memory_space>,
                             3>& factors,
            FieldOut& output, const Vector<size_t, 3>& n_modes, const Vector<size_t, 3>&) {
            using complex_type = Kokkos::complex<T>;

            constexpr unsigned Dim = 3;

            auto input_view  = input.getView();
            auto output_view = output.getView();

            const auto& layout = input.getLayout();
            const auto& lDom   = layout.getLocalNDIndex();

            const int nghost_in  = input.getNghost();
            const int nghost_out = output.getNghost();

            Vector<int, Dim> local_first, local_last;
            for (unsigned d = 0; d < Dim; ++d) {
                local_first[d] = lDom[d].first();
                local_last[d]  = lDom[d].last();
            }

            auto f0 = factors[0];
            auto f1 = factors[1];
            auto f2 = factors[2];

            const int nx = static_cast<int>(n_modes[0]);
            const int ny = static_cast<int>(n_modes[1]);
            const int nz = static_cast<int>(n_modes[2]);

            Kokkos::parallel_for(
                "deconv_type1_3d_local",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                    {local_first[0], local_first[1], local_first[2]},
                    {local_last[0] + 1, local_last[1] + 1, local_last[2] + 1}),
                KOKKOS_LAMBDA(int gi, int gj, int gk) {
                    // Corner-DC layout: modes in [0, N/2) ∪ [N+N/2, 2N)
                    auto in_bounds = [&](int g, int n) {
                        return (g >= 0 && g < n / 2) || (g >= n + n / 2 && g < 2 * n);
                    };

                    const int li_in  = gi - local_first[0] + nghost_in;
                    const int lj_in  = gj - local_first[1] + nghost_in;
                    const int lk_in  = gk - local_first[2] + nghost_in;
                    const int li_out = gi - local_first[0] + nghost_out;
                    const int lj_out = gj - local_first[1] + nghost_out;
                    const int lk_out = gk - local_first[2] + nghost_out;

                    if (in_bounds(gi, nx) && in_bounds(gj, ny) && in_bounds(gk, nz)) {
                        // Map upsampled-grid corner-DC index back to factor index [0, n_modes)
                        auto rescale = [](int g, int n) {
                            return (g < n) ? g : g - n;
                        };

                        // factor = deconv * exp(-i pi freq / N_grid)
                        const complex_type factor =
                            f0(rescale(gi, nx)) * f1(rescale(gj, ny)) * f2(rescale(gk, nz));

                        // f_k = conj(G_hat_k * factor)
                        output_view(li_out, lj_out, lk_out) =
                            Kokkos::conj(input_view(li_in, lj_in, lk_in) * factor);
                    } else {
                        output_view(li_out, lj_out, lk_out) = complex_type(0, 0);
                    }
                });

            Kokkos::fence();
        }

        /**
         * @brief Apply pre-correction for Type 2 NUFFT (pre-IFFT).
         *
         * Type 2: uniform Fourier modes -> nonuniform points.
         * Before the IFFT and cell-centered gather, the mode field must be
         * pre-multiplied so that the gather recovers the correct values:
         *
         *   G_hat_k = N * f_k * conj(factor_k)
         *
         * where factor_k = deconv_k * exp(-i pi k / N)  (stored in `factors`),
         * so conj(factor_k) = deconv_k * exp(+i pi k / N).
         *
         * Entries outside the mode band are set to zero.
         */
        template <typename FieldIn, typename FieldOut, typename ExecSpace, typename T>
        void applyPreCorrectionType2(
            FieldIn& input,
            const std::array<Kokkos::View<Kokkos::complex<T>*, typename ExecSpace::memory_space>,
                             3>& factors,
            FieldOut& output, const Vector<size_t, 3>& n_modes, const Vector<size_t, 3>&) {
            using complex_type = Kokkos::complex<T>;

            constexpr unsigned Dim = 3;

            auto input_view  = input.getView();
            auto output_view = output.getView();

            const auto& layout = input.getLayout();
            const auto& lDom   = layout.getLocalNDIndex();

            const int nghost_in  = input.getNghost();
            const int nghost_out = output.getNghost();

            Vector<int, Dim> local_first, local_last;
            for (unsigned d = 0; d < Dim; ++d) {
                local_first[d] = lDom[d].first();
                local_last[d]  = lDom[d].last();
            }

            Kokkos::deep_copy(output_view, complex_type(0, 0));

            auto f0 = factors[0];
            auto f1 = factors[1];
            auto f2 = factors[2];

            const int nx = static_cast<int>(n_modes[0]);
            const int ny = static_cast<int>(n_modes[1]);
            const int nz = static_cast<int>(n_modes[2]);

            Kokkos::parallel_for(
                "precorr_type2_3d_local",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                    {local_first[0], local_first[1], local_first[2]},
                    {local_last[0] + 1, local_last[1] + 1, local_last[2] + 1}),
                KOKKOS_LAMBDA(int gi, int gj, int gk) {
                    auto in_bounds = [&](int g, int n) {
                        return (g >= 0 && g < n / 2) || (g >= n + n / 2 && g < 2 * n);
                    };

                    const int li_in  = gi - local_first[0] + nghost_in;
                    const int lj_in  = gj - local_first[1] + nghost_in;
                    const int lk_in  = gk - local_first[2] + nghost_in;
                    const int li_out = gi - local_first[0] + nghost_out;
                    const int lj_out = gj - local_first[1] + nghost_out;
                    const int lk_out = gk - local_first[2] + nghost_out;

                    if (in_bounds(gi, nx) && in_bounds(gj, ny) && in_bounds(gk, nz)) {
                        auto rescale = [](int g, int n) {
                            return (g < n) ? g : g - n;
                        };

                        // factor = deconv * exp(-i pi freq / N_grid)
                        const complex_type factor =
                            f0(rescale(gi, nx)) * f1(rescale(gj, ny)) * f2(rescale(gk, nz));

                        // G_hat_k = f_k * conj(factor)  [conj gives +i pi phase]
                        output_view(li_out, lj_out, lk_out) =
                            input_view(li_in, lj_in, lk_in) * Kokkos::conj(factor);
                    } else {
                        output_view(li_out, lj_out, lk_out) = complex_type(0, 0);
                    }
                });

            Kokkos::fence();
        }

        /**
         * @brief Apply deconvolution for Type 1 NUFFT on the pruned mode grid.
         *
         * Same as applyDeconvolutionType1 but operates directly on a field
         * already living in mode space (0..n_modes[d]-1 per dim, corner-DC).
         *
         *   field(gi,gj,gk) <- conj( field(gi,gj,gk) * f0(gi)*f1(gj)*f2(gk) )
         */
        template <typename FieldType, typename ExecSpace, typename T>
        void applyDeconvolutionPruned(
            FieldType& field,
            const std::array<Kokkos::View<Kokkos::complex<T>*, typename ExecSpace::memory_space>,
                             3>& factors,
            const Vector<size_t, 3>& n_modes, const Vector<size_t, 3>& /*n_grid*/) {
            using complex_type     = Kokkos::complex<T>;
            constexpr unsigned Dim = 3;

            auto view    = field.getView();
            auto& layout = field.getLayout();
            auto lDom    = layout.getLocalNDIndex();

            const int nghost = field.getNghost();

            Vector<int, Dim> local_first, local_last;
            for (unsigned d = 0; d < Dim; ++d) {
                local_first[d] = lDom[d].first();
                local_last[d]  = lDom[d].last();
            }

            auto f0 = factors[0];
            auto f1 = factors[1];
            auto f2 = factors[2];

            const int nx = static_cast<int>(n_modes[0]);
            const int ny = static_cast<int>(n_modes[1]);
            const int nz = static_cast<int>(n_modes[2]);

            Kokkos::parallel_for(
                "deconv_type1_pruned_local",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                    {local_first[0], local_first[1], local_first[2]},
                    {local_last[0] + 1, local_last[1] + 1, local_last[2] + 1}),
                KOKKOS_LAMBDA(int gi, int gj, int gk) {
                    if (gi < 0 || gj < 0 || gk < 0 || gi >= nx || gj >= ny || gk >= nz)
                        return;

                    const int li = gi - local_first[0] + nghost;
                    const int lj = gj - local_first[1] + nghost;
                    const int lk = gk - local_first[2] + nghost;

                    // factor = deconv * exp(-i pi freq / N_grid)
                    const complex_type factor = f0(gi) * f1(gj) * f2(gk);

                    // f_k = conj(G_hat_k * factor)
                    view(li, lj, lk) = Kokkos::conj(view(li, lj, lk) * factor);
                });

            Kokkos::fence();
        }

        /**
         * @brief Apply pre-correction for Type 2 NUFFT on the pruned mode grid.
         *
         * Same as applyPreCorrectionType2 but operates directly on a field
         * already in mode space (0..n_modes[d]-1 per dim, corner-DC).
         *
         *   field(gi,gj,gk) <- field(gi,gj,gk) * conj( f0(gi)*f1(gj)*f2(gk) )
         *
         * The conj gives the +i pi phase needed for the IFFT + cell-centered gather
         * to recover the correct values.
         */
        template <typename FieldType, typename ExecSpace, typename T>
        void applyPrecorrectionPruned(
            FieldType& field,
            const std::array<Kokkos::View<Kokkos::complex<T>*, typename ExecSpace::memory_space>,
                             3>& factors,
            const Vector<size_t, 3>& n_modes, const Vector<size_t, 3>& /*n_grid*/) {
            using complex_type     = Kokkos::complex<T>;
            constexpr unsigned Dim = 3;

            auto view    = field.getView();
            auto& layout = field.getLayout();
            auto lDom    = layout.getLocalNDIndex();

            const int nghost = field.getNghost();

            Vector<int, Dim> local_first, local_last;
            for (unsigned d = 0; d < Dim; ++d) {
                local_first[d] = lDom[d].first();
                local_last[d]  = lDom[d].last();
            }

            auto f0 = factors[0];
            auto f1 = factors[1];
            auto f2 = factors[2];

            const int nx = static_cast<int>(n_modes[0]);
            const int ny = static_cast<int>(n_modes[1]);
            const int nz = static_cast<int>(n_modes[2]);

            Kokkos::parallel_for(
                "precorr_type2_pruned_local",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                    {local_first[0], local_first[1], local_first[2]},
                    {local_last[0] + 1, local_last[1] + 1, local_last[2] + 1}),
                KOKKOS_LAMBDA(int gi, int gj, int gk) {
                    if (gi < 0 || gj < 0 || gk < 0 || gi >= nx || gj >= ny || gk >= nz)
                        return;

                    const int li = gi - local_first[0] + nghost;
                    const int lj = gj - local_first[1] + nghost;
                    const int lk = gk - local_first[2] + nghost;

                    // factor = deconv * exp(-i pi freq / N_grid)
                    const complex_type factor = f0(gi) * f1(gj) * f2(gk);

                    // G_hat_k = f_k * conj(factor)  [conj gives +i pi phase]
                    view(li, lj, lk) *= factor;
                });

            Kokkos::fence();
        }

    }  // namespace nufft
}  // namespace ippl

#endif  // IPPL_NUFFT_CORRECTION_H