#ifndef IPPL_NUFFT_CORRECTION_H
#define IPPL_NUFFT_CORRECTION_H

#include <Kokkos_Core.hpp>

#include <Kokkos_Complex.hpp>
#include <array>

#include "Types/ViewTypes.h"

#include "Utility/ParallelDispatch.h"

#include "FFT/NUFFT/ESKernel.h"
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
        // The shared `factor` stored in the `factors` views is
        //   factor = deconv * exp(-i pi k / N)   [negative phase]
        //
        // Type 1 (post-FFT):  f_k = conj(G_hat_k * factor)
        // Type 2 (pre-IFFT):  G_hat_k = f_k * factor
        //   The +i pi phase needed by the cell-centered gather emerges from the
        //   IFFT of the conjugate-symmetric mode field, so no explicit conj() is
        //   applied here.
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
        template <typename FieldIn, typename FieldOut, typename ExecSpace, typename T,
                  unsigned Dim>
        void applyDeconvolutionType1(
            FieldIn& input,
            const std::array<Kokkos::View<Kokkos::complex<T>*, typename ExecSpace::memory_space>,
                             Dim>& factors,
            FieldOut& output, const Vector<size_t, Dim>& n_modes,
            const Vector<size_t, Dim>& /*n_grid*/) {
            using complex_type     = Kokkos::complex<T>;
            using factor_view_t    = Kokkos::View<Kokkos::complex<T>*,
                                                  typename ExecSpace::memory_space>;
            using index_array_type = typename ippl::RangePolicy<Dim, ExecSpace>::index_array_type;

            auto input_view  = input.getView();
            auto output_view = output.getView();

            const auto& layout = input.getLayout();
            const auto& lDom   = layout.getLocalNDIndex();

            const int nghost_in  = input.getNghost();
            const int nghost_out = output.getNghost();

            // Capture-by-value device-friendly arrays.
            Vector<int, Dim> local_first;
            Kokkos::Array<int64_t, Dim> begin, end;
            Kokkos::Array<factor_view_t, Dim> f_arr;
            Kokkos::Array<int, Dim> n_modes_arr;
            for (unsigned d = 0; d < Dim; ++d) {
                local_first[d] = lDom[d].first();
                begin[d]       = lDom[d].first();
                end[d]         = lDom[d].last() + 1;
                f_arr[d]       = factors[d];
                n_modes_arr[d] = static_cast<int>(n_modes[d]);
            }

            ippl::parallel_for(
                "deconv_type1_local",
                ippl::createRangePolicy<Dim, ExecSpace>(begin, end),
                KOKKOS_LAMBDA(const index_array_type& g) {
                    bool inside = true;
                    for (unsigned d = 0; d < Dim; ++d) {
                        const int gi = static_cast<int>(g[d]);
                        const int n  = n_modes_arr[d];
                        if (!((gi >= 0 && gi < n / 2) || (gi >= n + n / 2 && gi < 2 * n))) {
                            inside = false;
                            break;
                        }
                    }

                    index_array_type idx_in, idx_out;
                    for (unsigned d = 0; d < Dim; ++d) {
                        idx_in[d]  = static_cast<int>(g[d]) - local_first[d] + nghost_in;
                        idx_out[d] = static_cast<int>(g[d]) - local_first[d] + nghost_out;
                    }

                    if (inside) {
                        complex_type factor(T(1), T(0));
                        for (unsigned d = 0; d < Dim; ++d) {
                            const int gi = static_cast<int>(g[d]);
                            const int n  = n_modes_arr[d];
                            const int rescaled = (gi < n) ? gi : gi - n;
                            factor *= f_arr[d](rescaled);
                        }
                        ippl::apply(output_view, idx_out) =
                            Kokkos::conj(ippl::apply(input_view, idx_in) * factor);
                    } else {
                        ippl::apply(output_view, idx_out) = complex_type(0, 0);
                    }
                });

            Kokkos::fence();
        }

        /**
         * @brief Apply pre-correction for Type 2 NUFFT (pre-IFFT).
         */
        template <typename FieldIn, typename FieldOut, typename ExecSpace, typename T,
                  unsigned Dim>
        void applyPreCorrectionType2(
            FieldIn& input,
            const std::array<Kokkos::View<Kokkos::complex<T>*, typename ExecSpace::memory_space>,
                             Dim>& factors,
            FieldOut& output, const Vector<size_t, Dim>& n_modes,
            const Vector<size_t, Dim>& /*n_grid*/) {
            using complex_type     = Kokkos::complex<T>;
            using factor_view_t    = Kokkos::View<Kokkos::complex<T>*,
                                                  typename ExecSpace::memory_space>;
            using index_array_type = typename ippl::RangePolicy<Dim, ExecSpace>::index_array_type;

            auto input_view  = input.getView();
            auto output_view = output.getView();

            const auto& layout = input.getLayout();
            const auto& lDom   = layout.getLocalNDIndex();

            const int nghost_in  = input.getNghost();
            const int nghost_out = output.getNghost();

            Vector<int, Dim> local_first;
            Kokkos::Array<int64_t, Dim> begin, end;
            Kokkos::Array<factor_view_t, Dim> f_arr;
            Kokkos::Array<int, Dim> n_modes_arr;
            for (unsigned d = 0; d < Dim; ++d) {
                local_first[d] = lDom[d].first();
                begin[d]       = lDom[d].first();
                end[d]         = lDom[d].last() + 1;
                f_arr[d]       = factors[d];
                n_modes_arr[d] = static_cast<int>(n_modes[d]);
            }

            ippl::parallel_for(
                "precorr_type2_local",
                ippl::createRangePolicy<Dim, ExecSpace>(begin, end),
                KOKKOS_LAMBDA(const index_array_type& g) {
                    bool inside = true;
                    for (unsigned d = 0; d < Dim; ++d) {
                        const int gi = static_cast<int>(g[d]);
                        const int n  = n_modes_arr[d];
                        if (!((gi >= 0 && gi < n / 2) || (gi >= n + n / 2 && gi < 2 * n))) {
                            inside = false;
                            break;
                        }
                    }

                    index_array_type idx_in, idx_out;
                    for (unsigned d = 0; d < Dim; ++d) {
                        idx_in[d]  = static_cast<int>(g[d]) - local_first[d] + nghost_in;
                        idx_out[d] = static_cast<int>(g[d]) - local_first[d] + nghost_out;
                    }

                    if (inside) {
                        complex_type factor(T(1), T(0));
                        for (unsigned d = 0; d < Dim; ++d) {
                            const int gi = static_cast<int>(g[d]);
                            const int n  = n_modes_arr[d];
                            const int rescaled = (gi < n) ? gi : gi - n;
                            factor *= f_arr[d](rescaled);
                        }
                        ippl::apply(output_view, idx_out) =
                            ippl::apply(input_view, idx_in) * factor;
                    } else {
                        ippl::apply(output_view, idx_out) = complex_type(0, 0);
                    }
                });

            Kokkos::fence();
        }

        /**
         * @brief Apply deconvolution for Type 1 NUFFT on the pruned mode grid.
         */
        template <typename FieldType, typename ExecSpace, typename T, unsigned Dim>
        void applyDeconvolutionPruned(
            FieldType& field,
            const std::array<Kokkos::View<Kokkos::complex<T>*, typename ExecSpace::memory_space>,
                             Dim>& factors,
            const Vector<size_t, Dim>& n_modes, const Vector<size_t, Dim>& /*n_grid*/) {
            using complex_type     = Kokkos::complex<T>;
            using factor_view_t    = Kokkos::View<Kokkos::complex<T>*,
                                                  typename ExecSpace::memory_space>;
            using index_array_type = typename ippl::RangePolicy<Dim, ExecSpace>::index_array_type;

            auto view    = field.getView();
            auto& layout = field.getLayout();
            auto lDom    = layout.getLocalNDIndex();

            const int nghost = field.getNghost();

            Vector<int, Dim> local_first;
            Kokkos::Array<int64_t, Dim> begin, end;
            Kokkos::Array<factor_view_t, Dim> f_arr;
            Kokkos::Array<int, Dim> n_modes_arr;
            for (unsigned d = 0; d < Dim; ++d) {
                local_first[d] = lDom[d].first();
                begin[d]       = lDom[d].first();
                end[d]         = lDom[d].last() + 1;
                f_arr[d]       = factors[d];
                n_modes_arr[d] = static_cast<int>(n_modes[d]);
            }

            ippl::parallel_for(
                "deconv_type1_pruned_local",
                ippl::createRangePolicy<Dim, ExecSpace>(begin, end),
                KOKKOS_LAMBDA(const index_array_type& g) {
                    for (unsigned d = 0; d < Dim; ++d) {
                        const int gi = static_cast<int>(g[d]);
                        if (gi < 0 || gi >= n_modes_arr[d]) return;
                    }

                    index_array_type idx;
                    for (unsigned d = 0; d < Dim; ++d) {
                        idx[d] = static_cast<int>(g[d]) - local_first[d] + nghost;
                    }

                    complex_type factor(T(1), T(0));
                    for (unsigned d = 0; d < Dim; ++d) {
                        factor *= f_arr[d](static_cast<int>(g[d]));
                    }

                    ippl::apply(view, idx) = Kokkos::conj(ippl::apply(view, idx) * factor);
                });

            Kokkos::fence();
        }

        /**
         * @brief Apply pre-correction for Type 2 NUFFT on the pruned mode grid.
         */
        template <typename FieldType, typename ExecSpace, typename T, unsigned Dim>
        void applyPrecorrectionPruned(
            FieldType& field,
            const std::array<Kokkos::View<Kokkos::complex<T>*, typename ExecSpace::memory_space>,
                             Dim>& factors,
            const Vector<size_t, Dim>& n_modes, const Vector<size_t, Dim>& /*n_grid*/) {
            using complex_type     = Kokkos::complex<T>;
            using factor_view_t    = Kokkos::View<Kokkos::complex<T>*,
                                                  typename ExecSpace::memory_space>;
            using index_array_type = typename ippl::RangePolicy<Dim, ExecSpace>::index_array_type;

            auto view    = field.getView();
            auto& layout = field.getLayout();
            auto lDom    = layout.getLocalNDIndex();

            const int nghost = field.getNghost();

            Vector<int, Dim> local_first;
            Kokkos::Array<int64_t, Dim> begin, end;
            Kokkos::Array<factor_view_t, Dim> f_arr;
            Kokkos::Array<int, Dim> n_modes_arr;
            for (unsigned d = 0; d < Dim; ++d) {
                local_first[d] = lDom[d].first();
                begin[d]       = lDom[d].first();
                end[d]         = lDom[d].last() + 1;
                f_arr[d]       = factors[d];
                n_modes_arr[d] = static_cast<int>(n_modes[d]);
            }

            ippl::parallel_for(
                "precorr_type2_pruned_local",
                ippl::createRangePolicy<Dim, ExecSpace>(begin, end),
                KOKKOS_LAMBDA(const index_array_type& g) {
                    for (unsigned d = 0; d < Dim; ++d) {
                        const int gi = static_cast<int>(g[d]);
                        if (gi < 0 || gi >= n_modes_arr[d]) return;
                    }

                    index_array_type idx;
                    for (unsigned d = 0; d < Dim; ++d) {
                        idx[d] = static_cast<int>(g[d]) - local_first[d] + nghost;
                    }

                    complex_type factor(T(1), T(0));
                    for (unsigned d = 0; d < Dim; ++d) {
                        factor *= f_arr[d](static_cast<int>(g[d]));
                    }

                    ippl::apply(view, idx) = ippl::apply(view, idx) * factor;
                });

            Kokkos::fence();
        }

    }  // namespace nufft
}  // namespace ippl

#endif  // IPPL_NUFFT_CORRECTION_H
