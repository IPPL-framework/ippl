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

            const int nghost_in  = input.getNghost();
            const int nghost_out = output.getNghost();

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
                    auto in_bounds = [&](double g, double n) {
                        return (g >= 0 && g < n / 2) || (g >= n + n / 2 && g < 2 * n);
                    };

                    int li_in = gi - local_first[0] + nghost_in;
                    int lj_in = gj - local_first[1] + nghost_in;
                    int lk_in = gk - local_first[2] + nghost_in;

                    int li_out = gi - local_first[0] + nghost_out;
                    int lj_out = gj - local_first[1] + nghost_out;
                    int lk_out = gk - local_first[2] + nghost_out;

                    if (in_bounds(gi, nx) && in_bounds(gj, ny) && in_bounds(gk, nz)) {
                        // Apply FFT-shift to get the shifted index for factor lookup

                        auto rescale = [&](int in, int n_modes) {
                            if (in < n_modes) {
                                return in;
                            } else {
                                return in - n_modes;
                            }
                        };

                        // Compute factor using shifted indices
                        complex_type factor = f0(rescale(gi, n_modes[0]))
                                              * f1(rescale(gj, n_modes[1]))
                                              * f2(rescale(gk, n_modes[2]));
                        output_view(li_out, lj_out, lk_out) =
                            Kokkos::conj(input_view(li_in, lj_in, lk_in) * factor);
                    } else {
                        output_view(li_out, lj_out, lk_out) = 0.0;
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

            const int nghost_in  = input.getNghost();
            const int nghost_out = output.getNghost();

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
                               || (g >= n_modes + n_modes / 2 && g < 2 * n_modes);
                    };
                    int li_in = gi - local_first[0] + nghost_in;
                    int lj_in = gj - local_first[1] + nghost_in;
                    int lk_in = gk - local_first[2] + nghost_in;

                    int li_out = gi - local_first[0] + nghost_out;
                    int lj_out = gj - local_first[1] + nghost_out;
                    int lk_out = gk - local_first[2] + nghost_out;

                    if (in_bounds(gi, nx) && in_bounds(gj, ny) && in_bounds(gk, nz)) {
                        auto rescale = [&](int in, int n_modes) {
                            if (in < n_modes) {
                                return in;
                            } else {
                                return in - n_modes;
                            }
                        };

                        // Compute factor using shifted indices
                        complex_type factor = f0(rescale(gi, n_modes[0]))
                                              * f1(rescale(gj, n_modes[1]))
                                              * f2(rescale(gk, n_modes[2]));
                        output_view(li_out, lj_out, lk_out) =
                            input_view(li_in, lj_in, lk_in) * factor;
                    } else {
                        output_view(li_out, lj_out, lk_out) = 0.0;
                    }
                });

            Kokkos::fence();
        }

        /**
         * @brief Apply deconvolution for Type 1 NUFFT on the pruned mode grid.
         *
         * This assumes field is defined on the pruned mode layout
         * (0..n_modes[d]-1 per dim, distributed over ranks) and that
         * factors store the same per-dimension deconvolution factors
         * used in the upsampled case.
         *
         * For each local (global) mode index (gi,gj,gk), we do:
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

            // Local domain bounds (global indices in mode space)
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
                    // Guard in case n_modes is smaller than global layout (shouldn't happen,
                    // but keeps us safe on edges)
                    if (gi < 0 || gj < 0 || gk < 0 || gi >= nx || gj >= ny || gk >= nz) {
                        return;
                    }

                    const int li = gi - local_first[0] + nghost;
                    const int lj = gj - local_first[1] + nghost;
                    const int lk = gk - local_first[2] + nghost;

                    complex_type factor = f0(gi) * f1(gj) * f2(gk);
                    view(li, lj, lk)    = Kokkos::conj(view(li, lj, lk) * factor);
                });

            Kokkos::fence();
        }

        /**
         * @brief Apply pre-correction for Type 2 NUFFT on the pruned mode grid.
         *
         * This assumes field is the pruned mode field (0..n_modes[d]-1 per dim
         * in corner-DC format), and multiplies each local mode by the same
         * per-dimension factors used in the upsampled case:
         *
         *   field(gi,gj,gk) <- field(gi,gj,gk) * f0(gi)*f1(gj)*f2(gk)
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
                    if (gi < 0 || gj < 0 || gk < 0 || gi >= nx || gj >= ny || gk >= nz) {
                        return;
                    }

                    const int li = gi - local_first[0] + nghost;
                    const int lj = gj - local_first[1] + nghost;
                    const int lk = gk - local_first[2] + nghost;

                    complex_type factor = f0(gi) * f1(gj) * f2(gk);
                    view(li, lj, lk) *= factor;
                });

            Kokkos::fence();
        }
    }  // namespace NUFFT
}  // namespace ippl

#endif  // IPPL_NUFFT_CORRECTION_H
