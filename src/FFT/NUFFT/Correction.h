//
// NUFFT Correction
//   Deconvolution factor computation and correction operations for NUFFT.
//
#ifndef IPPL_NUFFT_CORRECTION_H
#define IPPL_NUFFT_CORRECTION_H

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <array>

#include "FFT/NUFFT/ESKernel.h"
#include "FFT/NUFFT/Quadrature.h"
#include "Types/ViewTypes.h"

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
        int64_t n_modes,
        int64_t n_grid,
        const ESKernel<T>& kernel) {
        using complex_type = Kokkos::complex<T>;
        using memory_space = typename ExecSpace::memory_space;

        constexpr int q = 100;
        Kokkos::View<T*, memory_space> nodes("quad_nodes", q);
        Kokkos::View<T*, memory_space> weights("quad_weights", q);

        gaussLegendre<memory_space, T>(q, nodes, weights);

        const T alpha = M_PI * kernel.width() / n_grid;
        const T beta = kernel.beta();
        const int w = kernel.width();

        Kokkos::parallel_for("compute_deconv_factors",
            Kokkos::RangePolicy<ExecSpace>(0, n_modes),
            KOKKOS_LAMBDA(const int64_t k) {
                int freq = (k < n_modes / 2) ? k : k - n_modes;
                T ft = 0.0;

                for (int i = 0; i < q; ++i) {
                    const T x = nodes(i);
                    const T wt = weights(i);
                    const T ker = Kokkos::exp(beta * (Kokkos::sqrt(T(1) - x * x) - T(1)));
                    ft += wt * ker * Kokkos::cos(freq * alpha * x);
                }

                factors(k) = complex_type(T(2) / (w * ft), T(0));
            });

        Kokkos::fence();
    }

    /**
     * @brief Apply deconvolution correction for Type 1 NUFFT (post-FFT).
     *
     * Maps upsampled FFT output to Fourier modes with deconvolution.
     * Type 1: grid -> modes
     *
     * @tparam Dim Number of dimensions
     * @tparam ExecSpace Kokkos execution space
     * @tparam T Floating point type
     */
    template <unsigned Dim, typename ExecSpace, typename T>
    void applyDeconvolutionType1(
        typename detail::ViewType<Kokkos::complex<T>, Dim>::view_type input,
        const std::array<Kokkos::View<Kokkos::complex<T>*, typename ExecSpace::memory_space>, Dim>& factors,
        typename detail::ViewType<Kokkos::complex<T>, Dim>::view_type output,
        const Vector<size_t, Dim>& n_modes,
        const Vector<size_t, Dim>& n_grid) {
        using complex_type = Kokkos::complex<T>;

        if constexpr (Dim == 3) {
            Kokkos::parallel_for("deconv_type1_3d",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                    {0, 0, 0},
                    {static_cast<int64_t>(n_modes[0]),
                     static_cast<int64_t>(n_modes[1]),
                     static_cast<int64_t>(n_modes[2])}),
                KOKKOS_LAMBDA(int64_t k0, int64_t k1, int64_t k2) {
                    int64_t i0 = (k0 < static_cast<int64_t>(n_modes[0]) / 2) ? k0 : n_grid[0] - (n_modes[0] - k0);
                    int64_t i1 = (k1 < static_cast<int64_t>(n_modes[1]) / 2) ? k1 : n_grid[1] - (n_modes[1] - k1);
                    int64_t i2 = (k2 < static_cast<int64_t>(n_modes[2]) / 2) ? k2 : n_grid[2] - (n_modes[2] - k2);

                    complex_type factor = factors[0](k0) * factors[1](k1) * factors[2](k2);
                    output(k0, k1, k2) = input(i0, i1, i2) * factor;
                });
        } else if constexpr (Dim == 2) {
            Kokkos::parallel_for("deconv_type1_2d",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                    {0, 0},
                    {static_cast<int64_t>(n_modes[0]),
                     static_cast<int64_t>(n_modes[1])}),
                KOKKOS_LAMBDA(int64_t k0, int64_t k1) {
                    int64_t i0 = (k0 < static_cast<int64_t>(n_modes[0]) / 2) ? k0 : n_grid[0] - (n_modes[0] - k0);
                    int64_t i1 = (k1 < static_cast<int64_t>(n_modes[1]) / 2) ? k1 : n_grid[1] - (n_modes[1] - k1);

                    complex_type factor = factors[0](k0) * factors[1](k1);
                    output(k0, k1) = input(i0, i1) * factor;
                });
        } else {
            Kokkos::parallel_for("deconv_type1_1d",
                Kokkos::RangePolicy<ExecSpace>(0, n_modes[0]),
                KOKKOS_LAMBDA(int64_t k0) {
                    int64_t i0 = (k0 < static_cast<int64_t>(n_modes[0]) / 2) ? k0 : n_grid[0] - (n_modes[0] - k0);
                    output(k0) = input(i0) * factors[0](k0);
                });
        }

        Kokkos::fence();
    }

    /**
     * @brief Apply pre-correction for Type 2 NUFFT (pre-FFT).
     *
     * Multiplies input modes by correction factors and zero-pads to upsampled grid.
     * Type 2: modes -> grid (with zero-padding)
     */
    template <unsigned Dim, typename ExecSpace, typename T>
    void applyPreCorrectionType2(
        typename detail::ViewType<Kokkos::complex<T>, Dim>::view_type input,
        const std::array<Kokkos::View<Kokkos::complex<T>*, typename ExecSpace::memory_space>, Dim>& factors,
        typename detail::ViewType<Kokkos::complex<T>, Dim>::view_type output,
        const Vector<size_t, Dim>& n_modes,
        const Vector<size_t, Dim>& n_grid) {
        using complex_type = Kokkos::complex<T>;

        Kokkos::deep_copy(output, complex_type(0, 0));

        if constexpr (Dim == 3) {
            const int nx = static_cast<int>(n_modes[0]);
            const int ny = static_cast<int>(n_modes[1]);
            const int nz = static_cast<int>(n_modes[2]);

            // Capture factors by value
            auto f0 = factors[0];
            auto f1 = factors[1];
            auto f2 = factors[2];

            Kokkos::parallel_for("precorr_type2_3d",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                    {0, 0, 0}, {nx, ny, nz}),
                KOKKOS_LAMBDA(int64_t k0, int64_t k1, int64_t k2) {
                    // Map both input and output using same formula (no shift, no conjugation)
                    int64_t in_idx0 = (k0 < nx / 2) ? k0 : n_modes[0] - (nx - k0);
                    int64_t in_idx1 = (k1 < ny / 2) ? k1 : n_modes[1] - (ny - k1);
                    int64_t in_idx2 = (k2 < nz / 2) ? k2 : n_modes[2] - (nz - k2);

                    int64_t out_idx0 = (k0 < nx / 2) ? k0 : n_grid[0] - (nx - k0);
                    int64_t out_idx1 = (k1 < ny / 2) ? k1 : n_grid[1] - (ny - k1);
                    int64_t out_idx2 = (k2 < nz / 2) ? k2 : n_grid[2] - (nz - k2);

                    complex_type factor = f0(k0) * f1(k1) * f2(k2);
                    output(out_idx0, out_idx1, out_idx2) = input(in_idx0, in_idx1, in_idx2) * factor;
                });

            Kokkos::fence();
        } else if constexpr (Dim == 2) {
            const int nx = static_cast<int>(n_modes[0]);
            const int ny = static_cast<int>(n_modes[1]);

            Kokkos::parallel_for("precorr_type2_2d",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                    {0, 0}, {nx, ny}),
                KOKKOS_LAMBDA(int64_t k0, int64_t k1) {
                    int64_t i0 = (k0 < nx / 2) ? k0 : n_grid[0] - (nx - k0);
                    int64_t i1 = (k1 < ny / 2) ? k1 : n_grid[1] - (ny - k1);

                    int64_t j0 = (k0 + nx/2) % nx;
                    int64_t j1 = (k1 + ny/2) % ny;

                    complex_type factor = factors[0](k0) * factors[1](k1);
                    output(i0, i1) = Kokkos::conj(input(j0, j1)) * factor;
                });
        } else {
            const int nx = static_cast<int>(n_modes[0]);

            Kokkos::parallel_for("precorr_type2_1d",
                Kokkos::RangePolicy<ExecSpace>(0, nx),
                KOKKOS_LAMBDA(int64_t k0) {
                    int64_t i0 = (k0 < nx / 2) ? k0 : n_grid[0] - (nx - k0);
                    int64_t j0 = (k0 + nx/2) % nx;
                    output(i0) = Kokkos::conj(input(j0)) * factors[0](k0);
                });
        }

        Kokkos::fence();
    }

}  // namespace NUFFT
}  // namespace ippl

#endif  // IPPL_NUFFT_CORRECTION_H
