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
        const Vector<size_t, Dim>& n_grid,
        int input_nghost = 0,
        int output_nghost = 0) {
        using complex_type = Kokkos::complex<T>;

        // Capture ghost offsets by value for lambda
        const int in_ghost = input_nghost;
        const int out_ghost = output_nghost;

        if constexpr (Dim == 3) {
            // Capture factors by value
            auto f0 = factors[0];
            auto f1 = factors[1];
            auto f2 = factors[2];

            Kokkos::parallel_for("deconv_type1_3d",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                    {0, 0, 0},
                    {static_cast<int64_t>(n_modes[0]),
                     static_cast<int64_t>(n_modes[1]),
                     static_cast<int64_t>(n_modes[2])}),
                KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                    // apply_correction outputs in corner-DC format
                    // Apply FFT-shift to read from shifted indices and conjugate
                    const int nx = static_cast<int>(n_modes[0]);
                    const int ny = static_cast<int>(n_modes[1]);
                    const int nz = static_cast<int>(n_modes[2]);

                    const int ii_shift = (i + nx/2) % nx;
                    const int jj_shift = (j + ny/2) % ny;
                    const int kk_shift = (k + nz/2) % nz;

                    // Map shifted indices to FFT grid indices
                    int64_t in_idx0 = (ii_shift < nx / 2) ? ii_shift : n_grid[0] - (nx - ii_shift);
                    int64_t in_idx1 = (jj_shift < ny / 2) ? jj_shift : n_grid[1] - (ny - jj_shift);
                    int64_t in_idx2 = (kk_shift < nz / 2) ? kk_shift : n_grid[2] - (nz - kk_shift);

                    complex_type factor = f0(ii_shift) * f1(jj_shift) * f2(kk_shift);
                    // Output to centered format with conjugation
                    output(i + out_ghost, j + out_ghost, k + out_ghost) =
                        Kokkos::conj(input(in_idx0 + in_ghost, in_idx1 + in_ghost, in_idx2 + in_ghost) * factor);
                });
        } else if constexpr (Dim == 2) {
            const int nx = static_cast<int>(n_modes[0]);
            const int ny = static_cast<int>(n_modes[1]);

            auto f0 = factors[0];
            auto f1 = factors[1];

            Kokkos::parallel_for("deconv_type1_2d",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                    {0, 0}, {nx, ny}),
                KOKKOS_LAMBDA(int i, int j) {
                    const int ii_shift = (i + nx/2) % nx;
                    const int jj_shift = (j + ny/2) % ny;

                    int64_t in_idx0 = (ii_shift < nx / 2) ? ii_shift : n_grid[0] - (nx - ii_shift);
                    int64_t in_idx1 = (jj_shift < ny / 2) ? jj_shift : n_grid[1] - (ny - jj_shift);

                    complex_type factor = f0(ii_shift) * f1(jj_shift);
                    output(i + out_ghost, j + out_ghost) =
                        Kokkos::conj(input(in_idx0 + in_ghost, in_idx1 + in_ghost) * factor);
                });
        } else {
            const int nx = static_cast<int>(n_modes[0]);

            auto f0 = factors[0];

            Kokkos::parallel_for("deconv_type1_1d",
                Kokkos::RangePolicy<ExecSpace>(0, nx),
                KOKKOS_LAMBDA(int i) {
                    const int ii_shift = (i + nx/2) % nx;

                    int64_t in_idx0 = (ii_shift < nx / 2) ? ii_shift : n_grid[0] - (nx - ii_shift);

                    output(i + out_ghost) = Kokkos::conj(input(in_idx0 + in_ghost) * f0(ii_shift));
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
        const Vector<size_t, Dim>& n_grid,
        int input_nghost = 0,
        int output_nghost = 0) {
        using complex_type = Kokkos::complex<T>;

        Kokkos::deep_copy(output, complex_type(0, 0));

        // Capture ghost offsets by value for lambda
        const int in_ghost = input_nghost;
        const int out_ghost = output_nghost;

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
                KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                    // Apply FFT-shift to read from centered IPPL format
                    const int ii_shift = (i + nx/2) % nx;
                    const int jj_shift = (j + ny/2) % ny;
                    const int kk_shift = (k + nz/2) % nz;

                    // Map shifted indices for zero-padding
                    int64_t out_idx0 = (ii_shift < nx / 2) ? ii_shift : n_grid[0] - (nx - ii_shift);
                    int64_t out_idx1 = (jj_shift < ny / 2) ? jj_shift : n_grid[1] - (ny - jj_shift);
                    int64_t out_idx2 = (kk_shift < nz / 2) ? kk_shift : n_grid[2] - (nz - kk_shift);

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

            Kokkos::parallel_for("precorr_type2_2d",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                    {0, 0}, {nx, ny}),
                KOKKOS_LAMBDA(int64_t k0, int64_t k1) {
                    int64_t i0 = (k0 < nx / 2) ? k0 : n_grid[0] - (nx - k0);
                    int64_t i1 = (k1 < ny / 2) ? k1 : n_grid[1] - (ny - k1);

                    int64_t j0 = (k0 + nx/2) % nx;
                    int64_t j1 = (k1 + ny/2) % ny;

                    complex_type factor = f0(k0) * f1(k1);
                    output(i0 + out_ghost, i1 + out_ghost) =
                        Kokkos::conj(input(j0 + in_ghost, j1 + in_ghost)) * factor;
                });
        } else {
            const int nx = static_cast<int>(n_modes[0]);

            auto f0 = factors[0];

            Kokkos::parallel_for("precorr_type2_1d",
                Kokkos::RangePolicy<ExecSpace>(0, nx),
                KOKKOS_LAMBDA(int64_t k0) {
                    int64_t i0 = (k0 < nx / 2) ? k0 : n_grid[0] - (nx - k0);
                    int64_t j0 = (k0 + nx/2) % nx;
                    output(i0 + out_ghost) = Kokkos::conj(input(j0 + in_ghost)) * f0(k0);
                });
        }

        Kokkos::fence();
    }

}  // namespace NUFFT
}  // namespace ippl

#endif  // IPPL_NUFFT_CORRECTION_H
