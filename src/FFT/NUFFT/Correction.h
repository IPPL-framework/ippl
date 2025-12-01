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
#include "Types/Vector.h"

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
        using complex_type  = Kokkos::complex<T>;
        using memory_space  = typename ExecSpace::memory_space;

        constexpr int q = 100;
        Kokkos::View<T*, memory_space> nodes("quad_nodes", q);
        Kokkos::View<T*, memory_space> weights("quad_weights", q);

        gaussLegendre<memory_space, T>(q, nodes, weights);

        const T alpha = M_PI * kernel.width() / n_grid;
        const T beta  = kernel.beta();
        const int w   = kernel.width();

        Kokkos::parallel_for(
            "compute_deconv_factors",
            Kokkos::RangePolicy<ExecSpace>(0, n_modes),
            KOKKOS_LAMBDA(const int64_t k) {
                int freq = (k < n_modes / 2) ? k : k - n_modes;
                T ft = 0.0;

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
     * @brief Apply deconvolution correction for Type 1 NUFFT (post-FFT), 3D only.
     *
     * Maps upsampled FFT output to Fourier modes with deconvolution.
     * Type 1: grid -> modes
     *
     * @tparam ExecSpace  Kokkos execution space
     * @tparam T          Floating point type
     * @tparam GridField  Complex Field type for the upsampled grid (3D)
     * @tparam ModesField Complex Field type for the Fourier modes (3D)
     *
     */
    template <typename ExecSpace, typename T, typename GridField, typename ModesField>
    void applyDeconvolutionType1(
        GridField& input_field,
        const std::array<
            Kokkos::View<Kokkos::complex<T>*, typename ExecSpace::memory_space>, 3>& factors,
        ModesField& output_field,
        const Vector<size_t, 3>& n_modes,
        const Vector<size_t, 3>& n_grid) {

        static_assert(GridField::dim  == 3, "applyDeconvolutionType1: GridField must be 3D");
        static_assert(ModesField::dim == 3, "applyDeconvolutionType1: ModesField must be 3D");

        using complex_type = Kokkos::complex<T>;

        // Views + ghost widths
        auto input  = input_field.getView();
        auto output = output_field.getView();

        const int in_ghost  = input_field.getNghost();
        const int out_ghost = output_field.getNghost();

        // Logical local extents of the modes field (no ghosts)
        const auto& out_layout = output_field.getLayout();
        const auto& out_nd     = out_layout.getLocalNDIndex();

        const int64_t nx_local = out_nd[0].length();
        const int64_t ny_local = out_nd[1].length();
        const int64_t nz_local = out_nd[2].length();

        // Global mode counts
        const int64_t nx = static_cast<int64_t>(n_modes[0]);
        const int64_t ny = static_cast<int64_t>(n_modes[1]);
        const int64_t nz = static_cast<int64_t>(n_modes[2]);

        const int64_t gx = static_cast<int64_t>(n_grid[0]);
        const int64_t gy = static_cast<int64_t>(n_grid[1]);
        const int64_t gz = static_cast<int64_t>(n_grid[2]);

        // Detect single-rank per dimension (local == global)
        const bool single_rank_x = (nx_local == nx);
        const bool single_rank_y = (ny_local == ny);
        const bool single_rank_z = (nz_local == nz);

        // Capture factors by value
        auto f0 = factors[0];
        auto f1 = factors[1];
        auto f2 = factors[2];

        Kokkos::parallel_for(
            "deconv_type1_3d",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                {0, 0, 0},
                {nx_local, ny_local, nz_local}),
            KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                // Compute shifted indices
                const int ii_shift = (i + nx/2) % nx;
                const int jj_shift = (j + ny/2) % ny;
                const int kk_shift = (k + nz/2) % nz;

                complex_type factor = f0(ii_shift) * f1(jj_shift) * f2(kk_shift);

                int64_t i_in, j_in, k_in;

                if (single_rank_x) {
                    i_in = (i < nx/2) ? (gx - nx/2 + i) : (i - nx/2);
                } else {
                    i_in = i;
                }

                if (single_rank_y) {
                    j_in = (j < ny/2) ? (gy - ny/2 + j) : (j - ny/2);
                } else {
                    j_in = j;
                }

                if (single_rank_z) {
                    k_in = (k < nz/2) ? (gz - nz/2 + k) : (k - nz/2);
                } else {
                    k_in = k;
                }

                output(i + out_ghost, j + out_ghost, k + out_ghost) =
                    Kokkos::conj(input(i_in + in_ghost,
                                       j_in + in_ghost,
                                       k_in + in_ghost) * factor);
            });

        Kokkos::fence();
    }

    /**
     * @brief Apply pre-correction for Type 2 NUFFT (pre-FFT), 3D only.
     *
     * Multiplies input modes by correction factors and zero-pads to upsampled grid.
     * Type 2: modes -> grid (with zero-padding)
     *
     * @tparam ExecSpace  Kokkos execution space
     * @tparam T          Floating point type
     * @tparam ModesField Complex modes field (3D)
     * @tparam GridField  Complex grid field (3D)
     */
    template <typename ExecSpace, typename T, typename ModesField, typename GridField>
    void applyPreCorrectionType2(
        ModesField& input_field,
        const std::array<
            Kokkos::View<Kokkos::complex<T>*, typename ExecSpace::memory_space>, 3>& factors,
        GridField& output_field,
        const Vector<size_t, 3>& n_modes,
        const Vector<size_t, 3>& n_grid) {

        static_assert(GridField::dim  == 3, "applyPreCorrectionType2: GridField must be 3D");
        static_assert(ModesField::dim == 3, "applyPreCorrectionType2: ModesField must be 3D");

        using complex_type = Kokkos::complex<T>;

        auto input  = input_field.getView();
        auto output = output_field.getView();

        const int in_ghost  = input_field.getNghost();
        const int out_ghost = output_field.getNghost();

        // Zero the upsampled grid
        Kokkos::deep_copy(output, complex_type(0, 0));

        // Logical local extents of the modes field (no ghosts)
        const auto& in_layout = input_field.getLayout();
        const auto& in_nd     = in_layout.getLocalNDIndex();

        const int64_t nx_local = in_nd[0].length();
        const int64_t ny_local = in_nd[1].length();
        const int64_t nz_local = in_nd[2].length();

        // Global mode counts
        const int64_t nx = static_cast<int64_t>(n_modes[0]);
        const int64_t ny = static_cast<int64_t>(n_modes[1]);
        const int64_t nz = static_cast<int64_t>(n_modes[2]);

        const int64_t gx = static_cast<int64_t>(n_grid[0]);
        const int64_t gy = static_cast<int64_t>(n_grid[1]);
        const int64_t gz = static_cast<int64_t>(n_grid[2]);

        // Detect single-rank per dimension
        const bool single_rank_x = (nx_local == nx);
        const bool single_rank_y = (ny_local == ny);
        const bool single_rank_z = (nz_local == nz);

        auto f0 = factors[0];
        auto f1 = factors[1];
        auto f2 = factors[2];

        Kokkos::parallel_for(
            "precorr_type2_3d",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                {0, 0, 0},
                {nx_local, ny_local, nz_local}),
            KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                // Compute shifted indices
                const int ii_shift = (i + nx/2) % nx;
                const int jj_shift = (j + ny/2) % ny;
                const int kk_shift = (k + nz/2) % nz;

                complex_type factor = f0(ii_shift) * f1(jj_shift) * f2(kk_shift);

                int64_t i_out, j_out, k_out;

                if (single_rank_x) {
                    i_out = (i < nx/2) ? (gx - nx/2 + i) : (i - nx/2);
                } else {
                    i_out = i;
                }

                if (single_rank_y) {
                    j_out = (j < ny/2) ? (gy - ny/2 + j) : (j - ny/2);
                } else {
                    j_out = j;
                }

                if (single_rank_z) {
                    k_out = (k < nz/2) ? (gz - nz/2 + k) : (k - nz/2);
                } else {
                    k_out = k;
                }

                // NOTE: keeping your current conjugation choice here
                output(i_out + out_ghost,
                       j_out + out_ghost,
                       k_out + out_ghost) =
                    Kokkos::conj(input(i + in_ghost,
                                       j + in_ghost,
                                       k + in_ghost)) * factor;
            });

        Kokkos::fence();
    }

}  // namespace NUFFT
}  // namespace ippl

#endif  // IPPL_NUFFT_CORRECTION_H
