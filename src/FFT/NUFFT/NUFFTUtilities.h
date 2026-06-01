#ifndef IPPL_NUFFT_UTILITIES_H
#define IPPL_NUFFT_UTILITIES_H

/**
 * @file NUFFTUtilities.h
 * @brief NUFFT utility helpers used by the native Kokkos NUFFT engine.
 */

#include <cmath>
#include <limits>

#include <Kokkos_Core.hpp>

namespace ippl {
namespace nufft {

    /**
     * @brief Computes Gauss-Legendre quadrature nodes and weights on [-1, 1].
     *
     * Runs on host and copies results to the device-accessible views.
     */
    template<typename ExecSpace, typename RealType = double>
    void gauss_legendre(int n,
                        Kokkos::View<RealType *, typename ExecSpace::memory_space> &nodes,
                        Kokkos::View<RealType *, typename ExecSpace::memory_space> &weights) {
        constexpr RealType eps = std::numeric_limits<RealType>::epsilon();

        auto h_nodes = Kokkos::create_mirror_view(Kokkos::HostSpace(), nodes);
        auto h_weights = Kokkos::create_mirror_view(Kokkos::HostSpace(), weights);

        for (int i = 0; i < n; ++i) {
            RealType x = std::cos(Kokkos::numbers::pi_v<RealType> * (i + 0.75) / (n + 0.5));
            RealType pp, delta;

            do {
                RealType p1 = 1.0, p2 = 0.0;
                for (int j = 0; j < n; ++j) {
                    const RealType p3 = p2;
                    p2 = p1;
                    p1 = ((2.0 * j + 1.0) * x * p2 - j * p3) / (j + 1.0);
                }
                pp = n * (x * p1 - p2) / (x * x - 1.0);
                delta = p1 / pp;
                x -= delta;
            } while (std::abs(delta) > eps);

            h_nodes(i) = x;
            h_weights(i) = 2.0 / ((1.0 - x * x) * pp * pp);
        }

        // Exploit symmetry
        for (int i = 0; i < n / 2; ++i) {
            const int j = n - 1 - i;
            h_nodes(j) = -h_nodes(i);
            h_weights(j) = h_weights(i);
        }

        Kokkos::deep_copy(nodes, h_nodes);
        Kokkos::deep_copy(weights, h_weights);
    }

} // namespace nufft
} // namespace ippl

#endif // IPPL_NUFFT_UTILITIES_H
