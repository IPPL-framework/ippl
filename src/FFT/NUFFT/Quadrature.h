//
// NUFFT Quadrature
//   Gauss-Legendre quadrature for computing ES kernel Fourier transforms.
//
#ifndef IPPL_NUFFT_QUADRATURE_H
#define IPPL_NUFFT_QUADRATURE_H

#include <Kokkos_Core.hpp>
#include <cmath>
#include <limits>

namespace ippl {
namespace NUFFT {

    /**
     * @brief Computes Gauss-Legendre quadrature nodes and weights on [-1, 1].
     *
     * Runs on host and copies results to device-accessible views.
     *
     * @tparam MemorySpace Kokkos memory space
     * @tparam T Floating point type
     * @param n Number of quadrature points
     * @param nodes Output view for quadrature nodes
     * @param weights Output view for quadrature weights
     */
    template <typename MemorySpace, typename T = double>
    void gaussLegendre(int n,
                       Kokkos::View<T*, MemorySpace>& nodes,
                       Kokkos::View<T*, MemorySpace>& weights) {
        constexpr T eps = std::numeric_limits<T>::epsilon();

        auto h_nodes = Kokkos::create_mirror_view(Kokkos::HostSpace(), nodes);
        auto h_weights = Kokkos::create_mirror_view(Kokkos::HostSpace(), weights);

        for (int i = 0; i < n; ++i) {
            T x = std::cos(M_PI * (i + 0.75) / (n + 0.5));
            T pp, delta;

            do {
                T p1 = 1.0, p2 = 0.0;
                for (int j = 0; j < n; ++j) {
                    const T p3 = p2;
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

}  // namespace NUFFT
}  // namespace ippl

#endif  // IPPL_NUFFT_QUADRATURE_H
