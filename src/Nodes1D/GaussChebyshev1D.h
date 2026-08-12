//
// GaussChebyshev1D — Gauss–Chebyshev (Jacobi α = β = −1/2) on [-1, 1], closed form.
//
#ifndef IPPL_NODES1D_GAUSS_CHEBYSHEV_1D_H
#define IPPL_NODES1D_GAUSS_CHEBYSHEV_1D_H

#include <cassert>
#include <cstddef>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "Types/Vector.h"

namespace ippl {
namespace nodes1d {

    /**
     * @brief i-th Chebyshev node on [-1, 1], ascending in i (i=0 near −1).
     */
    template <typename Scalar = double>
    KOKKOS_INLINE_FUNCTION Scalar chebyshevNode(std::size_t i, std::size_t numNodes) {
        return -Kokkos::cos((2.0 * static_cast<Scalar>(i) + 1.0) * Kokkos::numbers::pi_v<Scalar>
                            / (2.0 * static_cast<Scalar>(numNodes)));
    }

    /**
     * @brief Gauss–Chebyshev nodes and weights on [-1, 1] (closed form).
     *
     * Nodes: zeros of T_n (chebyshevNode). Weights: π/n. Same as Gauss–Jacobi with α = β = −1/2.
     */
    template <typename Scalar = double>
    void computeGaussChebyshev(std::size_t n, Scalar* nodes, Scalar* weights) {
        assert(n >= 1);
        assert(nodes != nullptr && weights != nullptr);
        const Scalar w = Kokkos::numbers::pi_v<Scalar> / static_cast<Scalar>(n);
        for (std::size_t i = 0; i < n; ++i) {
            nodes[i]   = chebyshevNode<Scalar>(i, n);
            weights[i] = w;
        }
    }

    template <typename T, unsigned N>
    void computeGaussChebyshev(Vector<T, N>& nodes, Vector<T, N>& weights) {
        using Scalar = double;
        Scalar nbuf[N];
        Scalar wbuf[N];
        computeGaussChebyshev<Scalar>(N, nbuf, wbuf);
        for (unsigned i = 0; i < N; ++i) {
            nodes[i]   = static_cast<T>(nbuf[i]);
            weights[i] = static_cast<T>(wbuf[i]);
        }
    }

    template <typename ExecSpace, typename RealType = double>
    void computeGaussChebyshev(Kokkos::View<RealType*, typename ExecSpace::memory_space>& nodes,
                               Kokkos::View<RealType*, typename ExecSpace::memory_space>& weights) {
        const int n = static_cast<int>(nodes.extent(0));
        assert(weights.extent(0) == nodes.extent(0));
        assert(n >= 1);

        auto h_nodes   = Kokkos::create_mirror_view(Kokkos::HostSpace(), nodes);
        auto h_weights = Kokkos::create_mirror_view(Kokkos::HostSpace(), weights);

        std::vector<RealType> nbuf(static_cast<std::size_t>(n));
        std::vector<RealType> wbuf(static_cast<std::size_t>(n));
        computeGaussChebyshev(static_cast<std::size_t>(n), nbuf.data(), wbuf.data());
        for (int i = 0; i < n; ++i) {
            h_nodes(i)   = nbuf[static_cast<std::size_t>(i)];
            h_weights(i) = wbuf[static_cast<std::size_t>(i)];
        }
        Kokkos::deep_copy(nodes, h_nodes);
        Kokkos::deep_copy(weights, h_weights);
    }

}  // namespace nodes1d
}  // namespace ippl

#endif
