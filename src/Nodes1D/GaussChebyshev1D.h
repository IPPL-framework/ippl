/**
 * @file GaussChebyshev1D.h
 * @brief Closed-form Gauss–Chebyshev quadrature on [-1, 1].
 *
 * Gauss–Chebyshev is Gauss–Jacobi with alpha = beta = -1/2. The rule is exact for polynomials
 * of degree up to 2n-1 with weight (1-x^2)^(-1/2).
 */
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
     * @brief i-th Chebyshev node of the first kind on [-1, 1], ascending in i.
     *
     * x_i = -cos((2i+1)*pi / (2n)) for i = 0, ..., n-1. Index i = 0 is nearest -1;
     * i = numNodes - 1 is nearest +1.
     *
     * @tparam Scalar Floating-point type (default double).
     * @param i Node index, 0 <= i < numNodes.
     * @param numNodes Number of nodes n.
     * @return Chebyshev node x_i.
     * @pre numNodes >= 1 and i < numNodes.
     */
    template <typename Scalar = double>
    KOKKOS_INLINE_FUNCTION Scalar chebyshevNode(std::size_t i, std::size_t numNodes) {
        return -Kokkos::cos((2.0 * static_cast<Scalar>(i) + 1.0) * Kokkos::numbers::pi_v<Scalar>
                            / (2.0 * static_cast<Scalar>(numNodes)));
    }

    /**
     * @brief Compute Gauss–Chebyshev nodes and weights on [-1, 1] (closed form).
     *
     * Nodes are zeros of T_n (chebyshevNode). Weights are constant pi/n. Equivalent to
     * computeGaussJacobi with alpha = beta = -1/2.
     *
     * @tparam Scalar Floating-point type (default double).
     * @param n Number of quadrature points (n >= 1).
     * @param nodes Output array of length n; nodes in ascending order.
     * @param weights Output array of length n.
     * @pre n >= 1; nodes and weights are non-null pointers to arrays of length n.
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

    /**
     * @brief Compute Gauss–Chebyshev nodes and weights into fixed-size Vector s.
     *
     * @tparam T Element type stored in the vectors.
     * @tparam N Number of quadrature points (compile-time).
     * @param nodes Output nodes; size N.
     * @param weights Output weights; size N.
     */
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

    /**
     * @brief Compute Gauss–Chebyshev nodes and weights into Kokkos Views on device.
     *
     * Computes on the host, then deep-copies into nodes and weights.
     *
     * @tparam ExecSpace Kokkos execution space used for the destination views.
     * @tparam RealType Element type of the views (default double).
     * @param nodes Output node view; extent must equal that of weights and be >= 1.
     * @param weights Output weight view; same extent as nodes.
     * @pre nodes.extent(0) == weights.extent(0) and >= 1.
     */
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
