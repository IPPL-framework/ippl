/**
 * @file GaussChebyshev1D.h
 * @brief Closed-form Gauss–Chebyshev quadrature on [-1, 1].
 *
 * Gauss–Chebyshev is Gauss–Jacobi with alpha = beta = -1/2. The rule is exact for polynomials
 * of degree up to 2n-1 with weight (1-x^2)^(-1/2).
 *
 * Host-only module (see Nodes1D.h). chebyshevNode uses KOKKOS_INLINE_FUNCTION for header
 * inlining only; computeGaussChebyshev and View overloads run on the host.
 */
//
// GaussChebyshev1D — Gauss–Chebyshev (Jacobi α = β = −1/2) on [-1, 1], closed form.
//
#ifndef IPPL_NODES1D_GAUSS_CHEBYSHEV_1D_H
#define IPPL_NODES1D_GAUSS_CHEBYSHEV_1D_H

#include <cassert>
#include <cstddef>

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "Nodes1D/Nodes1DDetail.hpp"

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
     * @brief Compute Gauss–Chebyshev nodes and weights into Kokkos Views.
     *
     * Computes on the host, then deep-copies into nodes and weights. Extent is taken from
     * nodes.extent(0) (not a separate n argument).
     *
     * @tparam ExecSpace Kokkos execution space for the destination views.
     * @tparam RealType Element type of the views (default double).
     * @param nodes Output node view; extent >= 1, must match weights.
     * @param weights Output weight view; same extent as nodes.
     * @pre nodes.extent(0) == weights.extent(0) and >= 1.
     */
    template <typename ExecSpace, typename RealType = double>
    void computeGaussChebyshev(Kokkos::View<RealType*, typename ExecSpace::memory_space>& nodes,
                               Kokkos::View<RealType*, typename ExecSpace::memory_space>& weights) {
        detail::fillHostThenDeepCopy<ExecSpace>(
            nodes, weights,
            [&](std::size_t n, RealType* x, RealType* w) { computeGaussChebyshev(n, x, w); });
    }

    /**
     * @brief Compute Gauss–Chebyshev nodes and weights into fixed-size Vectors (compile-time N).
     *
     * Thin wrapper around the pointer overload; n is N.
     *
     * @tparam T Element type stored in the vectors.
     * @tparam N Number of quadrature points (compile-time, N >= 1).
     * @tparam Scalar Working precision for the internal computation (default double).
     * @param nodes Output nodes; size N.
     * @param weights Output weights; size N.
     */
    template <typename T, unsigned N, typename Scalar = double>
    void computeGaussChebyshev(Vector<T, N>& nodes, Vector<T, N>& weights) {
        static_assert(N >= 1, "Gauss-Chebyshev quadrature requires N >= 1");
        detail::fillFixedVectors<T, N, Scalar>(
            nodes, weights,
            [](std::size_t n, Scalar* x, Scalar* w) { computeGaussChebyshev(n, x, w); });
    }

}  // namespace nodes1d
}  // namespace ippl

#endif
