/**
 * @file GaussLobatto1D.h
 * @brief Gauss–Lobatto–Legendre (GLL) nodes and weights on [-1, 1].
 *
 * Endpoints +/-1 are fixed; n-2 interior nodes are roots of P_{n-1}'.
 * An n-point GLL rule is exact for polynomials of degree up to 2n-3 (not 2n-1).
 *
 * Host-only module (see Nodes1D.h). evalLegendre uses KOKKOS_INLINE_FUNCTION for header
 * inlining only; computeGaussLobatto and View overloads run on the host.
 */
//
// GaussLobatto1D — Gauss–Lobatto–Legendre (GLL) nodes and weights on [-1, 1].
//
#ifndef IPPL_NODES1D_GAUSS_LOBATTO_1D_H
#define IPPL_NODES1D_GAUSS_LOBATTO_1D_H

#include <cstddef>

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>

#include "Nodes1D/RootFinderMethod.h"
#include "Nodes1D/Nodes1DDetail.hpp"

namespace ippl {
namespace nodes1d {

    /**
     * @brief Evaluate the Legendre polynomial P_degree(x) by three-term recurrence.
     *
     * @tparam Scalar Floating-point type (default double).
     * @param degree Polynomial degree (P_0 = 1, P_1 = x).
     * @param x Evaluation point.
     * @return P_degree(x).
     */
    template <typename Scalar = double>
    KOKKOS_INLINE_FUNCTION Scalar evalLegendre(std::size_t degree, Scalar x) {
        if (degree == 0) {
            return Scalar(1);
        }
        if (degree == 1) {
            return x;
        }
        Scalar p0 = Scalar(1);
        Scalar p1 = x;
        for (std::size_t k = 2; k <= degree; ++k) {
            const Scalar pk =
                ((2 * static_cast<Scalar>(k) - 1) * x * p1 - (static_cast<Scalar>(k) - 1) * p0)
                / static_cast<Scalar>(k);
            p0 = p1;
            p1 = pk;
        }
        return p1;
    }

    /**
     * @brief Compute GLL nodes and weights on [-1, 1] (runtime n).
     *
     * Requires n >= 2. Default: Golub–Welsch on Jacobi (1,1) interior nodes (n-2 points).
     * Alternative (RootFinderMethod::Newton): find each interior root of P_{n-1}' left to
     * right — Asymptotic Jacobi (1,1), then Chebyshev-Lobatto cosine, then local Brent in
     * a bracket between previously accepted interiors (or widened toward +1). Endpoints +/-1
     * are fixed; duplicate interiors are rejected.
     *
     * @tparam Scalar Floating-point type (default double).
     * @param n Number of quadrature points (n >= 2); includes +/-1.
     * @param nodes Output array of length n; ascending order with nodes[0] = -1, nodes[n-1] = +1.
     * @param weights Output array of length n; sum = 2.
     * @param maxNewtonIterations Maximum Newton iterations per interior root (Newton backend).
     * @param minNewtonIterations Minimum Newton iterations before convergence test (Newton only).
     * @param method Root-finding backend (GLL has no InitialGuessType parameter).
     * @pre n >= 2; nodes and weights non-null.
     * @throws std::runtime_error Newton backend on bracket failure or duplicate interiors.
     */
    template <typename Scalar = double>
    void computeGaussLobatto(std::size_t n, Scalar* nodes, Scalar* weights,
                             std::size_t maxNewtonIterations = 40,
                             std::size_t minNewtonIterations  = 1,
                             RootFinderMethod method = RootFinderMethod::GolubWelsch);

    /**
     * @brief Compute GLL nodes and weights into Kokkos Views.
     *
     * Computes on the host, then deep-copies into nodes and weights. Extent is taken from
     * nodes.extent(0) (not a separate n argument).
     *
     * @tparam ExecSpace Kokkos execution space for the destination views.
     * @tparam RealType Element type of the views (default double).
     * @param nodes Output node view; extent >= 2, must match weights.
     * @param weights Output weight view; same extent as nodes.
     * @param maxNewtonIterations See pointer overload.
     * @param minNewtonIterations See pointer overload.
     * @param method See pointer overload.
     * @pre nodes.extent(0) == weights.extent(0) and >= 2.
     */
    template <typename ExecSpace, typename RealType = double>
    void computeGaussLobatto(Kokkos::View<RealType*, typename ExecSpace::memory_space>& nodes,
                             Kokkos::View<RealType*, typename ExecSpace::memory_space>& weights,
                             std::size_t maxNewtonIterations = 40,
                             std::size_t minNewtonIterations  = 1,
                             RootFinderMethod method = RootFinderMethod::GolubWelsch) {
        detail::fillHostThenDeepCopy<ExecSpace>(
            nodes, weights, [&](std::size_t n, RealType* x, RealType* w) {
                computeGaussLobatto(n, x, w, maxNewtonIterations, minNewtonIterations, method);
            });
    }

}  // namespace nodes1d
}  // namespace ippl

#include "Nodes1D/GaussLobatto1D.hpp"

#endif
