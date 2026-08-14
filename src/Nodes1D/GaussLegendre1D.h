/**
 * @file GaussLegendre1D.h
 * @brief Gauss–Legendre quadrature on [-1, 1] (Jacobi with alpha = beta = 0).
 *
 * Weights sum to 2. Exact for polynomials of degree up to 2n-1 with weight 1.
 *
 * Host-only module (see Nodes1D.h): thin wrapper around computeGaussJacobi.
 */
//
// GaussLegendre1D — Gauss–Legendre (Jacobi α = β = 0) on [-1, 1].
//
#ifndef IPPL_NODES1D_GAUSS_LEGENDRE_1D_H
#define IPPL_NODES1D_GAUSS_LEGENDRE_1D_H

#include "Nodes1D/GaussJacobi1D.h"

namespace ippl {
namespace nodes1d {

    /**
     * @brief Compute Gauss–Legendre nodes and weights on [-1, 1] (runtime n).
     *
     * Thin wrapper around computeGaussJacobi with alpha = beta = 0.
     *
     * @tparam Scalar Floating-point type (default double).
     * @param n Number of quadrature points (n >= 1).
     * @param nodes Output array of length n; ascending order.
     * @param weights Output array of length n; sum = 2.
     * @param maxNewtonIterations Passed to computeGaussJacobi (Newton backend).
     * @param minNewtonIterations Passed to computeGaussJacobi (Newton backend).
     * @param initialGuess Passed to computeGaussJacobi (Newton backend).
     * @param method Root-finding backend.
     */
    template <typename Scalar = double>
    void computeGaussLegendre(std::size_t n, Scalar* nodes, Scalar* weights,
                              std::size_t maxNewtonIterations = 40,
                              std::size_t minNewtonIterations  = 1,
                              InitialGuessType initialGuess = InitialGuessType::Asymptotic,
                              RootFinderMethod method = RootFinderMethod::GolubWelsch) {
        computeGaussJacobi(n, Scalar(0), Scalar(0), nodes, weights, maxNewtonIterations,
                           minNewtonIterations, initialGuess, method);
    }

    /**
     * @brief Compute Gauss–Legendre nodes and weights into Kokkos Views.
     *
     * Computes on the host, then deep-copies into nodes and weights. Extent is taken from
     * nodes.extent(0) (not a separate n argument).
     *
     * @tparam ExecSpace Kokkos execution space for the destination views.
     * @tparam RealType Element type of the views (default double).
     * @param nodes Output node view; extent >= 1, must match weights.
     * @param weights Output weight view; same extent as nodes.
     * @param maxNewtonIterations Passed to computeGaussLegendre (Newton backend).
     * @param minNewtonIterations Passed to computeGaussLegendre (Newton backend).
     * @param initialGuess Passed to computeGaussLegendre (Newton backend).
     * @param method Root-finding backend.
     * @pre nodes.extent(0) == weights.extent(0) and >= 1.
     */
    template <typename ExecSpace, typename RealType = double>
    void computeGaussLegendre(Kokkos::View<RealType*, typename ExecSpace::memory_space>& nodes,
                              Kokkos::View<RealType*, typename ExecSpace::memory_space>& weights,
                              std::size_t maxNewtonIterations = 40,
                              std::size_t minNewtonIterations  = 1,
                              InitialGuessType initialGuess = InitialGuessType::Asymptotic,
                              RootFinderMethod method = RootFinderMethod::GolubWelsch) {
        detail::fillHostThenDeepCopy<ExecSpace>(
            nodes, weights, [&](std::size_t n, RealType* x, RealType* w) {
                computeGaussLegendre(n, x, w, maxNewtonIterations, minNewtonIterations,
                                     initialGuess, method);
            });
    }

    /**
     * @brief Compute Gauss–Legendre nodes and weights into fixed-size Vectors (compile-time N).
     *
     * Thin wrapper around computeGaussJacobi with alpha = beta = 0; n is N.
     *
     * @tparam T Element type stored in the vectors.
     * @tparam N Number of quadrature points (compile-time, N >= 1).
     * @tparam Scalar Working precision for the internal computation (default double).
     */
    template <typename T, unsigned N, typename Scalar = double>
    void computeGaussLegendre(Vector<T, N>& nodes, Vector<T, N>& weights,
                              std::size_t maxNewtonIterations = 40,
                              std::size_t minNewtonIterations  = 1,
                              InitialGuessType initialGuess = InitialGuessType::Asymptotic,
                              RootFinderMethod method = RootFinderMethod::GolubWelsch) {
        computeGaussJacobi(nodes, weights, Scalar(0), Scalar(0), maxNewtonIterations,
                           minNewtonIterations, initialGuess, method);
    }

}  // namespace nodes1d
}  // namespace ippl

#endif
