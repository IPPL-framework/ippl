/**
 * @file RootFinderMethod.h
 * @brief Backend selector for Gauss–Jacobi and GLL node computation.
 *
 * All backends run host-side only (see Nodes1D.h); this enum selects serial setup code,
 * not a Kokkos execution-space dispatch.
 */
//
// RootFinderMethod — node/weight computation backend for Nodes1D.
//
#ifndef IPPL_NODES1D_ROOT_FINDER_METHOD_H
#define IPPL_NODES1D_ROOT_FINDER_METHOD_H

namespace ippl {
namespace nodes1d {

    /**
     * @brief Algorithm used to compute Gauss–Jacobi (and GLL interior) nodes and weights.
     *
     * Passed to computeGaussJacobi, computeGaussLegendre, and computeGaussLobatto.
     * Applications should keep the default GolubWelsch unless cross-checking or debugging.
     */
    enum class RootFinderMethod {
        /**
         * Golub–Welsch via the symmetric tridiagonal Jacobi companion matrix
         * (implicit QL eigen-solve, O(n^2)) plus one Newton polish per node. Default.
         */
        GolubWelsch,

        /**
         * Golub–Welsch via the full dense symmetric companion matrix
         * (cyclic Jacobi eigen-solve, O(n^3)). Reference / cross-check backend.
         */
        DenseGolubWelsch,
        
        /**
         * Newton root-finding with Brent fallback.
         *
         * Jacobi/Legendre: staged guess ladder over all k (InitialGuessType/Asymptotic/Chebyshev
         * StroudSecrest is never auto-appended), merge and dedupe after each stage, then a global
         * Brent scan of P_n on (-1, 1) if still incomplete.
         *
         * GLL: sequential left-to-right on each interior root of P_{n-1}' (Asymptotic Jacobi
         * (1,1), then Chebyshev-Lobatto cosine, then local Brent in a bracket bounded by
         * previously accepted interiors). No InitialGuessType parameter.
         *
         * @throws std::runtime_error if n distinct roots (Jacobi) or all interiors (GLL)
         *         cannot be isolated.
         */
        Newton,
    };

}  // namespace nodes1d
}  // namespace ippl

#endif
