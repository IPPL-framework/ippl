/**
 * @file RootFinderMethod.h
 * @brief Backend selector for Gauss–Jacobi and GLL node computation.
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
     * Passed to computeGaussJacobi and computeGaussLegendre. Applications should keep the
     * default GolubWelsch unless cross-checking or debugging.
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
         * Independent Newton per root with configurable initial guesses and a Brent bracket
         * fallback when the Newton ladder fails to isolate n distinct roots.
         * @throws std::runtime_error on unresolved duplicates or non-convergence.
         */
        Newton,
    };

}  // namespace nodes1d
}  // namespace ippl

#endif
