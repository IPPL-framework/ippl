//
// RootFinderMethod — node/weight computation backend for Nodes1D.
//
#ifndef IPPL_NODES1D_ROOT_FINDER_METHOD_H
#define IPPL_NODES1D_ROOT_FINDER_METHOD_H

namespace ippl {
namespace nodes1d {

    enum class RootFinderMethod {
        /**
         * Golub–Welsch via the symmetric tridiagonal Jacobi companion matrix
         * (QL eigen-solve, O(n²)) + one Newton polish. Default.
         */
        GolubWelsch,
        /**
         * Golub–Welsch via the full dense symmetric companion matrix
         * (cyclic Jacobi eigen-solve, O(n³)). Useful as a reference / cross-check.
         */
        DenseGolubWelsch,
        /**
         * Independent Newton per root with asymptotic / Chebyshev starts and a
         * Brent bracket fallback. Throws std::runtime_error on unresolved
         * duplicates or non-convergence.
         */
        Newton,
    };

}  // namespace nodes1d
}  // namespace ippl

#endif
