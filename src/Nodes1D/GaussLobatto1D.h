//
// GaussLobatto1D — Gauss–Lobatto–Legendre (GLL) nodes and weights on [-1, 1].
//
#ifndef IPPL_NODES1D_GAUSS_LOBATTO_1D_H
#define IPPL_NODES1D_GAUSS_LOBATTO_1D_H

#include <cstddef>

#include <Kokkos_Macros.hpp>

#include "Nodes1D/RootFinderMethod.h"
#include "Types/Vector.h"

namespace ippl {
namespace nodes1d {

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
     * @brief Runtime GLL nodes/weights on [-1, 1]. Requires n >= 2.
     *
     * Default: Golub–Welsch on Jacobi(1,1) interior nodes (n−2 points).
     * Alternative: Newton on P'_{n-1} with duplicate detection (throws).
     */
    template <typename Scalar = double>
    void computeGaussLobatto(std::size_t n, Scalar* nodes, Scalar* weights,
                             std::size_t maxNewtonIterations = 40,
                             std::size_t minNewtonIterations  = 1,
                             RootFinderMethod method = RootFinderMethod::GolubWelsch);

    template <typename T, unsigned N>
    void computeGaussLobatto(Vector<T, N>& nodes, Vector<T, N>& weights,
                             std::size_t maxNewtonIterations = 40,
                             std::size_t minNewtonIterations  = 1,
                             RootFinderMethod method = RootFinderMethod::GolubWelsch) {
        static_assert(N >= 2, "Gauss–Lobatto requires at least 2 nodes");
        using Scalar = double;
        Scalar nbuf[N];
        Scalar wbuf[N];
        computeGaussLobatto<Scalar>(N, nbuf, wbuf, maxNewtonIterations, minNewtonIterations, method);
        for (unsigned i = 0; i < N; ++i) {
            nodes[i]   = static_cast<T>(nbuf[i]);
            weights[i] = static_cast<T>(wbuf[i]);
        }
    }

}  // namespace nodes1d
}  // namespace ippl

#include "Nodes1D/GaussLobatto1D.hpp"

#endif
