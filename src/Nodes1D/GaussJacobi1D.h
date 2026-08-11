//
// GaussJacobi1D — 1D Gauss–Jacobi nodes and weights on [-1, 1].
//
#ifndef IPPL_NODES1D_GAUSS_JACOBI_1D_H
#define IPPL_NODES1D_GAUSS_JACOBI_1D_H

#include <cstddef>

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "Nodes1D/RootFinderMethod.h"
#include "Types/Vector.h"

namespace ippl {
namespace nodes1d {

    enum class InitialGuessType {
        /** Tricomi-style asymptotic for the k-th Jacobi zero (preferred Newton default). */
        Asymptotic,
        Chebyshev,
        LehrFEM,
    };

    /**
     * @brief i-th Chebyshev node on [-1, 1], ascending in i (i=0 near −1).
     */
    template <typename Scalar = double>
    KOKKOS_INLINE_FUNCTION Scalar chebyshevNode(std::size_t i, std::size_t numNodes) {
        return -Kokkos::cos((2.0 * static_cast<Scalar>(i) + 1.0) * Kokkos::numbers::pi_v<Scalar>
                            / (2.0 * static_cast<Scalar>(numNodes)));
    }

    /**
     * @brief Compute Gauss–Jacobi nodes and weights on [-1, 1] (runtime n).
     *
     * Default: tridiagonal Golub–Welsch. Alternative: Newton + Brent retry ladder.
     */
    template <typename Scalar = double>
    void computeGaussJacobi(std::size_t n, Scalar alpha, Scalar beta, Scalar* nodes, Scalar* weights,
                            std::size_t maxNewtonIterations = 40, std::size_t minNewtonIterations = 1,
                            InitialGuessType initialGuess = InitialGuessType::Asymptotic,
                            RootFinderMethod method = RootFinderMethod::GolubWelsch);

    template <typename T, unsigned N>
    void computeGaussJacobi(Vector<T, N>& nodes, Vector<T, N>& weights, T alpha, T beta,
                            std::size_t maxNewtonIterations = 40, std::size_t minNewtonIterations = 1,
                            InitialGuessType initialGuess = InitialGuessType::Asymptotic,
                            RootFinderMethod method = RootFinderMethod::GolubWelsch) {
        using Scalar = double;
        Scalar nbuf[N];
        Scalar wbuf[N];
        computeGaussJacobi<Scalar>(N, static_cast<Scalar>(alpha), static_cast<Scalar>(beta), nbuf,
                                   wbuf, maxNewtonIterations, minNewtonIterations, initialGuess,
                                   method);
        for (unsigned i = 0; i < N; ++i) {
            nodes[i]   = static_cast<T>(nbuf[i]);
            weights[i] = static_cast<T>(wbuf[i]);
        }
    }

}  // namespace nodes1d
}  // namespace ippl

#include "Nodes1D/GaussJacobi1D.hpp"

#endif
