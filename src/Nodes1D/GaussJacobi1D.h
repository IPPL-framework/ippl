/**
 * @file GaussJacobi1D.h
 * @brief Gauss–Jacobi quadrature nodes and weights on [-1, 1].
 *
 * Nodes are roots of P_n^(alpha,beta); weights integrate (1-x)^alpha (1+x)^beta.
 * Requires alpha, beta > -1. For alpha = beta = -1/2, delegates to computeGaussChebyshev.
 */

#ifndef IPPL_NODES1D_GAUSS_JACOBI_1D_H
#define IPPL_NODES1D_GAUSS_JACOBI_1D_H

#include <cstddef>

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "Nodes1D/GaussChebyshev1D.h"
#include "Nodes1D/RootFinderMethod.h"
#include "Types/Vector.h"

namespace ippl {
namespace nodes1d {

    /**
     * @brief Initial guess for the Newton root-finder (RootFinderMethod::Newton only).
     *
     * Ignored by Golub–Welsch backends. Newton tries the primary guess for all k first.
     * If fewer than n distinct roots are found, later ladder steps (Asymptotic, then Chebyshev
     * if they were not the primary) are merged in. StroudSecrest is never auto-appended. If the
     * ladder is still incomplete, a global Brent scan of P_n is used.
     */
    enum class InitialGuessType {
        /** Initial Newton guess for the k-th Jacobi root using a Tricomi-type large-n approximation. */
        Asymptotic,
        /** Initial Newton guess for the k-th Jacobi root using a Chebyshev node. Good near alpha = beta = -1/2.*/
        Chebyshev,
        /**
         * Empirical endpoint starts after Stroud and Secrest (1966), as in Numerical Recipes
         * gaujac. The formulas were taken from LehrFEM++ (historical IPPL FEM quadrature).
         * Not the exact Stroud-Secrest method: only the first two special cases, mirrored
         * to an ascending scan from -1; later indices use the Tricomi asymptotic.
         */
        StroudSecrest,
    };

    /**
     * @brief Compute Gauss–Jacobi nodes and weights on [-1, 1] (runtime n).
     *
     * Default backend: tridiagonal Golub–Welsch (RootFinderMethod::GolubWelsch).
     * Alternative (RootFinderMethod::Newton): staged Newton on P_n — primary
     * InitialGuessType for all k, then Asymptotic and Chebyshev if needed, then global
     * Brent on (-1, 1). See InitialGuessType for the ladder and StroudSecrest.
     *
     * @tparam Scalar Floating-point type (default double).
     * @param n Number of quadrature points (n >= 1).
     * @param alpha Jacobi parameter alpha (> -1).
     * @param beta Jacobi parameter beta (> -1).
     * @param nodes Output array of length n; ascending order.
     * @param weights Output array of length n; sum equals integral of weight on [-1, 1].
     * @param maxNewtonIterations Maximum Newton iterations per root (Newton backend only).
     * @param minNewtonIterations Minimum Newton iterations before convergence test (Newton only).
     * @param initialGuess Primary initial guess (Newton only).
     * @param method Root-finding backend.
     * @pre n >= 1; alpha, beta > -1; nodes and weights non-null.
     * @throws std::runtime_error Newton backend if n distinct roots cannot be isolated.
     */
    template <typename Scalar = double>
    void computeGaussJacobi(std::size_t n, Scalar alpha, Scalar beta, Scalar* nodes, Scalar* weights,
                            std::size_t maxNewtonIterations = 40, std::size_t minNewtonIterations = 1,
                            InitialGuessType initialGuess = InitialGuessType::Asymptotic,
                            RootFinderMethod method = RootFinderMethod::GolubWelsch);

    /**
     * @brief Compute Gauss–Jacobi nodes and weights into fixed-size Vector s.
     *
     * @tparam T Element type stored in the vectors.
     * @tparam N Number of quadrature points (compile-time).
     * @param nodes Output nodes; size N.
     * @param weights Output weights; size N.
     * @param alpha Jacobi parameter alpha.
     * @param beta Jacobi parameter beta.
     * @param maxNewtonIterations See pointer overload.
     * @param minNewtonIterations See pointer overload.
     * @param initialGuess See pointer overload.
     * @param method See pointer overload.
     */
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
