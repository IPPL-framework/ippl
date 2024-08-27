// Class GaussJacobiQuadrature
//   The GaussJacobiQuadrature class. This is a class representing a Gauss-Jacobi quadrature
//   The algoirthm in computeNodesAndWeights is based on the LehrFEM++ library.
//   https://craffael.github.io/lehrfempp/gauss__quadrature_8cc_source.html

#ifndef IPPL_GAUSSJACOBIQUADRATURE_H
#define IPPL_GAUSSJACOBIQUADRATURE_H

#include "FEM/Quadrature/Quadrature.h"

namespace ippl {

    enum InitialGuessType {
        Chebyshev,
        LehrFEM,
    };

    /**
     * @brief This is class represents the Gauss-Jacobi quadrature rule
     * on a reference element.
     *
     * @tparam T floating point number type of the quadrature nodes and weights
     * @tparam NumNodes1D number of quadrature nodes for one dimension
     * @tparam ElementType element type for which the quadrature rule is defined
     */
    template <typename T, unsigned NumNodes1D, typename ElementType>
    class GaussJacobiQuadrature : public Quadrature<T, NumNodes1D, ElementType> {
    public:
        // a higher precision floating point number type used for the computation
        // of the quadrature nodes and weights
        using scalar_t = long double;  // might be equivalant to double, depending on compiler

        /**
         * @brief Construct a new Gauss Jacobi Quadrature rule object
         *
         * @param ref_element reference element to compute the quadrature nodes on
         * @param alpha first Jacobi parameter alpha
         * @param beta second Jacobi parameter beta
         * @param max_newton_itersations maximum number of Newton iterations (default 10)
         * @param min_newton_iterations minimum number of Newton iterations (default 1)
         */
        GaussJacobiQuadrature(const ElementType& ref_element, const T& alpha, const T& beta,
                              const size_t& max_newton_itersations = 10,
                              const size_t& min_newton_iterations  = 1);

        /**
         * Computes the quadrature nodes and weights and stores them in the
         * quadrature nodes and weights arrays.
         */
        void computeNodesAndWeights() override;

        /**
         * @brief Returns the i-th Chebyshev node, used as initial guess for the Newton iterations.
         *
         * @param i index of the Chebyshev node
         *
         * @return scalar_t - i-th Chebyshev node
         */
        scalar_t getChebyshevNodes(const size_t& i) const;  // FIXME maybe move somewhere else?

    private:
        /**
         * @brief Computes the initial guess for the Newton iterations, the way they are computed in
         * the implementation from LehrFEM++.
         *
         * @param i index of the initial guess (corresponding to the i-th quadrature node)
         * @param integration_nodes the integration nodes
         *
         * @return scalar_t - initial guess
         */
        scalar_t getLehrFEMInitialGuess(
            const size_t& i, const Vector<scalar_t, NumNodes1D>& integration_nodes) const;

        const T alpha_m;
        const T beta_m;

        const size_t max_newton_iterations_m;
        const size_t min_newton_iterations_m;
    };

    /**
     * @brief This is class represents the Gauss-Legendre quadrature rule.
     * It is a special case of the Gauss-Jacobi quadrature rule with alpha = beta = 0.0.
     *
     * @tparam T floating point number type of the quadrature nodes and weights
     * @tparam NumNodes1D number of quadrature nodes for one dimension
     * @tparam ElementType element type for which the quadrature rule is defined
     */
    template <typename T, unsigned NumNodes1D, typename ElementType>
    class GaussLegendreQuadrature : public GaussJacobiQuadrature<T, NumNodes1D, ElementType> {
    public:
        /**
         * @brief Construct a new Gauss Legendre Quadrature rule object
         *
         * @param ref_element reference element to compute the quadrature nodes on
         * @param max_newton_itersations maximum number of Newton iterations (default 10)
         * @param min_newton_iterations minimum number of Newton iterations (default 1)
         */
        GaussLegendreQuadrature(const ElementType& ref_element,
                                const size_t& max_newton_itersations = 10,
                                const size_t& min_newton_iterations  = 1)
            : GaussJacobiQuadrature<T, NumNodes1D, ElementType>(
                ref_element, 0.0, 0.0, max_newton_itersations, min_newton_iterations) {}
    };

    /**
     * @brief This is class represents the Chebyshev-Gauss quadrature rule.
     * It is a special case of the Gauss-Jacobi quadrature rule with alpha = beta = -0.5.
     *
     * @tparam T floating point number type of the quadrature nodes and weights
     * @tparam NumNodes1D number of quadrature nodes for one dimension
     * @tparam ElementType element type for which the quadrature rule is defined
     */
    template <typename T, unsigned NumNodes1D, typename ElementType>
    class ChebyshevGaussQuadrature : public GaussJacobiQuadrature<T, NumNodes1D, ElementType> {
    public:
        /**
         * @brief Construct a new Chebyshev Gauss Quadrature rule object
         *
         * @param ref_element reference element to compute the quadrature nodes on
         * @param max_newton_itersations maximum number of Newton iterations (default 10)
         * @param min_newton_iterations minimum number of Newton iterations (default 1)
         */
        ChebyshevGaussQuadrature(const ElementType& ref_element,
                                 const size_t& max_newton_itersations = 10,
                                 const size_t& min_newton_iterations  = 1)
            : GaussJacobiQuadrature<T, NumNodes1D, ElementType>(
                ref_element, -0.5, -0.5, max_newton_itersations, min_newton_iterations) {}
    };

}  // namespace ippl

#include "FEM/Quadrature/GaussJacobiQuadrature.hpp"

#endif
