// Class GaussJacobiQuadrature
//   Gauss-Jacobi quadrature on [-1, 1]. Node/weight generation delegates to ippl::nodes1d.

#ifndef IPPL_GAUSSJACOBIQUADRATURE_H
#define IPPL_GAUSSJACOBIQUADRATURE_H

#include <cassert>

#include "FEM/Quadrature/Quadrature.h"
#include "Nodes1D/GaussJacobi1D.h"

namespace ippl {

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
        /**
         * @brief Construct a Gauss-Jacobi quadrature rule on [-1, 1].
         *
         * @param ref_element reference element to compute the quadrature nodes on
         * @param alpha first Jacobi parameter alpha (> -1)
         * @param beta second Jacobi parameter beta (> -1)
         * @param max_newton_itersations maximum Newton iterations (Nodes1D Newton backend only)
         * @param min_newton_iterations minimum Newton iterations (Nodes1D Newton backend only)
         */
        GaussJacobiQuadrature(const ElementType& ref_element, const T& alpha, const T& beta,
                              const size_t& max_newton_itersations = 10,
                              const size_t& min_newton_iterations  = 1);

        /** @brief Fill integration_nodes_m and weights_m via nodes1d::computeGaussJacobi. */
        void computeNodesAndWeights() override {
            nodes1d::computeGaussJacobi(this->integration_nodes_m, this->weights_m, alpha_m, beta_m,
                                        max_newton_iterations_m, min_newton_iterations_m);
        }

    private:
        const T alpha_m;
        const T beta_m;

        const size_t max_newton_iterations_m;
        const size_t min_newton_iterations_m;
    };

    /**
     * @brief Gauss-Legendre quadrature (Jacobi with alpha = beta = 0).
     *
     * @tparam T floating point number type of the quadrature nodes and weights
     * @tparam NumNodes1D number of quadrature nodes for one dimension
     * @tparam ElementType element type for which the quadrature rule is defined
     */
    template <typename T, unsigned NumNodes1D, typename ElementType>
    class GaussLegendreQuadrature : public GaussJacobiQuadrature<T, NumNodes1D, ElementType> {
    public:
        /**
         * @brief Gauss-Legendre quadrature (Jacobi with alpha = beta = 0).
         *
         * @param ref_element reference element to compute the quadrature nodes on
         * @param max_newton_itersations maximum Newton iterations (Nodes1D Newton backend only)
         * @param min_newton_iterations minimum Newton iterations (Nodes1D Newton backend only)
         */
        GaussLegendreQuadrature(const ElementType& ref_element,
                                const size_t& max_newton_itersations = 10,
                                const size_t& min_newton_iterations  = 1)
            : GaussJacobiQuadrature<T, NumNodes1D, ElementType>(
                ref_element, 0.0, 0.0, max_newton_itersations, min_newton_iterations) {}
    };

    /**
     * @brief Gauss-Chebyshev quadrature (Jacobi with alpha = beta = -1/2).
     *
     * @tparam T floating point number type of the quadrature nodes and weights
     * @tparam NumNodes1D number of quadrature nodes for one dimension
     * @tparam ElementType element type for which the quadrature rule is defined
     */
    template <typename T, unsigned NumNodes1D, typename ElementType>
    class GaussChebyshevQuadrature : public GaussJacobiQuadrature<T, NumNodes1D, ElementType> {
    public:
        /**
         * @brief Construct a Gauss-Chebyshev quadrature rule on [-1, 1].
         *
         * @param ref_element reference element to compute the quadrature nodes on
         * @param max_newton_itersations maximum Newton iterations (Nodes1D Newton backend only)
         * @param min_newton_iterations minimum Newton iterations (Nodes1D Newton backend only)
         */
        GaussChebyshevQuadrature(const ElementType& ref_element,
                                 const size_t& max_newton_itersations = 10,
                                 const size_t& min_newton_iterations  = 1)
            : GaussJacobiQuadrature<T, NumNodes1D, ElementType>(
                ref_element, -0.5, -0.5, max_newton_itersations, min_newton_iterations) {}
    };

    template <typename T, unsigned NumNodes1D, typename ElementType>
    GaussJacobiQuadrature<T, NumNodes1D, ElementType>::GaussJacobiQuadrature(
        const ElementType& ref_element, const T& alpha, const T& beta,
        const size_t& max_newton_iterations, const size_t& min_newton_iterations)
        : Quadrature<T, NumNodes1D, ElementType>(ref_element)
        , alpha_m(alpha)
        , beta_m(beta)
        , max_newton_iterations_m(max_newton_iterations)
        , min_newton_iterations_m(min_newton_iterations) {
        assert(alpha > -1.0 && "alpha >= -1.0 is not satisfied");
        assert(beta > -1.0 && "beta >= -1.0 is not satisfied");
        assert(max_newton_iterations >= 1 && "max_newton_iterations >= 1 is not satisfied");
        assert(min_newton_iterations_m >= 1 && "min_newton_iterations_m >= 1 is not satisfied");
        assert(min_newton_iterations_m <= max_newton_iterations_m
               && "min_newton_iterations_m <= max_newton_iterations_m is not satisfied");

        this->degree_m = 2 * NumNodes1D - 1;

        this->a_m = -1.0;
        this->b_m = 1.0;

        this->integration_nodes_m = Vector<T, NumNodes1D>();
        this->weights_m           = Vector<T, NumNodes1D>();

        this->computeNodesAndWeights();
    }

}  // namespace ippl

#endif
