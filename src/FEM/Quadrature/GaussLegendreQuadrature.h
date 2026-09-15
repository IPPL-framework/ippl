// Class GaussLegendreQuadrature
//   Gauss-Legendre quadrature on [-1, 1]. Node/weight generation delegates to ippl::nodes1d.
//
// HISTORY (2026-08-17): This file used to be GaussJacobiQuadrature.h and hosted the whole
// Gauss-Jacobi hierarchy: the base GaussJacobiQuadrature, plus GaussLegendreQuadrature and
// GaussChebyshevQuadrature as (alpha,beta) specializations.
//
// The general Gauss-Jacobi rule approximates the WEIGHTED integral
//     \int_{-1}^{1} f(x) (1-x)^alpha (1+x)^beta dx   (weights already fold in the weight fn),
// so it only reduces to the plain, unweighted integral \int_{-1}^{1} f(x) dx when
// alpha = beta = 0 (Gauss-Legendre). FEM assembly (LagrangeSpace/NedelecSpace stiffness and
// load vectors) multiplies these quadrature weights by the Jacobian determinant assuming an
// unweighted rule, so feeding it any alpha,beta != 0 (in particular the Gauss-Chebyshev case
// alpha = beta = -1/2, whose weights sum to pi instead of 2) silently integrates against the
// wrong measure with no compile-time or runtime error.
//
// Because only alpha = beta = 0 is valid here and nothing consumed the weighted variants,
// GaussJacobiQuadrature and GaussChebyshevQuadrature are commented out below (kept for
// reference), and Gauss-Legendre is now provided as a standalone class that inherits directly
// from Quadrature (mirroring GaussLobattoQuadrature) rather than through a weighted base class.
// The underlying node math still lives in ippl::nodes1d (computeGaussJacobi / computeGaussChebyshev)
// for anyone who genuinely needs a weighted rule.

#ifndef IPPL_GAUSSLEGENDREQUADRATURE_H
#define IPPL_GAUSSLEGENDREQUADRATURE_H

#include "FEM/Quadrature/Quadrature.h"
#include "Nodes1D/GaussLegendre1D.h"

namespace ippl {

    /**
     * @brief Gauss-Legendre quadrature (Gauss-Jacobi with alpha = beta = 0) on a reference element.
     *
     * Integrates the UNWEIGHTED integral \f$\int_{-1}^{1} f(x)\,dx\f$ (weight function 1); the
     * weights sum to 2. This is the rule FEM assembly assumes, which is why it is a standalone,
     * directly instantiable class rather than a specialization of a weighted Jacobi base.
     * An n-point rule is exact for polynomials of degree up to 2n-1.
     *
     * Node/weight generation delegates to ippl::nodes1d::computeGaussLegendre.
     *
     * @tparam T floating point number type of the quadrature nodes and weights
     * @tparam NumNodes1D number of quadrature nodes for one dimension (>= 1)
     * @tparam ElementType element type for which the quadrature rule is defined
     */
    template <typename T, unsigned NumNodes1D, typename ElementType>
    class GaussLegendreQuadrature : public Quadrature<T, NumNodes1D, ElementType> {
    public:
        static_assert(NumNodes1D >= 1, "Gauss-Legendre quadrature requires NumNodes1D >= 1");

        /**
         * @brief Construct a Gauss-Legendre quadrature rule on [-1, 1].
         *
         * Node/weight generation uses the Nodes1D default backend
         * (RootFinderMethod::GolubWelsch). Callers that need a different root-finding backend
         * should use ippl::nodes1d::computeGaussLegendre directly.
         *
         * @param ref_element reference element to compute the quadrature nodes on
         */
        GaussLegendreQuadrature(const ElementType& ref_element)
            : Quadrature<T, NumNodes1D, ElementType>(ref_element, 2 * NumNodes1D - 1, T(-1), T(1)) {
            computeNodesAndWeights();
        }

        /** @brief Fill integration_nodes_m and weights_m via nodes1d::computeGaussLegendre. */
        void computeNodesAndWeights() override {
            nodes1d::computeGaussLegendre(this->integration_nodes_m, this->weights_m);
        }
    };

    // =============================================================================================
    // COMMENTED OUT (2026-08-17) — WEIGHTED Gauss-Jacobi rules, unsuitable as a drop-in FEM
    // quadrature. Gauss-Jacobi integrates \int f(x) (1-x)^alpha (1+x)^beta dx; only alpha=beta=0
    // (Gauss-Legendre, above) is the unweighted rule FEM assembly expects. Same for GaussChebyshevQuadrature
    // (alpha=beta=-1/2). Kept here for reference; the node math remains available via ippl::nodes1d.
    // ---------------------------------------------------------------------------------------------
    //
    // /**
    //  * @brief This is class represents the Gauss-Jacobi quadrature rule
    //  * on a reference element.
    //  *
    //  * @tparam T floating point number type of the quadrature nodes and weights
    //  * @tparam NumNodes1D number of quadrature nodes for one dimension
    //  * @tparam ElementType element type for which the quadrature rule is defined
    //  */
    // template <typename T, unsigned NumNodes1D, typename ElementType>
    // class GaussJacobiQuadrature : public Quadrature<T, NumNodes1D, ElementType> {
    // public:
    //     /**
    //      * @brief Construct a Gauss-Jacobi quadrature rule on [-1, 1].
    //      *
    //      * @param ref_element reference element to compute the quadrature nodes on
    //      * @param alpha first Jacobi parameter alpha (> -1)
    //      * @param beta second Jacobi parameter beta (> -1)
    //      * @param max_newton_iterations maximum Newton iterations (Nodes1D Newton backend only)
    //      * @param min_newton_iterations minimum Newton iterations (Nodes1D Newton backend only)
    //      */
    //     GaussJacobiQuadrature(const ElementType& ref_element, const T& alpha, const T& beta,
    //                           const size_t& max_newton_iterations = 10,
    //                           const size_t& min_newton_iterations  = 1);
    //
    //     /** @brief Fill integration_nodes_m and weights_m via nodes1d::computeGaussJacobi. */
    //     void computeNodesAndWeights() override {
    //         nodes1d::computeGaussJacobi(this->integration_nodes_m, this->weights_m, alpha_m, beta_m,
    //                                     max_newton_iterations_m, min_newton_iterations_m);
    //     }
    //
    // private:
    //     const T alpha_m;
    //     const T beta_m;
    //
    //     const size_t max_newton_iterations_m;
    //     const size_t min_newton_iterations_m;
    // };
    //
    // /**
    //  * @brief Gauss-Legendre quadrature (Jacobi with alpha = beta = 0).
    //  *
    //  * OLD VERSION: inherited from the weighted GaussJacobiQuadrature base. Superseded by the
    //  * standalone GaussLegendreQuadrature above, which inherits directly from Quadrature.
    //  *
    //  * @tparam T floating point number type of the quadrature nodes and weights
    //  * @tparam NumNodes1D number of quadrature nodes for one dimension
    //  * @tparam ElementType element type for which the quadrature rule is defined
    //  */
    // template <typename T, unsigned NumNodes1D, typename ElementType>
    // class GaussLegendreQuadrature : public GaussJacobiQuadrature<T, NumNodes1D, ElementType> {
    // public:
    //     /**
    //      * @brief Gauss-Legendre quadrature (Jacobi with alpha = beta = 0).
    //      *
    //      * @param ref_element reference element to compute the quadrature nodes on
    //      * @param max_newton_iterations maximum Newton iterations (Nodes1D Newton backend only)
    //      * @param min_newton_iterations minimum Newton iterations (Nodes1D Newton backend only)
    //      */
    //     GaussLegendreQuadrature(const ElementType& ref_element,
    //                             const size_t& max_newton_iterations = 10,
    //                             const size_t& min_newton_iterations  = 1)
    //         : GaussJacobiQuadrature<T, NumNodes1D, ElementType>(
    //             ref_element, 0.0, 0.0, max_newton_iterations, min_newton_iterations) {}
    // };
    //
    // /**
    //  * @brief Gauss-Chebyshev quadrature (Jacobi with alpha = beta = -1/2).
    //  *
    //  * WEIGHTED rule: approximates \int f(x) (1-x^2)^(-1/2) dx, weights sum to pi. NOT a valid
    //  * drop-in for unweighted FEM assembly (see the file header). Kept for reference only.
    //  *
    //  * @tparam T floating point number type of the quadrature nodes and weights
    //  * @tparam NumNodes1D number of quadrature nodes for one dimension
    //  * @tparam ElementType element type for which the quadrature rule is defined
    //  */
    // template <typename T, unsigned NumNodes1D, typename ElementType>
    // class GaussChebyshevQuadrature : public GaussJacobiQuadrature<T, NumNodes1D, ElementType> {
    // public:
    //     /**
    //      * @brief Construct a Gauss-Chebyshev quadrature rule on [-1, 1].
    //      *
    //      * @param ref_element reference element to compute the quadrature nodes on
    //      * @param max_newton_iterations maximum Newton iterations (Nodes1D Newton backend only)
    //      * @param min_newton_iterations minimum Newton iterations (Nodes1D Newton backend only)
    //      */
    //     GaussChebyshevQuadrature(const ElementType& ref_element,
    //                              const size_t& max_newton_iterations = 10,
    //                              const size_t& min_newton_iterations  = 1)
    //         : GaussJacobiQuadrature<T, NumNodes1D, ElementType>(
    //             ref_element, -0.5, -0.5, max_newton_iterations, min_newton_iterations) {}
    // };
    //
    // template <typename T, unsigned NumNodes1D, typename ElementType>
    // GaussJacobiQuadrature<T, NumNodes1D, ElementType>::GaussJacobiQuadrature(
    //     const ElementType& ref_element, const T& alpha, const T& beta,
    //     const size_t& max_newton_iterations, const size_t& min_newton_iterations)
    //     : Quadrature<T, NumNodes1D, ElementType>(ref_element)
    //     , alpha_m(alpha)
    //     , beta_m(beta)
    //     , max_newton_iterations_m(max_newton_iterations)
    //     , min_newton_iterations_m(min_newton_iterations) {
    //     assert(alpha > -1.0 && "alpha >= -1.0 is not satisfied");
    //     assert(beta > -1.0 && "beta >= -1.0 is not satisfied");
    //     assert(max_newton_iterations >= 1 && "max_newton_iterations >= 1 is not satisfied");
    //     assert(min_newton_iterations_m >= 1 && "min_newton_iterations_m >= 1 is not satisfied");
    //     assert(min_newton_iterations_m <= max_newton_iterations_m
    //            && "min_newton_iterations_m <= max_newton_iterations_m is not satisfied");
    //
    //     this->degree_m = 2 * NumNodes1D - 1;
    //
    //     this->a_m = -1.0;
    //     this->b_m = 1.0;
    //
    //     this->integration_nodes_m = Vector<T, NumNodes1D>();
    //     this->weights_m           = Vector<T, NumNodes1D>();
    //
    //     this->computeNodesAndWeights();
    // }
    // =============================================================================================

}  // namespace ippl

#endif
