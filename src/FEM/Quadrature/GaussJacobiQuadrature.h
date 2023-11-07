// Class GaussJacobiQuadrature
//   See: https://craffael.github.io/lehrfempp/gauss__quadrature_8cc_source.html

#ifndef IPPL_GAUSSJACOBIQUADRATURE_H
#define IPPL_GAUSSJACOBIQUADRATURE_H

#include "FEM/Quadrature/Quadrature.h"

namespace ippl {

    template <typename T>
    class GaussJacobiQuadrature : public Quadrature<T> {
    public:
        /**
         * @brief Construct a new Gauss Jacobi Quadrature object
         * https://en.wikipedia.org/wiki/Gauss%E2%80%93Jacobi_quadrature
         *
         * For alpha = beta = 0.0. The quadrature rule is equivalent to the Gauss-Legendre
         * quadrature rule. For alpha = beta = -0.5 or alpha = beta = 0.5, the quadrature rule is
         * equivalent to the Gauss-Chebyshev quadrature rule.
         *
         * @param degree Polynomial degree of exactness
         * @param alpha
         * @param beta
         */
        GaussJacobiQuadrature(const unsigned& degree, const T& alpha, const T& beta);

        /**
         * @brief Return the number of points for the Gauss-Jacobi quadrature rule.
         *
         * @return unsigned - Return the number of points
         */
        unsigned getNumberOfIntegrationPoints() const override;

        /**
         * @brief Get the Nodes for the quadrature
         *
         * @param a
         * @param b
         * @tparam NumNodes1D Number of nodes in the quadrature rule.
         * @return std::vector<Vector<T, Dim>> - Returns a vector with number_of_points many nodes.
         */
        template <unsigned NumNodes1D>
        Vector<T, NumNodes1D> getIntegrationNodes(const T& a = -1.0, const T& b = 1.0) const override;

        /**
         * @brief Get the weights for the quadrature
         * @tparam NumNodes1D Number of nodes in the quadrature rule.
         *
         * @return std::vector<T> - Returns a vector with number_of_points many weights.
         */
        template <unsigned NumNodes1D>
        Vector<T, NumNodes1D> getWeights() const override;

    private:
        T alpha_m;
        T beta_m;
    };

    template <typename T>
    class GaussLegendreQuadrature : public GaussJacobiQuadrature<T> {
    public:
        GaussLegendreQuadrature()
            : GaussJacobiQuadrature(0.0, 0.0) {}
    };

    template <typename T>
    class ChebyshevGaussQuadrature : public GaussJacobiQuadrature<T> {
    public:
        ChebyshevGaussQuadrature()
            : GaussJacobiQuadrature(-0.5, -0.5) {}
    };

}  // namespace ippl

#include "FEM/Quadrature/GaussJacobiQuadrature.hpp"

#endif