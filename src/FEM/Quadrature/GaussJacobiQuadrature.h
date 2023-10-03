// Class GaussJacobiQuadrature
//   See: https://craffael.github.io/lehrfempp/gauss__quadrature_8cc_source.html

#ifndef IPPL_GAUSSJACOBIQUADRATURE_H
#define IPPL_GAUSSJACOBIQUADRATURE_H

#include "FEM/Quadrature/Quadrature.h"

namespace ippl {

    template <typename T, unsigned Order>
    class GaussJacobiQuadrature : public Quadrature {
    public:
        typedef NumNodes = Order / 2;  // TODO fix possible bugs

        /**
         * @brief Construct a new Gauss Jacobi Quadrature object
         * https://en.wikipedia.org/wiki/Gauss%E2%80%93Jacobi_quadrature
         *
         * For alpha = beta = 0.0. The quadrature rule is equivalent to the Gauss-Legendre
         * quadrature rule. For alpha = beta = -0.5 or alpha = beta = 0.5, the quadrature rule is
         * equivalent to the Gauss-Chebyshev quadrature rule.
         *
         * @param number_of_points Number of points in the quadrature rule
         * @param alpha Default value is 0.0
         * @param beta Default value is 0.0
         */
        GaussJacobiQuadrature(const T& alpha = 0.0, const T& beta = 0.0);

        /**
         * @brief Get the Nodes for the quadrature
         *
         * @param a
         * @param b
         * @return std::vector<Vector<T, Dim>> - Returns a vector with number_of_points many nodes.
         */
        Vector<Vector<T, Dim>, NumNodes> getNodes(const T& a = -1.0,
                                                  const T& b = 1.0) const override;

        /**
         * @brief Get the weights for the quadrature
         *
         * @return std::vector<T> - Returns a vector with number_of_points many weights.
         */
        Vector<T, NumNodes> getWeights() const override;

        /**
         * @brief Return the number of points for the quadrature rule.
         *
         * @return unsigned - Return the number of points
         */
        unsigned getNumberOfPoints() const;

        /**
         * @brief Gets the order of the Gauss-Jacobi quadrature rule.
         * @details order = 2 * number_of_points
         * @example
         * order 2: 1 point
         * order 4: 2 points
         *
         * @return unsigned - Returns the order of the quadrature rule
         */
        unsigned getOrder() const override;

    private:
        unsigned number_of_points_m = NumNodes;
        T alpha_m;
        T beta_m;
    };

}  // namespace ippl

#include "FEM/Quadrature/GaussJacobiQuadrature.hpp"

#endif