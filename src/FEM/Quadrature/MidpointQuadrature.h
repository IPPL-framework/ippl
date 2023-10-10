// Class Midpoint Quadrature

#ifndef IPPL_MIDPOINTQUADRATURE_H
#define IPPL_MIDPOINTQUADRATURE_H

#include "FEM/Quadrature/Quadrature.h"

namespace ippl {

    template <typename T>
    class MidpointQuadrature : public Quadrature<T> {
    public:
        /**
         * @brief Construct a new Midpoint Quadrature object
         *
         * @param num_integration_points Number of integration points
         */
        MidpointQuadrature(const unsigned& num_integration_points = 1);

        /**
         * @brief Get the number Of nodes for this quadrature rule.
         *
         * @return unsigned - Number of integration points
         */
        unsigned getNumberOfIntegrationPoints() const override;

        /**
         * @brief Get the integration nodes for the quadrature
         *
         * @param a Lower bound of the interval
         * @param b Upper bound of the interval
         * @tparam NumNodes Number of nodes in the quadrature rule.
         * @return std::vector<Vector<T, Dim>> - Returns a vector with number_of_points many nodes.
         */
        template <unsigned NumNodes>
        Vector<T, NumNodes> getIntegrationNodes(const T& a = 0.0, const T& b = 1.0) const override;

        /**
         * @brief Get the weights for the quadrature
         * @tparam NumNodes Number of nodes in the quadrature rule.
         *
         * @return std::vector<T> - Returns a vector with number_of_points many weights.
         */
        template <unsigned NumNodes>
        Vector<T, NumNodes> getWeights() const override;

        /**
         * @brief Get the degree of exactness of the quadrature rule.
         *
         * @return unsigned - Degree of exactness
         */
        unsigned getDegree() const override;

    private:
        unsigned num_integration_points_m;
    };

}  // namespace ippl

#include "FEM/Quadrature/MidpointQuadrature.hpp"

#endif
