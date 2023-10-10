// Class Quadrature
//

#ifndef IPPL_QUADRATURE_H
#define IPPL_QUADRATURE_H

#include "Types/Vector.h"

#include "FEM/Singleton.h"

namespace ippl {

    template <typename T, unsigned NumNodes>
    class Quadrature {
    public:
        /**
         * @brief Returns the order of the quadrature rule. (order = degree + 1)
         *
         * @return unsigned - order
         */
        virtual unsigned getOrder() const = 0;

        /**
         * @brief Returns the degree of exactness of the quadrature rule. (degree = order - 1)
         *
         * @return unsigned - degree
         */
        virtual unsigned getDegree() const = 0;

        /**
         * @brief Get the number Of nodes for this quadrature rule.
         *
         * @return unsigned
         */
        unsigned getNumberOfIntegrationPoints() const;

        /**
         * @brief Get the nodes for the quadrature rule scaled to the interval [a, b].
         *
         * @param a Start of the interval [a, b]
         * @param b End of the interval [a, b]
         * @return std::vector<Vector<T, Dim>> Returns a vector of nodes.)
         */
        virtual Vector<T, NumNodes> getIntegrationNodes(const T& a, const T& b) const = 0;

        /**
         * @brief Get the weights object
         *
         * @tparam NumNodes Number of nodes in the quadrature rule.
         * @return std::vector<T, NumNodes>
         */
        virtual Vector<T, NumNodes> getWeights() const = 0;

    protected:
        // the degree of exactness of the quadrature rule
        unsigned degree_m;
    };

}  // namespace ippl

#include "FEM/Quadrature/Quadrature.hpp"

#endif