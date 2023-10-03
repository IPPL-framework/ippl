// Class Quadrature
//

#ifndef IPPL_QUADRATURE_H
#define IPPL_QUADRATURE_H

#include "Types/Vector.h"

namespace ippl {

    template <typename T, unsigned NumNodes>
    class Quadrature {
    public:
        Quadrature();

        /**
         * @brief Returns the order of the quadrature rule. (order = degree + 1)
         *
         * @return unsigned
         */
        virtual unsigned getOrder() const;

        /**
         * @brief Returns the degree of exactness of the quadrature rule. (degree = order - 1)
         *
         * @return unsigned
         */
        virtual unsigned getDegree() const;

        /**
         * @brief Get the nodes for the quadrature rule scaled to the interval [a, b].
         *
         * @param a Start of the interval [a, b]
         * @param b End of the interval [a, b]
         * @return std::vector<Vector<T, Dim>> Returns a vector of nodes.)
         */
        virtual Vector<T, NumNodes> getNodes(const T& a, const T& b) const = 0;

        /**
         * @brief Get the weights object
         *
         * @return std::vector<T>
         */
        virtual Vector<T, NumNodes> getWeights() const = 0;
    };

}  // namespace ippl

#include "FEM/Quadrature/Quadrature.hpp"

#endif