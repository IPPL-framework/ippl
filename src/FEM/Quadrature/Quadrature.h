// Class Quadrature
//

#ifndef IPPL_QUADRATURE_H
#define IPPL_QUADRATURE_H

#include "Types/Vector.h"

#include "FEM/Singleton.h"

namespace ippl {

    template <typename T>
    class Quadrature {
    public:
        /**
         * @brief Construct a new Quadrature object with a given degree.
         *
         * @param degree Degree of exactness
         */
        Quadrature(const unsigned& degree);

        /**
         * @brief Returns the order of the quadrature rule. (order = degree + 1)
         *
         * @return unsigned - order
         */
        unsigned getOrder() const;

        /**
         * @brief Returns the degree of exactness of the quadrature rule. (degree = order - 1)
         *
         * @return unsigned - degree
         */
        virtual unsigned getDegree() const;

        /**
         * @brief Set the order of the quadrature rule.
         *
         * @param order
         */
        void setOrder(const unsigned& order);

        /**
         * @brief Set the degree of the quadrature rule.
         *
         * @param order
         */
        void setDegree(const unsigned& order);

        /**
         * @brief Get the number Of nodes for this quadrature rule.
         *
         * @return unsigned
         */
        virtual unsigned getNumberOfIntegrationPoints() const = 0;

        /**
         * @brief Get the nodes for the quadrature rule scaled to the interval [a, b].
         *
         * @param a Start of the interval [a, b]
         * @param b End of the interval [a, b]
         * @tparam NumNodes Number of nodes in the quadrature rule.
         * @return std::vector<Vector<T, Dim>> Returns a vector of nodes.)
         */
        template <unsigned NumNodes>
        virtual Vector<T, NumNodes> getIntegrationNodes(const T& a, const T& b) const = 0;

        /**
         * @brief Get the weights object
         *
         * @tparam NumNodes Number of nodes in the quadrature rule.
         * @return std::vector<T, NumNodes>
         */
        template <unsigned NumNodes>
        virtual Vector<T, NumNodes> getWeights() const = 0;

    protected:
        // the degree of exactness of the quadrature rule
        unsigned degree_m;
    };

}  // namespace ippl

#include "FEM/Quadrature/Quadrature.hpp"

#endif