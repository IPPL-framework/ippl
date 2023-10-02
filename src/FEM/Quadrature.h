//
//
#ifndef IPPL_QUADRATURE_H
#define IPPL_QUADRATURE_H

#include "Types/Vector.h"

namespace ippl {

    template <typename T, unsigned Dim, unsigned NumNodes>
    class Quadrature {
    public:
        Quadrature();

        virtual unsigned getOrder() const;

        virtual void setOrder();

        /**
         * @brief Get the nodes for the quadrature rule scaled to the interval [a, b].
         *
         * @param a Start of the interval [a, b]
         * @param b End of the interval [a, b]
         * @return std::vector<Vector<T, Dim>> Returns a vector of nodes.)
         */
        virtual Vector<Vector<T, Dim>, NumNodes> getNodes(const T& a, const T& b) const = 0;

        /**
         * @brief Get the weights object
         *
         * @return std::vector<T>
         */
        virtual Vector<T, NumNodes> getWeights() const = 0;
    };

}  // namespace ippl

#endif