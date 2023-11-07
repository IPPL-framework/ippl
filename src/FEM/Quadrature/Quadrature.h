// Class Quadrature
//

#ifndef IPPL_QUADRATURE_H
#define IPPL_QUADRATURE_H

#include <cmath>

#include "Types/Vector.h"

#include "FEM/Elements/Element.h"

inline constexpr unsigned numElementNodes(unsigned NumNodes, unsigned Dim) {
    return static_cast<unsigned>(pow(static_cast<double>(NumNodes), static_cast<double>(Dim)));
}

namespace ippl {

    template <typename T, unsigned NumNodes, unsigned Dim, unsigned NumElementVertices>
    class Quadrature {
    public:
        Quadrature(const Element<T, Dim, NumElementVertices>& ref_element);

        /**
         * @brief Returns the order of the quadrature rule. (order = degree + 1)
         *
         * @return unsigned - order
         */
        virtual std::size_t getOrder() const;

        virtual std::size_t getDegree() const = 0;

        /**
         * @brief Get the number of nodes for this quadrature rule in 1D.
         *
         * @return unsigned
         */
        std::size_t num1DIntegrationPoints() const;

        /**
         * @brief Get the number Of nodes for this quadrature rule on the reference element.
         *
         * @return unsigned
         */
        std::size_t numElementIntegrationPoints() const;

        /**
         * @brief Get the quadrature weights for the reference element.
         *
         * @return Vector<T, NumNodes>
         */
        Vector<T, numElementNodes(NumNodes, Dim)> getWeightsForRefElement() const;

        /**
         * @brief Get the integration nodes for the reference element.
         *
         * @return Vector<Vector<T, Dim>, NumNodes>
         */
        Vector<Vector<T, Dim>, numElementNodes(NumNodes, Dim)> getIntegrationNodesForRefElement()
            const;

    protected:
        virtual Vector<T, NumNodes> getIntegrationNodes() const = 0;

        virtual Vector<T, NumNodes> getWeights() const = 0;

    private:
        const Element<T, Dim, NumElementVertices>& ref_element_m;
    };

}  // namespace ippl

#include "FEM/Quadrature/Quadrature.hpp"

#endif