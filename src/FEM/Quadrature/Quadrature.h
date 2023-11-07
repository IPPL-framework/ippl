// Class Quadrature
//

#ifndef IPPL_QUADRATURE_H
#define IPPL_QUADRATURE_H

#include <cmath>
#include <type_traits>

#include "Types/Vector.h"

#include "FEM/Elements/Element.h"

inline constexpr unsigned getNumElementDOFs(unsigned NumNodes1D, unsigned Dim) {
    return static_cast<unsigned>(pow(static_cast<double>(NumNodes1D), static_cast<double>(Dim)));
}

namespace ippl {

    // template <typename T>
    //  concept IsElement = std::is_base_of<Element, T>::value;

    template <typename T, unsigned NumNodes1D, typename ElementType>
    // requires IsElement<ElementType>
    class Quadrature {
    public:
        static constexpr unsigned numNodes1D     = NumNodes1D;
        static constexpr unsigned dim            = ElementType::dim;
        static constexpr unsigned numElementDOFs = getNumElementDOFs(NumNodes1D, ElementType::dim);

        Quadrature(const ElementType& ref_element);

        /**
         * @bridef Returns the order of the quadrature rule. (order = degree + 1)
         *
         * @return unsigned - order
         */
        virtual std::size_t getOrder() const;

        virtual std::size_t getDegree() const = 0;

        /**
         * @brief Get the quadrature weights for the reference element.
         *
         * @return Vector<T, NumNodes1D>
         */
        Vector<T, numElementDOFs> getWeightsForRefElement() const;

        /**
         * @brief Get the integration nodes for the reference element.
         *
         * @return Vector<Vector<T, Dim>, NumNodes1D>
         */
        Vector<Vector<T, dim>, numElementDOFs> getIntegrationNodesForRefElement() const;

    protected:
        virtual Vector<T, NumNodes1D> getIntegrationNodes() const = 0;

        virtual Vector<T, NumNodes1D> getWeights() const = 0;

    private:
        const ElementType& ref_element_m;
    };

}  // namespace ippl

#include "FEM/Quadrature/Quadrature.hpp"

#endif