// Class MidpointQuadrature
//  This is a class representing a midpoint quadrature rule.

#ifndef IPPL_MIDPOINTQUADRATURE_H
#define IPPL_MIDPOINTQUADRATURE_H

#include "FEM/Quadrature/Quadrature.h"

namespace ippl {

    /**
     * @brief This is a class representing the midpoint quadrature rule.
     *
     * @tparam T floating point number type of the quadrature nodes and weights
     * @tparam NumNodes1D number of quadrature nodes for one dimension
     * @tparam ElementType element type for which the quadrature rule is defined
     */
    template <typename T, unsigned NumNodes1D, typename ElementType>
    class MidpointQuadrature : public Quadrature<T, NumNodes1D, ElementType> {
    public:
        /**
         * @brief Construct a new Midpoint Quadrature object
         *
         * @param ref_element reference element to compute the quadrature nodes on
         */
        MidpointQuadrature(const ElementType& ref_element);

        /**
         * @brief Computes the quadrature nodes and weights.
         */
        void computeNodesAndWeights() override;
    };

}  // namespace ippl

#include "FEM/Quadrature/MidpointQuadrature.hpp"

#endif
