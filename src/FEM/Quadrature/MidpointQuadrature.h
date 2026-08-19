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
         * @brief Construct a midpoint quadrature rule on [0, 1].
         *
         * @param ref_element reference element to compute the quadrature nodes on
         */
        MidpointQuadrature(const ElementType& ref_element)
            : Quadrature<T, NumNodes1D, ElementType>(ref_element, 1, T(0), T(1)) {
            computeNodesAndWeights();
        }

        /** @brief Fill equally spaced midpoints and uniform weights on [a_m, b_m]. */
        void computeNodesAndWeights() override {
            const T segment_length = (this->b_m - this->a_m) / NumNodes1D;

            this->weights_m = Vector<T, NumNodes1D>(segment_length);

            this->integration_nodes_m = Vector<T, NumNodes1D>();
            for (unsigned i = 0; i < NumNodes1D; ++i) {
                this->integration_nodes_m[i] =
                    0.5 * segment_length + i * segment_length + this->a_m;
            }
        }
    };

}  // namespace ippl

#endif
