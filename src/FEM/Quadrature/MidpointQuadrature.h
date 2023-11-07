// Class Midpoint Quadrature

#ifndef IPPL_MIDPOINTQUADRATURE_H
#define IPPL_MIDPOINTQUADRATURE_H

#include "FEM/Quadrature/Quadrature.h"

namespace ippl {

    template <typename T, unsigned NumNodes1D, typename ElementType>
    class MidpointQuadrature : public Quadrature<T, NumNodes1D, ElementType> {
    public:
        MidpointQuadrature(const ElementType& ref_element);

        /**
         * @brief Get the degree of exactness of the quadrature rule.
         *
         * @return unsigned - Degree of exactness
         */
        unsigned getDegree() const override;

    protected:
        Vector<T, NumNodes1D> getIntegrationNodes() const override;

        Vector<T, NumNodes1D> getWeights() const override;
    };

}  // namespace ippl

#include "FEM/Quadrature/MidpointQuadrature.hpp"

#endif
