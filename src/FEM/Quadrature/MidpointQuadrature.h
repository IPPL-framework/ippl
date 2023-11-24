// Class Midpoint Quadrature

#ifndef IPPL_MIDPOINTQUADRATURE_H
#define IPPL_MIDPOINTQUADRATURE_H

#include "FEM/Quadrature/Quadrature.h"

namespace ippl {

    template <typename T, unsigned NumNodes1D, typename ElementType>
    class MidpointQuadrature : public Quadrature<T, NumNodes1D, ElementType> {
    public:
        MidpointQuadrature(const ElementType& ref_element);

        void computeNodesAndWeights() override;
    };

}  // namespace ippl

#include "FEM/Quadrature/MidpointQuadrature.hpp"

#endif
