// Class Midpoint Quadrature

#ifndef IPPL_MIDPOINTQUADRATURE_H
#define IPPL_MIDPOINTQUADRATURE_H

#include "FEM/Quadrature/Quadrature.h"

namespace ippl {

    template <typename T>
    class MidpointQuadrature : public Quadrature<T, 1> {
    public:
        MidpointQuadrature();

        virtual Vector<T, 1> getIntegrationNodes(const T& a = 0.0, const T& b = 1.0) const override;

        virtual Vector<T, 1> getWeights() const override;
    };

}  // namespace ippl

#include "FEM/Quadrature/MidpointQuadrature.hpp"

#endif