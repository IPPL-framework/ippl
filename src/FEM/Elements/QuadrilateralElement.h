// Class QuadriateralElement
//   The QuadrilateralElement class. This is a class representing a quadrilateral element
//   for finite element methods.

#ifndef IPPL_QUADRILATERALELEMENT_H
#define IPPL_QUADRILATERALELEMENT_H

#include "FEM/Elements/Element.h"

namespace ippl {

    template <typename T, unsgined GeometricDim, unsigned NumVertices = 4>
    class QuadrilateralElement : public Element2D<T, GeometricDim, NumVertices> {
    public:
    };

}  // namespace ippl

#include "FEM/Elements/QuadrilateralElement.hpp"

#endif