// Class QuadriateralElement
//   The QuadrilateralElement class. This is a class representing a quadrilateral element
//   for finite element methods.

#ifndef IPPL_QUADRILATERALELEMENT_H
#define IPPL_QUADRILATERALELEMENT_H

#include "FEM/Elements/Element.h"

namespace ippl {

    template <unsigned NumVertices = 4>
    class QuadrilateralElement : public Element<2, NumVertices> {
    public:
    };

}  // namespace ippl

#include "FEM/Elements/QuadrilateralElement.hpp"

#endif