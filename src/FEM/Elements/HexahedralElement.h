//
// Class HexahedralElement
//   The HexahedralElement class. This is a class representing a hexahedron element
//   for finite element methods.
#ifndef IPPL_HEXAHEDRALELEMENT_H
#define IPPL_HEXAHEDRALELEMENT_H

#include "FEM/Elements/Element.h"

namespace ippl {

    template <typename T, unsigned GeometricDim>
    class HexahedralElement : public Element3D<T, GeometricDim, 8> {
    public:
    };

}  // namespace ippl

#include "FEM/Elements/HexahedralElement.hpp"

#endif