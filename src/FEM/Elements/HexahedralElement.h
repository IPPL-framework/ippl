//
// Class HexahedralElement
//   The HexahedralElement class. This is a class representing a hexahedron element
//   for finite element methods.
#ifndef IPPL_HEXAHEDRALELEMENT_H
#define IPPL_HEXAHEDRALELEMENT_H

#include "FEM/Elements/Element.h"

namespace ippl {

    template <unsigned NumVertices = 8>
    class HexahedralElement : public Element<3, NumVertices> {
    public:
    };

}  // namespace ippl

#include "FEM/Elements/HexahedralElement.hpp"

#endif