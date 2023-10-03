//
// Class HexahedralElement
//   The HexahedralElement class. This is a class representing a hexahedron element
//   for finite element methods.
#ifndef IPPL_HEXAHEDRALELEMENT_H
#define IPPL_HEXAHEDRALELEMENT_H

#include "FEM/Elements/Element.h"

namespace ippl {

    class HexahedralElement : public Element<3, 8> {
    public:
        HexahedralElement(std::size_t global_index,
                          Vector<std::size_t, 8> global_indices_of_vertices);
    };

}  // namespace ippl

#include "FEM/Elements/HexahedralElement.hpp"

#endif