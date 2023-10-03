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
        HexahedralElement(std::size_t global_index,
                          Vector<std::size_t, NumVertices> global_indices_of_vertices);

        template <unsigned Order, unsigned NumNodes = (Order + 1) * (Order + 1) * (Order + 1)>
        virtual const Vector<Vector<T, Dim>, NumNodes>& getGlobalNodes() const override;
    };

}  // namespace ippl

#include "FEM/Elements/HexahedralElement.hpp"

#endif