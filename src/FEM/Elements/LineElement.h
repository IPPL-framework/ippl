// Class LineElement
//   The LineElement class. This is a class representing a line element
//   for finite element methods.

#ifndef IPPL_LINEELEMENT_H
#define IPPL_LINEELEMENT_H

#include "FEM/Elements/Element.h"

namespace ippl {

    template <unsigned NumVertices = 2>
    class LineElement : public Element<1, NumVertices> {
    public:
        LineElement(std::size_t global_index,
                    Vector<std::size_t, NumVertices> global_indices_of_vertices);

        template <unsigned Order, unsigned NumNodes = Order + 1>
        virtual const Vector<Vector<T, Dim>, NumNodes>& getGlobalNodes() const override;
    };

}  // namespace ippl

#include "FEM/Elements/LineElement.hpp"

#endif