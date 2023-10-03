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
        QuadrilateralElement(std::size_t global_index,
                             Vector<std::size_t, NumVertices> global_indices_of_vertices);

        template <unsigned Order, unsigned NumNodes = (Order + 1) * (Order + 1)>
        virtual const Vector<Vector<T, Dim>, NumNodes>& getGlobalNodes() const override;
    };

}  // namespace ippl

#include "FEM/Elements/QuadrilateralElement.hpp"

#endif