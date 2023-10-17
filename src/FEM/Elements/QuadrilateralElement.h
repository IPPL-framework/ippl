// Class QuadriateralElement
//   The QuadrilateralElement class. This is a class representing a quadrilateral element
//   for finite element methods.

#ifndef IPPL_QUADRILATERALELEMENT_H
#define IPPL_QUADRILATERALELEMENT_H

#include "FEM/Elements/Element.h"

namespace ippl {

    template <typename T, unsigned GeometricDim>
    class QuadrilateralElement : public Element2D<T, GeometricDim, 4> {
    public:
        static constexpr unsigned NumVertices    = 4;
        static constexpr unsigned TopologicalDim = 2;

        typedef typename Element2D<T, GeometricDim, NumVertices>::local_vertex_vector
            local_vertex_vector;
        typedef typename Element2D<T, GeometricDim, NumVertices>::global_vertex_vector
            global_vertex_vector;
        typedef typename Element2D<T, GeometricDim, NumVertices>::jacobian_type jacobian_type;

        local_vertex_vector getLocalVertices() const override;
    };
}  // namespace ippl

#include "FEM/Elements/QuadrilateralElement.hpp"

#endif