// Class EdgeElement
//   The EdgeElement class. This is a class representing an edge element
//   for finite element methods.

#ifndef IPPL_EDGEELEMENT_H
#define IPPL_EDGEELEMENT_H

#include "FEM/Elements/Element.h"

namespace ippl {

    template <typename T, unsigned GeometricDim>
    class EdgeElement : public Element1D<T, GeometricDim, 2> {
    public:
        static constexpr unsigned NumVertices = 2;
        static constexpr unsigned Dim         = 1;

        typedef typename Element1D<T, GeometricDim, NumVertices>::local_vertex_vector
            local_vertex_vector;
        typedef typename Element1D<T, GeometricDim, NumVertices>::global_vertex_vector
            global_vertex_vector;
        typedef typename Element1D<T, GeometricDim, NumVertices>::jacobian_type jacobian_type;

        local_vertex_vector getLocalVertices() const override;

        jacobian_type getTransformationJacobian(
            const global_vertex_vector& global_vertices) const override;

        global_vertex_vector getGlobalNodes(
            const jacobian_type& transformation_jacobian) const override;
    };

}  // namespace ippl

#include "FEM/Elements/EdgeElement.hpp"

#endif