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
        typedef typename Element2D<T, GeometricDim, NumVertices>::jacobian_t jacobian_t;
        typedef
            typename Element2D<T, GeometricDim, NumVertices>::inverse_jacobian_t inverse_jacobian_t;

        local_vertex_vector getLocalVertices() const override;

        /**
         * @brief Returns the transformation matrix without the translation
         * from the global coordinate system to the local element coordinate system.
         *
         * @param global_vertices the vertices of the element in the global coordinate system.
         * @return jacobian_t
         */
        jacobian_t getLinearTransformationJacobian(
            const global_vertex_vector& global_vertices) const;

        /**
         * @brief Returns the transformation matrix without the translation
         * from the local element coordinate system to the global coordinate system.
         *
         * @param global_vertices the vertices of the element in the global coordinate system.
         * @return inverse_jacobian_t
         */
        inverse_jacobian_t getInverseLinearTransformationJacobian(
            const global_vertex_vector& global_vertices) const;
    };
}  // namespace ippl

#include "FEM/Elements/QuadrilateralElement.hpp"

#endif