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
        static constexpr unsigned NumVertices    = 2;
        static constexpr unsigned TopologicalDim = 1;

        typedef typename Element1D<T, GeometricDim, NumVertices>::local_point_t local_point_t;
        typedef typename Element1D<T, GeometricDim, NumVertices>::global_point_t global_point_t;
        typedef
            typename Element1D<T, GeometricDim, NumVertices>::local_vertex_vec_t local_vertex_vec_t;
        typedef typename Element1D<T, GeometricDim, NumVertices>::global_vertex_vec_t
            global_vertex_vec_t;
        typedef typename Element1D<T, GeometricDim, NumVertices>::jacobian_t jacobian_t;
        typedef
            typename Element1D<T, GeometricDim, NumVertices>::inverse_jacobian_t inverse_jacobian_t;

        local_vertex_vec_t getLocalVertices() const override;

        /**
         * @brief Returns the transformation matrix without the translation
         * from the global coordinate system to the local element coordinate system.
         *
         * @param global_vertices the vertices of the element in the global coordinate system.
         * @return jacobian_t
         */
        jacobian_t getLinearTransformationJacobian(
            const global_vertex_vec_t& global_vertices) const;

        /**
         * @brief Returns the transformation matrix without the translation
         * from the local element coordinate system to the global coordinate system.
         *
         * @details The transformation is given by:
         * \f$\boldsymbol{x} = \mathbf{J}^{-1}_K \hat{\boldsymbol{x}} + \boldsymbol{v}_0\f$
         * where \f$\mathbf{J}^{-1}\f$ is the transformation matrix returned by this function and
         * \f$\boldsymbol{v}_0\f$ is the translation vector (given by the coordinates of the first
         * vertex of the element).
         *
         * @param global_vertices the vertices of the element in the global coordinate system.
         * @return inverse_jacobian_t
         */
        inverse_jacobian_t getInverseLinearTransformationJacobian(
            const global_vertex_vec_t& global_vertices) const;
    };

}  // namespace ippl

#include "FEM/Elements/EdgeElement.hpp"

#endif