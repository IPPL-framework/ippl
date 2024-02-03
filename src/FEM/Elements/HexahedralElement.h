//
// Class HexahedralElement
//   The HexahedralElement class. This is a class representing a hexahedron element
//   for finite element methods.
#ifndef IPPL_HEXAHEDRALELEMENT_H
#define IPPL_HEXAHEDRALELEMENT_H

#include "FEM/Elements/Element.h"

namespace ippl {

    template <typename T>
    class HexahedralElement : public Element3D<T, 8> {
    public:
        static constexpr unsigned NumVertices = 8;

        typedef typename Element3D<T, NumVertices>::point_t point_t;
        typedef typename Element3D<T, NumVertices>::mesh_element_vertex_point_vec_t
            mesh_element_vertex_point_vec_t;
        typedef typename Element3D<T, NumVertices>::diag_matrix_vec_t diag_matrix_vec_t;

        /**
         * @brief Returns the coordinates of the vertices of the reference element.
         *
         * @return mesh_element_vertex_point_vec_t (Vector<Vector<T, 3>, 8>)
         */
        mesh_element_vertex_point_vec_t getLocalVertices() const override;

        /**
         * @brief Returns the Jacobian of the transformation matrix.
         *
         * @param global_vertices A vector of the vertex coordinates of the global element to
         * transform to.
         *
         * @return diag_matrix_vec_t (Vector<T, 3>) - A vector representing the diagonal elements of
         * the Jacobian matrix
         */
        diag_matrix_vec_t getTransformationJacobian(
            const mesh_element_vertex_point_vec_t& global_vertices) const override;

        /**
         * @brief Returns the inverse of the Jacobian of the transformation matrix.
         *
         * @param global_vertices A vector of the vertex coordinates of the global element to
         * transform to.
         *
         * @return diag_matrix_vec_t (Vector<T, 3>) - A vector representing the diagonal elements of
         *  the inverse Jacobian matrix
         */
        diag_matrix_vec_t getInverseTransformationJacobian(
            const mesh_element_vertex_point_vec_t& global_vertices) const override;
    };

}  // namespace ippl

#include "FEM/Elements/HexahedralElement.hpp"

#endif