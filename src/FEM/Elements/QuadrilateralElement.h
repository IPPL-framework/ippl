// Class QuadriateralElement
//   The QuadrilateralElement class. This is a class representing a quadrilateral element
//   for finite element methods.

#ifndef IPPL_QUADRILATERALELEMENT_H
#define IPPL_QUADRILATERALELEMENT_H

#include "FEM/Elements/Element.h"

namespace ippl {

    template <typename T>
    class QuadrilateralElement : public Element2D<T, 4> {
    public:
        static constexpr unsigned NumVertices = 4;

        typedef typename Element2D<T, NumVertices>::point_t point_t;
        typedef typename Element2D<T, NumVertices>::mesh_element_vertex_point_vec_t
            mesh_element_vertex_point_vec_t;
        typedef typename Element2D<T, NumVertices>::diag_matrix_vec_t diag_matrix_vec_t;

        /**
         * @brief Returns the coordinates of the vertices of the reference element.
         * in the order of the local degrees of freedom. (right-hand rule)
         *
         * @return mesh_element_vertex_point_vec_t (Vector<Vector<T, 2>, 4>)
         */
        KOKKOS_FUNCTION mesh_element_vertex_point_vec_t getLocalVertices() const;

        /**
         * @brief Returns the Jacobian of the transformation matrix.
         *
         * @param global_vertices A vector of the vertex coordinates of the global element to
         * transform to.
         *
         * @return diag_matrix_vec_t (Vector<T, 2>) - A vector representing the diagonal elements of
         * the Jacobian matrix
         */
        KOKKOS_FUNCTION diag_matrix_vec_t getTransformationJacobian(
            const mesh_element_vertex_point_vec_t& global_vertices) const;

        /**
         * @brief Returns the inverse of the Jacobian of the transformation matrix.
         *
         * @param global_vertices A vector of the vertex coordinates of the global element to
         * transform to.
         *
         * @return diag_matrix_vec_t (Vector<T, 2>) - A vector representing the diagonal elements of
         * the inverse Jacobian matrix
         */
        KOKKOS_FUNCTION diag_matrix_vec_t getInverseTransformationJacobian(
            const mesh_element_vertex_point_vec_t& global_vertices) const;

        KOKKOS_FUNCTION point_t globalToLocal(const mesh_element_vertex_point_vec_t&, const point_t&) const;

        KOKKOS_FUNCTION point_t localToGlobal(const mesh_element_vertex_point_vec_t& global_vertices,
                              const point_t& point) const;
        
        KOKKOS_FUNCTION T getDeterminantOfTransformationJacobian(
            const mesh_element_vertex_point_vec_t& global_vertices) const;
        
        KOKKOS_FUNCTION diag_matrix_vec_t getInverseTransposeTransformationJacobian(
            const mesh_element_vertex_point_vec_t& global_vertices) const;

        KOKKOS_FUNCTION bool isPointInRefElement(const Vector<T, 2>& point) const;
    };
}  // namespace ippl

#include "FEM/Elements/QuadrilateralElement.hpp"

#endif
