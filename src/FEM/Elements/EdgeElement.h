// Class EdgeElement

#ifndef IPPL_EDGEELEMENT_H
#define IPPL_EDGEELEMENT_H

#include "FEM/Elements/Element.h"

namespace ippl {

    template <typename T>
    class EdgeElement : public Element1D<T, 2> {
    public:
        static constexpr unsigned NumVertices = 2;

        typedef typename Element1D<T, NumVertices>::point_t point_t;
        typedef typename Element1D<T, NumVertices>::mesh_element_vertex_point_vec_t
            mesh_element_vertex_point_vec_t;
        typedef typename Element1D<T, NumVertices>::diag_matrix_vec_t diag_matrix_vec_t;

        KOKKOS_FUNCTION mesh_element_vertex_point_vec_t getLocalVertices() const;

        KOKKOS_FUNCTION diag_matrix_vec_t getTransformationJacobian(
            const mesh_element_vertex_point_vec_t& global_vertices) const;

        KOKKOS_FUNCTION diag_matrix_vec_t getInverseTransformationJacobian(
            const mesh_element_vertex_point_vec_t& global_vertices) const;

        KOKKOS_FUNCTION point_t globalToLocal(const mesh_element_vertex_point_vec_t&, const point_t&) const;

        KOKKOS_FUNCTION point_t localToGlobal(const mesh_element_vertex_point_vec_t& global_vertices,
                              const point_t& point) const;
        
        KOKKOS_FUNCTION T getDeterminantOfTransformationJacobian(
            const mesh_element_vertex_point_vec_t& global_vertices) const;
        
        KOKKOS_FUNCTION diag_matrix_vec_t getInverseTransposeTransformationJacobian(
            const mesh_element_vertex_point_vec_t& global_vertices) const;

        KOKKOS_FUNCTION bool isPointInRefElement(const Vector<T, 1>& point) const;
    };

}  // namespace ippl

#include "FEM/Elements/EdgeElement.hpp"

#endif
