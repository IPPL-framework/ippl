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
        typedef typename Element1D<T, NumVertices>::vertex_points_t
            vertex_points_t;

        KOKKOS_FUNCTION vertex_points_t getLocalVertices() const;

        KOKKOS_FUNCTION point_t getTransformationJacobian(
            const vertex_points_t& global_vertices) const;

        KOKKOS_FUNCTION point_t getInverseTransformationJacobian(
            const vertex_points_t& global_vertices) const;

        KOKKOS_FUNCTION point_t globalToLocal(const vertex_points_t&, const point_t&) const;

        KOKKOS_FUNCTION point_t localToGlobal(const vertex_points_t& global_vertices,
                              const point_t& point) const;
        
        KOKKOS_FUNCTION T getDeterminantOfTransformationJacobian(
            const vertex_points_t& global_vertices) const;
        
        KOKKOS_FUNCTION point_t getInverseTransposeTransformationJacobian(
            const vertex_points_t& global_vertices) const;

        KOKKOS_FUNCTION bool isPointInRefElement(const Vector<T, 1>& point) const;
    };

}  // namespace ippl

#include "FEM/Elements/EdgeElement.hpp"

#endif
