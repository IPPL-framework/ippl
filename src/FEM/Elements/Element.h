//

#ifndef IPPL_ELEMENT_H
#define IPPL_ELEMENT_H

#include "Types/Vector.h"

namespace ippl {

    template <typename T, unsigned Dim, unsigned NumVertices>
    class Element {
    public:
        // A point in the local or global coordinate system
        typedef Vector<T, Dim> point_t;

        // A list of all vertices
        typedef Vector<point_t, NumVertices> mesh_vertex_vec_t;

        // a matrix defining a transformtaion in the local or global coordinate system
        typedef Vector<T, Dim> diag_matrix_vec_t;

        virtual mesh_vertex_vec_t getLocalVertices() const = 0;

        virtual diag_matrix_vec_t getTransformationJacobian(
            const mesh_vertex_vec_t& global_vertices) const = 0;

        virtual diag_matrix_vec_t getInverseTransformationJacobian(
            const mesh_vertex_vec_t& global_vertices) const = 0;

        virtual T getDeterminantOfTransformationJacobian(
            const mesh_vertex_vec_t& global_vertices) const;

        virtual point_t globalToLocal(const mesh_vertex_vec_t&, const point_t&) const;

        virtual point_t localToGlobal(const mesh_vertex_vec_t&, const point_t&) const;

        /**
         * @brief Returns whether a point in local coordinates ([0, 1]^Dim) is inside the reference
         * element.
         *
         * @param point A point in local coordinates with respect to the reference element.
         * @return boolean - Returns true when the point is inside the reference element or on the
         * boundary. Returns false else
         */
        bool isLocalPointInRefElement(const Vector<T, Dim>& point) const;
    };

    template <typename T, unsigned NumVertices>
    using Element1D = Element<T, 1, NumVertices>;

    template <typename T, unsigned NumVertices>
    using Element2D = Element<T, 2, NumVertices>;

    template <typename T, unsigned NumVertices>
    using Element3D = Element<T, 3, NumVertices>;

}  // namespace ippl

#include "FEM/Elements/Element.hpp"

#endif
