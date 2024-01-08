//

#ifndef IPPL_ELEMENT_H
#define IPPL_ELEMENT_H

#include "Types/Vector.h"

namespace ippl {

    /**
     * @brief Base class for all elements.
     * @tparam T The type of the coordinates of the vertices of the element.
     * @tparam Dim The dimension of the element.
     * @tparam NumVertices The number of vertices of the element.
     */
    template <typename T, unsigned Dim, unsigned NumVertices>
    class Element {
    public:
        static constexpr unsigned dim         = Dim;
        static constexpr unsigned numVertices = NumVertices;

        // A point in the local or global coordinate system
        typedef Vector<T, Dim> point_t;

        // A list of all vertices
        typedef Vector<point_t, NumVertices> mesh_element_vertex_point_vec_t;

        // a matrix defining a transformtaion in the local or global coordinate system
        typedef Vector<T, Dim> diag_matrix_vec_t;

        virtual mesh_element_vertex_point_vec_t getLocalVertices() const = 0;

        point_t globalToLocal(const mesh_element_vertex_point_vec_t&, const point_t&) const;

        /**
         * @brief Transforms a point from local to global coordinates.
         *
         * @param global_vertices A vector of the vertex indices of the global element to transform
         * to in the mesh.
         * @param point A point in local coordinates with respect to the reference element.
         *
         * @details Equivalent to transforming a local point \f$\hat{\boldsymbol{x}}\f$ on the local
         * element \f$\hat{K}\f$ to a point in the global coordinate system \f$\boldsymbol{x}\f$ on
         * \f$K\f$ by applying the transformation \f$\mathbf{\Phi}_K\f$ \f\[\boldsymbol{x} =
         * \mathbf{\Phi}_K(\hat{\boldsymbol{x}})\f\]
         *
         * @return point_t
         */
        point_t localToGlobal(const mesh_element_vertex_point_vec_t& global_vertices,
                              const point_t& point) const;

        T getDeterminantOfTransformationJacobian(
            const mesh_element_vertex_point_vec_t& global_vertices) const;

        diag_matrix_vec_t getInverseTransposeTransformationJacobian(
            const mesh_element_vertex_point_vec_t& global_vertices) const;

        /**
         * @brief Returns whether a point in local coordinates ([0, 1]^Dim) is inside the reference
         * element.
         *
         * @param point A point in local coordinates with respect to the reference element.
         * @return boolean - Returns true when the point is inside the reference element or on the
         * boundary. Returns false else
         */
        bool isPointInRefElement(const Vector<T, Dim>& point) const;

    protected:
        virtual diag_matrix_vec_t getTransformationJacobian(
            const mesh_element_vertex_point_vec_t& global_vertices) const = 0;

        virtual diag_matrix_vec_t getInverseTransformationJacobian(
            const mesh_element_vertex_point_vec_t& global_vertices) const = 0;
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
