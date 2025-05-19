// Class EdgeElement

#ifndef IPPL_EDGEELEMENT_H
#define IPPL_EDGEELEMENT_H

#include "FEM/Elements/Element.h"

namespace ippl {

    // this is the basic 1D element with 2 vertices
    // N.B. to not be confused with Nédélec edge elements
    template <typename T>
    class EdgeElement : public Element1D<T, 2> {
    public:
        static constexpr unsigned NumVertices = 2;

        typedef typename Element1D<T, NumVertices>::point_t point_t;
        typedef typename Element1D<T, NumVertices>::vertex_points_t vertex_points_t;

        /**
         * @brief Function to return the coordinates of the vertices of the reference
         * element.
         *
         * @return vertex_points_t (Vector<Vector<T, 1>, 2>)
         */
        KOKKOS_FUNCTION vertex_points_t getLocalVertices() const;

        /**
         * @brief Function to return the Jacobian of the transformation matrix.
         *
         * @param global_vertices A vector of the vertex coordinates of the global element to
         * transform to.
         *
         * @return point_t (Vector<T, 1>) - A vector representing the diagonal elements
         * of the Jacobian matrix
         */
        KOKKOS_FUNCTION point_t
        getTransformationJacobian(const vertex_points_t& global_vertices) const;

        /**
         * @brief Function to return the inverse of the Jacobian of the transformation
         * matrix.
         *
         * @param global_vertices A vector of the vertex coordinates of the global element to
         * transform to.
         *
         * @return point_t (Vector<T, 1>) - A vector representing the diagonal elements
         * of the inverse Jacobian matrix
         */
        KOKKOS_FUNCTION point_t
        getInverseTransformationJacobian(const vertex_points_t& global_vertices) const;

        /**
         * @brief Transforms a point from global to local coordinates.
         *
         * @param global_vertices A vector of the vertex indices of the global element to transform
         * to in the mesh.
         * @param point A point in global coordinates with respect to the global element.
         *
         * @return point_t
         */
        KOKKOS_FUNCTION point_t globalToLocal(const vertex_points_t&, const point_t&) const;

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
        KOKKOS_FUNCTION point_t localToGlobal(const vertex_points_t& global_vertices,
                                              const point_t& point) const;

        /**
         * @brief Returns the determinant of the transformation Jacobian.
         *
         * @param global_vertices A vector of the vertex coordinates of the global element to
         * transform to.
         *
         * @return T - The determinant of the transformation Jacobian
         */
        KOKKOS_FUNCTION T
        getDeterminantOfTransformationJacobian(const vertex_points_t& global_vertices) const;

        /**
         * @brief Returns the inverse of the transpose of the transformation Jacobian.
         *
         * @param global_vertices A vector of the vertex coordinates of the global element to
         *  transform to.
         *
         * @return point_t (Vector<T, 1>) - A vector representing the diagonal elements
         * of the inverse transpose Jacobian matrix
         */
        KOKKOS_FUNCTION point_t
        getInverseTransposeTransformationJacobian(const vertex_points_t& global_vertices) const;

        /**
         * @brief Returns whether a point in local coordinates ([0, 1]) is inside the reference
         * element.
         *
         * @param point A point in local coordinates with respect to the reference element.
         * @return boolean - Returns true when the point is inside the reference element or on the
         * boundary. Returns false else
         */
        KOKKOS_FUNCTION bool isPointInRefElement(const Vector<T, 1>& point) const;
    };

}  // namespace ippl

#include "FEM/Elements/EdgeElement.hpp"

#endif
