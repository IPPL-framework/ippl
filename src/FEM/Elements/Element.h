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
        typedef Vector<point_t, NumVertices> vertex_vec_t;

        // a matrix defining a transformtaion in the local or global coordinate system
        typedef Vector<Vector<T, Dim>, Dim> matrix_t;

        virtual vertex_vec_t getLocalVertices() const = 0;

        virtual matrix_t getTransformationJacobian(const vertex_vec_t& global_vertices) const = 0;

        virtual matrix_t getInverseTransformationJacobian(
            const vertex_vec_t& global_vertices) const = 0;

        virtual T getDeterminantOfTransformationJacobian(
            const vertex_vec_t& global_vertices) const = 0;

        virtual point_t globalToLocal(const vertex_vec_t&, const point_t&) const;

        virtual point_t localToGlobal(const vertex_vec_t&, const point_t&) const;
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
