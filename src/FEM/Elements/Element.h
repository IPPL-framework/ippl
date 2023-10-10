//
// Class Element
//   The Element base class. This is an abstract base class representing an
//   element type for finite element methods.
//
#ifndef IPPL_ELEMENT_H
#define IPPL_ELEMENT_H

#include "Types/Vector.h"

#include "FEM/Singleton.h"

namespace ippl {

    /**
     * @brief A base reference element class that follows the singleton design pattern.
     * Meaning only one instance exists at a time for all classes that inherit it.
     *
     * @tparam T type of the element
     * @tparam Dim dimension of the element
     */
    template <typename T, unsigned GeometricDim, unsigned TopologicalDim, unsigned NumVertices>
    class Element : public Singleton<Element<T, GeometricDim, TopologicalDim, NumVertices>> {
    public:
        using local_vertex_vector  = Vector<Vector<T, TopologicalDim>, NumVertices>;
        using global_vertex_vector = Vector<Vector<T, GeometricDim>, NumVertices>;

        using jacobian_type = int;  // TODO

        /***/
        virtual local_vertex_vector getLocalVertices() const = 0;

        /***/
        virtual jacobian_type getTransformationJacobian(
            const global_vertex_vector& global_vertices) const = 0;

        virtual global_vertex_vector getGlobalNodes(
            const jacobian_type& transformation_jacobian) const = 0;

    private:
        Element()  = default;
        ~Element() = default;
    };

    template <typename T, unsigned GeometricDim, unsigned NumVertices>
    using Element1D = Element<T, GeometricDim, 1, NumVertices>;

    template <typename T, unsigned GeometricDim, unsigned NumVertices>
    using Element2D = Element<T, GeometricDim, 2, NumVertices>;

    template <typename T, unsigned GeometricDim, unsigned NumVertices>
    using Element3D = Element<T, GeometricDim, 3, NumVertices>;

}  // namespace ippl

#endif
