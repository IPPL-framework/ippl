//
// Class Element
//   The Element base class. This is an abstract base class representing an
//   element type for finite element methods.
//
#ifndef IPPL_ELEMENT_H
#define IPPL_ELEMENT_H

#include "Types/Vector.h"

namespace ippl {

    /**
     * @brief A base reference element class that follows the singleton design pattern.
     * Meaning only one instance exists at a time for all classes that inherit it.
     *
     * @tparam T type of the element
     * @tparam Dim dimension of the element
     */
    template <typename T, unsigned GeometricDim, unsigned TopologicalDim, unsigned NumVertices>
    class Element {
    public:
        static_assert(GeometricDim >= TopologicalDim,
                      "The finite element geometric dimension must greater or equal the "
                      "topological dimension");

        using local_vertex_vector  = Vector<Vector<T, TopologicalDim>, NumVertices>;
        using global_vertex_vector = Vector<Vector<T, GeometricDim>, NumVertices>;

        typedef Vector<Vector<T, GeometricDim>, TopologicalDim>
            jacobian_t;  // TODO this does not include the translation
        typedef Vector<Vector<T, TopologicalDim>, GeometricDim>
            inverse_jacobian_t;  // TODO this does not include the translation

        /**
         * @brief Get the vertices of the element in the local coordinate system.
         *
         * @return local_vertex_vector the vertices of the element in the local coordinate system.
         */
        virtual local_vertex_vector getLocalVertices() const = 0;

        /**
         * @brief Pure virtual function that child elements need to override that returns the
         * transformation matrix without the translation from the global coordinate system to the
         * local element coordinate system.
         *
         * @param global_vertices the vertices of the element in the global coordinate system.
         * @return jacobian_t
         */
        virtual jacobian_t getLinearTransformationJacobian(
            const global_vertex_vector& global_vertices) const = 0;

        /**
         * @brief Pure virtual function that child elements need to override that returns the
         * transformation matrix without the translation from the local element coordinate system to
         * the global coordinate system.
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
        virtual inverse_jacobian_t getInverseLinearTransformationJacobian(
            const global_vertex_vector& global_vertices) const = 0;
    };

    template <typename T, unsigned GeometricDim, unsigned NumVertices>
    using Element1D = Element<T, GeometricDim, 1, NumVertices>;

    template <typename T, unsigned GeometricDim, unsigned NumVertices>
    using Element2D = Element<T, GeometricDim, 2, NumVertices>;

    template <typename T, unsigned GeometricDim, unsigned NumVertices>
    using Element3D = Element<T, GeometricDim, 3, NumVertices>;

}  // namespace ippl

#endif
