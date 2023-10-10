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
    template <typename T, unsigned Dim>
    class Element : public Singleton<Element> {
    public:
        template <unsigned NumVertices>
        using set_of_vertices_type Vector<Vector<T, Dim>, NumVertices>;

        typedef int jacobian_type;  // TODO

        /***/
        template <typename NumVertices>
        virtual set_of_vertices_type<NumVertices> getLocalVertices() const = 0;

        /***/
        template <typename NumVertices>
        virtual jacobian_type getTransformationJacobian(
            const set_of_vertices_type<NumVertices>& global_vertices) const = 0;

        /***/
        template <typename NumVertices>
        virtual set_of_vertices_type<NumVertices> getGlobalNodes(
            const jacobian_type& transformation_jacobian) const = 0;

    private:
        Element() = 0;
    };

    template <typename T>
    using Element1D = Element<T, 1>;

    template <typename T>
    using Element2D = Element<T, 2>;

    template <typename T>
    using Element3D = Element<T, 3>;

}  // namespace ippl

#endif
