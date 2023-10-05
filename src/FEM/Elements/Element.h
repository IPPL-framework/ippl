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
     * @tparam Dim
     * @tparam NumVertices
     */
    template <typename T, unsigned Dim, unsigned NumVertices>
    class Element : public Singleton<Element> {
    public:
        typedef Vector<Vector<T, Dim>, NumVertices> set_of_vertices_type;
        typedef int jacobian_type;  // TODO

        /***/
        virtual set_of_vertices_type getLocalVertices() const = 0;

        /***/
        virtual jacobian_type getTransformationJacobian(
            const set_of_vertices_type& global_vertices) const = 0;

        /***/
        virtual set_of_vertices_type getGlobalNodes(
            const jacobian_type& transformation_jacobian) const = 0;

    private:
        Element() = 0;
    };

}  // namespace ippl

#endif