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
     * @brief This class represents an element in a finite element mesh.
     * It contains both the information about the local and global geometry
     * and its location in the mesh.
     *
     * @tparam Dim The dimension (1 for 1D, 2 for 2D, 3 for 3D)
     * @tparam NumVertices the number of vertices (corners of the element)
     * @tparam NumNodes The total number of nodes (vertices and midpoints of the element, if
     * existent)
     */
    template <unsigned Dim, unsigned NumVertices>
    class Element {
    public:
        Element(const std::size_t& global_index,
                Vector<std::size_t, NumVertices>& global_indices_of_vertices);

        virtual const Vector<std::size_t, NumVertices>& getGlobalIndicesOfVertices() const;

        virtual const Vector<Vector<T, Dim>, NumVertices>& getGlobalVertices() const = 0;

        template <unsigned NumNodes>
        virtual const Vector<Vector<T, Dim>, NumNodes>& getGlobalNodes() const = 0;

        virtual bool operator==(const Element& other) const;

    protected:
        /**
         * @brief The global index of the element.
         *
         * - In a 3D uniform cartesian mesh with hexahedral elements, this global
         * index is the index of the cell.
         * - In a 2D triangular mesh this global index is the index of the cell
         * of the triangle.
         */
        std::size_t global_index_m;
        // TODO maybe add references to entities with a higher co-dimension.

        Vector<std::size_t, NumVertices> global_indices_of_vertices_m;
    };

}  // namespace ippl

#include "FEM/Elements/Element.hpp"

#endif