//
// Class Element
//   The Element base class. This is an abstract base class representing an
//   element type for finite element methods.
//
#ifndef IPPL_ELEMENT_H
#define IPPL_ELEMENT_H

#include "Types/Vector.h"

namespace ippl {

    template <unsigned Dim, unsigned NumVertices>
    class Element {
    public:
        Element(const std::size_t& global_index,
                Vector<std::size_t, NumVertices>& global_indices_of_vertices);

        virtual const Vector<std::size_t, NumVertices>& getGlobalIndicesOfVertices() const;

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

#include "FEM/Element.hpp"

#endif