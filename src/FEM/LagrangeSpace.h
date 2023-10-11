// Class LagrangeSpace
//    This is the LagrangeSpace class. It is a class representing a Lagrange space
//    for finite element methods on a structured grid.

#ifndef IPPL_LAGRANGESPACE_H
#define IPPL_LAGRANGESPACE_H

#include "FEM/FiniteElementSpace.h"

namespace ippl {

    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints>
    class LagrangeSpace
        : public FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints> {
    public:
        // This is the number of vertices per element.
        // Since it is assumed that Mesh is a structured grid, the number of vertices per element
        // follows 2^Dim.
        static constexpr std::size_t NumVertices = 1 << Dim;

        LagrangeSpace(const Mesh<T, Dim>& mesh,
                      const Element<T, Dim, Dim, NumElementVertices>& ref_element,
                      const Quadrature<T, NumIntegrationPoints>& quadrature);

        /**
         * @brief Get the vertices for an element given the element index.
         *
         * @param element_index The index of the element.
         * @return Vector<std::size_t, NumVertices>
         */
        Vector<std::size_t, NumVertices> getVerticesForElement(
            const std::size_t& element_index) const;

        /***/
        Vector<std::size_t, Dim> getElementDimIndices(const std::size_t& element_index) const;

        /**
         * @brief Get the vertices for an elment given the element indices in each dimension of the
         * mesh.
         *
         * @param element_indices The indices of the element in each dimension of the mesh.
         * @return Vector<std::size_t, NumVertices>
         */
        Vector<std::size_t, NumVertices> getVerticesForElement(
            const Vector<std::size_t, Dim>& element_indices) const;
    };

}  // namespace ippl

#include "FEM/LagrangeSpace.hpp"

#endif