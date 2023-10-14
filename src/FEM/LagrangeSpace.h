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
        LagrangeSpace(const Mesh<T, Dim>& mesh,
                      const Element<T, Dim, Dim, NumElementVertices>& ref_element,
                      const Quadrature<T, NumIntegrationPoints>& quadrature);

        typedef typename FiniteElementSpace<T, Dim, NumElementVertices,
                                            NumIntegrationPoints>::index_vector_t index_vector_t;
        typedef typename FiniteElementSpace<T, Dim, NumElementVertices,
                                            NumIntegrationPoints>::vertex_vector_t vertex_vector_t;

        /**
         * @brief Eveluate the load vector at the given index.
         *
         * @param j The index of the load vector
         * @return T The value of the load vector at the given index
         */
        T evaluateLoadVector(const std::size_t& j) const override;

        /**
         * @brief Evaluate the stiffness matrix at the given indices.
         *
         * @param i The row index of the stiffness matrix
         * @param j The column index of the stiffness matrix
         * @return T The value of the stiffness matrix at the given indices
         */
        T evaluateStiffnessMatrix(const std::size_t& i, const std::size_t& j) const override;

        /**
         * @brief Get the index vector from the element index.
         *
         * @param element_index The index of the element.
         * @return index_vector_t
         */
        index_vector_t getDimensionIndicesForElement(
            const std::size_t& element_index) const override;

        /**
         * @brief Get the vertices for an elment given the element indices in each dimension of the
         * mesh.
         *
         * @param element_indices The indices of the element in each dimension of the mesh.
         * @return Vector<std::size_t, NumVertices>
         */
        vertex_vector_t getGlobalVerticesForElement(
            const index_vector_t& element_indices) const override;
    };

}  // namespace ippl

#include "FEM/LagrangeSpace.hpp"

#endif