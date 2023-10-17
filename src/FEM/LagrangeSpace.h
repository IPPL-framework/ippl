// Class LagrangeSpace
//    This is the LagrangeSpace class. It is a class representing a Lagrange space
//    for finite element methods on a structured grid.

#ifndef IPPL_LAGRANGESPACE_H
#define IPPL_LAGRANGESPACE_H

#include "FEM/FiniteElementSpace.h"

namespace ippl {

    /**
     * @brief This is the LagrangeSpace class. It is a class representing a Lagrange finite element
     * space for finite element methods on a structured grid.
     *
     * @tparam T The type of the coordinates of the mesh.
     * @tparam Dim The dimension of the mesh.
     * @tparam NumElementVertices The number of vertices per element.
     * @tparam NumIntegrationPoints The number of integration points per element.
     */
    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints>
    class LagrangeSpace
        : public FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints> {
    public:
        LagrangeSpace(const Mesh<T, Dim>& mesh,
                      const Element<T, Dim, Dim, NumElementVertices>& ref_element,
                      const Quadrature<T, NumIntegrationPoints>& quadrature);

        typedef typename FiniteElementSpace<T, Dim, NumElementVertices,
                                            NumIntegrationPoints>::vertex_vector_t vertex_vector_t;
        typedef
            typename FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints>::index_t
                index_t;
        typedef typename FiniteElementSpace<T, Dim, NumElementVertices,
                                            NumIntegrationPoints>::index_vec_t index_vec_t;

        /**
         * @brief Eveluate the load vector at the given index.
         *
         * @param j The index of the load vector
         * @return T The value of the load vector at the given index
         */
        T evaluateLoadVector(const index_t& j) const override;

        /**
         * @brief Evaluate the stiffness matrix at the given indices.
         *
         * @param i The row index of the stiffness matrix
         * @param j The column index of the stiffness matrix
         * @return T The value of the stiffness matrix at the given indices
         */
        T evaluateStiffnessMatrix(const index_t& i, const index_t& j) const override;

        /**
         * @brief Get the index vector from the element index.
         *
         * @param element_index The index of the element.
         * @return index_vector_t
         */
        index_vec_t getDimensionIndicesForElement(const index_t& element_index) const override;

        /**
         * @brief Get the dimension indices for vertex object
         *
         * @param vertex_index
         * @return NDIndex<Dim>
         */
        index_vec_t getDimensionIndicesForVertex(const index_t& vertex_index) const override;

        Vector<T, Dim> getCoordinatesForVertex(const index_vec_t& vertex_indices) const;
        Vector<T, Dim> getCoordinatesForVertex(const index_t& vertex_index) const;

        /**
         * @brief Get the vertices for an elment given the element indices in each dimension of the
         * mesh.
         *
         * @param element_indices The indices of the element in each dimension of the mesh.
         * @return Vector<std::size_t, NumVertices>
         */
        vertex_vector_t getGlobalVerticesForElement(
            const index_vec_t& element_indices) const override;

        /**
         * @brief Evaluate the basis functions at the given global vertex and at the given global
         * coordinates.
         *
         * @param vertex_index The index of the vertex to evaluate the shape functions for.
         * @param global_coordinates The local coordinates to evaluate the shape functions at.
         * @return T The value of the shape functions at the given local coordinates.
         */
        T evaluateBasis(const index_t& vertex_index,
                        const Vector<T, Dim>& global_coordinates) const override;

    private:
        NDIndex<Dim> makeNDIndex(const index_vec_t& indices) const;
    };

}  // namespace ippl

#include "FEM/LagrangeSpace.hpp"

#endif