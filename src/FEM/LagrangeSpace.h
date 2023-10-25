// Class LagrangeSpace
//    This is the LagrangeSpace class. It is a class representing a Lagrange space
//    for finite element methods on a structured grid.

#ifndef IPPL_LAGRANGESPACE_H
#define IPPL_LAGRANGESPACE_H

#include <cmath>  // for pow // TODO maybe replace to make Kokkos compatible

#include "FEM/FiniteElementSpace.h"

constexpr unsigned calculateLagrangeNumDoFs(unsigned Dim, unsigned Order) {
    return static_cast<unsigned>(pow(static_cast<double>(Order + 1), static_cast<double>(Dim)));
}
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
    template <typename T, unsigned Dim, unsigned Order, unsigned NumElementVertices,
              unsigned NumIntegrationPoints>
    class LagrangeSpace
        : public FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                    calculateLagrangeNumDoFs(Dim, Order)> {
    public:
        static constexpr unsigned NumDoFs = calculateLagrangeNumDoFs(Dim, Order);

        LagrangeSpace(const Mesh<T, Dim>& mesh,
                      const Element<T, Dim, Dim, NumElementVertices>& ref_element,
                      const Quadrature<T, NumIntegrationPoints>& quadrature);

        typedef typename FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                            NumDoFs>::vertex_vec_t vertex_vec_t;
        typedef typename FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                            NumDoFs>::index_t index_t;
        typedef typename FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                            NumDoFs>::index_vec_t index_vec_t;
        typedef typename FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                            NumDoFs>::coord_vec_t coord_vec_t;
        typedef typename FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                            NumDoFs>::value_vec_t value_vec_t;
        typedef typename FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                            NumDoFs>::gradient_vec_t gradient_vec_t;

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
         * @param global_vertex_index
         * @return NDIndex<Dim>
         */
        index_vec_t getDimensionIndicesForVertex(const index_t& global_vertex_index) const override;

        /**
         * @brief Get the coordinates for a vertex given the vertex indices in each dimension of the
         * mesh.
         *
         * @param vertex_indices The indices of the vertex in each dimension of the mesh.
         * @return Vector<T, Dim>
         */
        coord_vec_t getCoordinatesForVertex(const index_vec_t& vertex_indices) const;

        /**
         * @brief Get the coordinates for a vertex given the global vertex index.
         *
         * @param global_vertex_index The global index of the vertex.
         * @return Vector<T, Dim>
         */
        coord_vec_t getCoordinatesForVertex(const index_t& global_vertex_index) const;

        /**
         * @brief Get the vertices for an elment given the element indices in each dimension of the
         * mesh.
         *
         * @param element_indices The indices of the element in each dimension of the mesh.
         * @return Vector<std::size_t, NumVertices>
         */
        vertex_vec_t getGlobalVerticesForElement(const index_vec_t& element_indices) const override;

        /**
         * @brief Returns whether a point in local coordinates ([0, 1]^Dim) is inside the reference
         * element.
         *
         * @param point A point in local coordinates with respect to the reference element.
         * @return boolean - Returns true when the point is inside the reference element or on the
         * boundary. Returns false else
         */
        bool isLocalPointInRefElement(const Vector<T, Dim>& point) const;

        /**
         * @brief Evaluate the element shape functions at the given local coordinates.
         *
         * @param local_vertex_index The local index of the vertex to evaluate the shape functions
         * @param local_coordinates The local coordinates to evaluate the shape functions at.
         * @return T The value of the shape functions at the given local coordinates.
         */
        T evaluateBasis(const index_t& local_vertex_index,
                        const Vector<T, Dim>& local_coordinates) const override;

        /**
         * @brief Function to evaluate the gradient of the element shape functions at
         * the given global vertex and at the given global coordinates.
         *
         * @param local_vertex_index The index of the local vertex to evaluate the gradient of the
         * shape functions for.
         * @param local_coordinates
         * @return Vector<T, Dim> The value of the gradient of the shape functions at the given
         * local coordinates.
         */
        gradient_vec_t evaluateBasisGradient(
            const index_t& local_vertex_index,
            const Vector<T, Dim>& local_coordinates) const override;

        /**
         * @brief Eveluate the load vector at the given index.
         *
         * @param b the load vector to store the result in.
         */
        void evaluateLoadVector(value_vec_t& b) const override;

        /**
         * @brief Evaluate the stiffness matrix at the given indices.
         *
         * @param x The vector x to multiply the stiffness matrix with.
         * @param Ax The vector Ax to store the result in.
         */
        void evaluateAx(const value_vec_t& x, value_vec_t& Ax) const override;

    private:
        NDIndex<Dim> makeNDIndex(const index_vec_t& indices) const;
    };

}  // namespace ippl

#include "FEM/LagrangeSpace.hpp"

#endif