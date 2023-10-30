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

        ///////////////////////////////////////////////////////////////////////
        // Mesh types /////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        // A point in the global coordinate system
        typedef typename FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                            NumDoFs>::point_t point_t;

        // A gradient vector in the global coordinate system
        typedef typename FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                            NumDoFs>::gradient_vec_t gradient_vec_t;

        // An index of a vertex in the mesh
        typedef typename FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                            NumDoFs>::mesh_vertex_index_t mesh_vertex_index_t;

        // A vector of vertex indices of the mesh
        typedef typename FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                            NumDoFs>::mesh_vertex_vec_t mesh_vertex_vec_t;

        // A vector with the position of the vertex in the mesh in each dimension
        typedef typename FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                            NumDoFs>::mesh_vertex_pos_t mesh_vertex_pos_t;

        ///////////////////////////////////////////////////////////////////////
        // Element types //////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        // A point in the local coordinate system of an element
        typedef typename FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                            NumDoFs>::local_point_t local_point_t;

        // An index of an element in the mesh
        typedef typename FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                            NumDoFs>::element_index_t element_index_t;

        // A vector with the position of the element in the mesh in each dimension
        typedef typename FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                            NumDoFs>::element_pos_t element_pos_t;

        ///////////////////////////////////////////////////////////////////////
        // DoF types //////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        // An index of a global degree of freedom of the finite element space
        typedef typename FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                            NumDoFs>::global_dof_index_t global_dof_index_t;

        // An index of a local degree of freedom of an element
        typedef typename FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                            NumDoFs>::local_dof_index_t local_dof_index_t;

        // A vector of storing a value for all degrees of freedom
        typedef typename FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                            NumDoFs>::dof_val_vec_t dof_val_vec_t;

        LagrangeSpace(const Mesh<T, Dim>& mesh,
                      const Element<T, Dim, NumElementVertices>& ref_element,
                      const Quadrature<T, NumIntegrationPoints>& quadrature);

        ///////////////////////////////////////////////////////////////////////
        /// Mesh and element operations ///////////////////////////////////////
        // TODO maybe move to Finite Element Space instead or own
        ///////////////////////////////////////////////////////////////////////

        mesh_vertex_pos_t getElementPositionFromIndex(
            const global_dof_index_t& element_index) const override;

        mesh_vertex_pos_t getMeshVertexPositionFromIndex(
            const global_dof_index_t& dof_index) const override;

        mesh_vertex_vec_t getMeshVerticesForElement(
            const mesh_vertex_pos_t& element_indices) const override;

        ///////////////////////////////////////////////////////////////////////
        /// Degree of Freedom operations //////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        point_t getCoordinatesForDof(const global_dof_index_t& dof_index) const;

        std::vector<global_dof_index_t> getElementsInSupportOfDof(
            const global_dof_index_t& dof_index) const override;

        ///////////////////////////////////////////////////////////////////////
        /// Basis functions and gradients /////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        T evaluateBasis(const global_dof_index_t& local_vertex_index,
                        const Vector<T, Dim>& local_coordinates) const override;

        gradient_vec_t evaluateBasisGradient(
            const global_dof_index_t& local_vertex_index,
            const Vector<T, Dim>& local_coordinates) const override;

        ///////////////////////////////////////////////////////////////////////
        /// Assembly operations ///////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        void evaluateLoadVector(dof_val_vec_t& b) const override;

        void evaluateAx(const dof_val_vec_t& x, dof_val_vec_t& Ax) const override;

    private:
        NDIndex<Dim> makeNDIndex(const mesh_vertex_pos_t& indices) const;
    };

}  // namespace ippl

#include "FEM/LagrangeSpace.hpp"

#endif