// Class LagrangeSpace
//    This is the LagrangeSpace class. It is a class representing a Lagrange space
//    for finite element methods on a structured grid.

#ifndef IPPL_LAGRANGESPACE_H
#define IPPL_LAGRANGESPACE_H

#include <cmath>  // for pow // TODO maybe replace to make Kokkos compatible

#include "FEM/FiniteElementSpace.h"

inline constexpr unsigned calculateLagrangeNumDoFs(unsigned Dim, unsigned Order) {
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
        static constexpr unsigned NumElementDOFs = calculateLagrangeNumDoFs(Dim, Order);

        // An unsigned integer number representing an index
        typedef
            typename FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                        NumElementDOFs>::index_t index_t;  // look at ippl::Index

        // A vector with the position of the element in the mesh in each dimension
        typedef typename FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                            NumElementDOFs>::nd_index_t
            nd_index_t;  // TODO look ad NDINDEX

        // A point in the global coordinate system
        typedef typename FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                            NumElementDOFs>::point_t point_t;

        // A gradient vector in the global coordinate system
        typedef typename FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                            NumElementDOFs>::gradient_vec_t gradient_vec_t;

        // A vector of vertex indices of the mesh
        typedef typename FiniteElementSpace<T, Dim, NumElementVertices, NumIntegrationPoints,
                                            NumElementDOFs>::mesh_element_vertex_vec_t
            mesh_element_vertex_vec_t;

        ///////////////////////////////////////////////////////////////////////
        // Constructors ///////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        LagrangeSpace(const Mesh<T, Dim>& mesh,
                      const Element<T, Dim, NumElementVertices>& ref_element,
                      const Quadrature<T, NumIntegrationPoints>& quadrature);

        ///////////////////////////////////////////////////////////////////////
        /// Assembly operations ///////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        template <unsigned NumGlobalDOFs>
        void evaluateAx(const Vector<T, NumGlobalDOFs>& x,
                        Vector<T, NumGlobalDOFs>& Ax) const override;

        template <unsigned NumGlobalDOFs>
        void evaluateLoadVector(Vector<T, NumGlobalDOFs>& b) const override;

    protected:
        ///////////////////////////////////////////////////////////////////////
        /// Degree of Freedom operations //////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        point_t getCoordsOfDOF(const index_t& dof_index) const override;

        ///////////////////////////////////////////////////////////////////////
        /// Basis functions and gradients /////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        T evaluateRefElementBasis(const index_t& local_vertex_index,
                                  const point_t& local_coordinates) const override;

        gradient_vec_t evaluateRefElementBasisGradient(
            const index_t& local_vertex_index, const point_t& local_coordinates) const override;

    private:
        NDIndex<Dim> makeNDIndex(const Vector<T, Dim>& indices) const;
    };

}  // namespace ippl

#include "FEM/LagrangeSpace.hpp"

#endif