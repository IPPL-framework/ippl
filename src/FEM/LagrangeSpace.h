// Class LagrangeSpace
//    This is the LagrangeSpace class. It is a class representing a Lagrange space
//    for finite element methods on a structured grid.

#ifndef IPPL_LAGRANGESPACE_H
#define IPPL_LAGRANGESPACE_H

#include <cmath>

#include "FEM/FiniteElementSpace.h"

constexpr unsigned getLagrangeNumElementDOFs(unsigned Dim, unsigned Order) {
    // needs to be constexpr pow function to work at compile time. Kokkos::pow doesn't work.
    return static_cast<unsigned>(pow(static_cast<double>(Order + 1), static_cast<double>(Dim)));
}

namespace ippl {

    // template <typename T>
    //  IsQuadrature = std::is_base_of<Quadrature, T>::value;

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    // requires IsQuadrature<QuadratureType>
    class LagrangeSpace
        : public FiniteElementSpace<T, Dim, getLagrangeNumElementDOFs(Dim, Order), QuadratureType> {
    public:
        static constexpr unsigned numElementDOFs = getLagrangeNumElementDOFs(Dim, Order);

        static constexpr unsigned dim =
            FiniteElementSpace<T, Dim, numElementDOFs, QuadratureType>::dim;
        static constexpr unsigned order = Order;
        static constexpr unsigned numElementVertices =
            FiniteElementSpace<T, Dim, numElementDOFs, QuadratureType>::numElementVertices;
        static constexpr unsigned numIntegrationPoints =
            FiniteElementSpace<T, Dim, numElementDOFs, QuadratureType>::numIntegrationPoints;

        typedef typename FiniteElementSpace<T, Dim, numElementDOFs, QuadratureType>::ElementType
            ElementType;

        // An unsigned integer number representing an index
        typedef
            typename FiniteElementSpace<T, Dim, numElementDOFs,
                                        QuadratureType>::index_t index_t;  // look at ippl::Index

        // A vector with the position of the element in the mesh in each dimension
        typedef typename FiniteElementSpace<T, Dim, numElementDOFs, QuadratureType>::ndindex_t
            ndindex_t;

        // A point in the global coordinate system
        typedef
            typename FiniteElementSpace<T, Dim, numElementDOFs, QuadratureType>::point_t point_t;

        // A gradient vector in the global coordinate system
        typedef typename FiniteElementSpace<T, Dim, numElementDOFs, QuadratureType>::gradient_vec_t
            gradient_vec_t;

        // A vector of vertex indices of the mesh
        typedef typename FiniteElementSpace<T, Dim, numElementDOFs,
                                            QuadratureType>::mesh_element_vertex_index_vec_t
            mesh_element_vertex_index_vec_t;

        typedef typename FiniteElementSpace<T, Dim, numElementDOFs,
                                            QuadratureType>::mesh_element_vertex_point_vec_t
            mesh_element_vertex_point_vec_t;

        ///////////////////////////////////////////////////////////////////////
        // Constructors ///////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        LagrangeSpace(const Mesh<T, Dim>& mesh, const ElementType& ref_element,
                      const QuadratureType& quadrature);

        ///////////////////////////////////////////////////////////////////////
        /// Degree of Freedom operations //////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        std::size_t numGlobalDOFs() const override;

        // point_t getCoordsOfDOF(const index_t& dof_index) const override;

        index_t getLocalDOFIndex(const index_t& elementIndex,
                                 const index_t& globalDOFIndex) const override;

        index_t getGlobalDOFIndex(const index_t& elementIndex,
                                  const index_t& localDOFIndex) const override;

        Vector<index_t, numElementDOFs> getLocalDOFIndices() const override;

        Vector<index_t, numElementDOFs> getGlobalDOFIndices(
            const index_t& element_index) const override;

        ///////////////////////////////////////////////////////////////////////
        /// Basis functions and gradients /////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        T evaluateRefElementBasis(const index_t& localDOF,
                                  const point_t& localPoint) const override;

        gradient_vec_t evaluateRefElementBasisGradient(const index_t& localDOF,
                                                       const point_t& localPoint) const override;

        ///////////////////////////////////////////////////////////////////////
        /// Assembly operations ///////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        Kokkos::View<T*> evaluateAx(
            Kokkos::View<const T*> x,
            const std::function<
                T(const index_t&, const index_t&,
                  const Vector<Vector<T, Dim>,
                               LagrangeSpace<T, Dim, Order, QuadratureType>::numElementDOFs>&)>&
                evalFunction) const override;

        Kokkos::View<T*> evaluateLoadVector() const override;

        ///////////////////////////////////////////////////////////////////////
        /// Helper functions ///////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        static NDIndex<Dim> makeNDIndex(const Vector<T, Dim>& indices);
    };

}  // namespace ippl

#include "FEM/LagrangeSpace.hpp"

#endif