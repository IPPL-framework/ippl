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

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType, typename FieldLHS,
              typename FieldRHS>
    // requires IsQuadrature<QuadratureType>
    class LagrangeSpace : public FiniteElementSpace<T, Dim, getLagrangeNumElementDOFs(Dim, Order),
                                                    QuadratureType, FieldLHS, FieldRHS> {
    public:
        static constexpr unsigned numElementDOFs = getLagrangeNumElementDOFs(Dim, Order);

        static constexpr unsigned dim =
            FiniteElementSpace<T, Dim, numElementDOFs, QuadratureType, FieldLHS, FieldRHS>::dim;
        static constexpr unsigned order = Order;
        static constexpr unsigned numElementVertices =
            FiniteElementSpace<T, Dim, numElementDOFs, QuadratureType, FieldLHS,
                               FieldRHS>::numElementVertices;
        static constexpr unsigned numIntegrationPoints =
            FiniteElementSpace<T, Dim, numElementDOFs, QuadratureType, FieldLHS,
                               FieldRHS>::numIntegrationPoints;

        typedef typename FiniteElementSpace<T, Dim, numElementDOFs, QuadratureType, FieldLHS,
                                            FieldRHS>::ElementType ElementType;

        // An unsigned integer number representing an index
        typedef typename FiniteElementSpace<T, Dim, numElementDOFs, QuadratureType, FieldLHS,
                                            FieldRHS>::index_t index_t;  // look at ippl::Index

        // A vector with the position of the element in the mesh in each dimension
        typedef typename FiniteElementSpace<T, Dim, numElementDOFs, QuadratureType, FieldLHS,
                                            FieldRHS>::ndindex_t ndindex_t;

        // A point in the global coordinate system
        typedef typename FiniteElementSpace<T, Dim, numElementDOFs, QuadratureType, FieldLHS,
                                            FieldRHS>::point_t point_t;

        // A gradient vector in the global coordinate system
        typedef typename FiniteElementSpace<T, Dim, numElementDOFs, QuadratureType, FieldLHS,
                                            FieldRHS>::gradient_vec_t gradient_vec_t;

        // A vector of vertex indices of the mesh
        typedef typename FiniteElementSpace<T, Dim, numElementDOFs, QuadratureType, FieldLHS,
                                            FieldRHS>::mesh_element_vertex_index_vec_t
            mesh_element_vertex_index_vec_t;

        typedef typename FiniteElementSpace<T, Dim, numElementDOFs, QuadratureType, FieldLHS,
                                            FieldRHS>::mesh_element_vertex_point_vec_t
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

        Vector<ndindex_t, numElementDOFs> getGlobalDOFNDIndices(
            const index_t& element_index) const override;

        ///////////////////////////////////////////////////////////////////////
        /// Basis functions and gradients /////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        T evaluateRefElementShapeFunction(const index_t& localDOF,
                                          const point_t& localPoint) const override;

        gradient_vec_t evaluateRefElementShapeFunctionGradient(
            const index_t& localDOF, const point_t& localPoint) const override;

        ///////////////////////////////////////////////////////////////////////
        /// Assembly operations ///////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        FieldLHS evaluateAx(
            const FieldLHS& field,
            const std::function<T(
                const index_t&, const index_t&,
                const Vector<Vector<T, Dim>, LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS,
                                                           FieldRHS>::numElementDOFs>&)>&
                evalFunction) const override;

        void evaluateLoadVector(FieldRHS& rhs_field,
                                const std::function<T(const point_t&)>& f) const override;

        ///////////////////////////////////////////////////////////////////////
        /// Helper functions //////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        template <typename FieldType, std::size_t... Is>
        static T& getFieldEntry(FieldType& field, const ndindex_t& ndindex) {
            return getFieldEntry(field, ndindex, std::make_index_sequence<Dim>());
        }

        template <typename FieldType, std::size_t... Is>
        static T& getFieldEntry(FieldType& field, ndindex_t ndindex,
                                const std::index_sequence<Is...>) {
            static_assert(sizeof...(Is) == Dim, "Number of indices must match the dimension");
            static_assert(sizeof...(Is) == FieldType::view_type::rank,
                          "Number of indices must match the field view rank");

            const unsigned num_ghosts = field.getNghost();

            // offset the NDIndex for the ghost cells
            ndindex += num_ghosts;

            // make sure that the index is within the field (without the ghost cells)
            for (std::size_t i = 0; i < Dim; ++i) {
                if (ndindex[i] < num_ghosts
                    || ndindex[i] > field.getLayout().getDomain()[i].length()) {
                    // throw std::out_of_range("Index out of range");
                    std::cerr << "Index out of range" << std::endl;
                    std::cerr << "Index: " << ndindex[i] << std::endl;
                    std::cerr << "Domain: " << num_ghosts << " - "
                              << field.getLayout().getDomain()[i].length() << std::endl;
                }
            }

            return field.getView()(ndindex[Is]...);
        };

        bool isDOFOnBoundary(const ndindex_t& ndindex, const unsigned& numGhosts) const {
            assert(numGhosts <= 1 && "Only one ghost cell is supported");
            for (index_t k = 0; k < Dim; ++k) {
                if (ndindex[k] <= 0 || ndindex[k] >= this->mesh_m.getGridsize(k) - 1) {
                    return true;
                }
            }
            return false;
        }
    };

}  // namespace ippl

#include "FEM/LagrangeSpace.hpp"

#endif