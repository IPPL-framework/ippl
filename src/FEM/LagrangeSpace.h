// Class LagrangeSpace
//    This is the LagrangeSpace class. It is a class representing a Lagrange space
//    for finite element methods on a structured grid.

#ifndef IPPL_LAGRANGESPACE_H
#define IPPL_LAGRANGESPACE_H

#include <cmath>

#include "FEM/FiniteElementSpace.h"

constexpr unsigned getLagrangeNumElementDOFs(unsigned Dim, unsigned Order) {
    // needs to be constexpr pow function to work at compile time. Kokkos::pow doesn't work.
    return static_cast<unsigned>(power(static_cast<int>(Order + 1), static_cast<int>(Dim)));
}

namespace ippl {

    /**
     * @brief A class representing a Lagrange space for finite element methods on a structured,
     * rectilinear grid.
     *
     * @tparam T The floating point number type of the field values
     * @tparam Dim The dimension of the mesh
     * @tparam Order The order of the Lagrange space
     * @tparam QuadratureType The type of the quadrature rule
     * @tparam FieldLHS The type of the left hand side field
     * @tparam FieldRHS The type of the right hand side field
     */
    template <typename T, unsigned Dim, unsigned Order, typename ElementType, 
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    // requires IsQuadrature<QuadratureType>
    class LagrangeSpace : public FiniteElementSpace<T, Dim, getLagrangeNumElementDOFs(Dim, Order),
                                                    ElementType, QuadratureType, FieldLHS, FieldRHS> {
    public:
        // The number of degrees of freedom per element
        static constexpr unsigned numElementDOFs = getLagrangeNumElementDOFs(Dim, Order);

        // The dimension of the mesh
        static constexpr unsigned dim =
            FiniteElementSpace<T, Dim, numElementDOFs, ElementType, QuadratureType, FieldLHS, FieldRHS>::dim;

        // The order of the Lagrange space
        static constexpr unsigned order = Order;

        // The number of mesh vertices per element
        static constexpr unsigned numElementVertices =
            FiniteElementSpace<T, Dim, numElementDOFs, ElementType, QuadratureType, FieldLHS,
                               FieldRHS>::numElementVertices;

        // An unsigned integer number representing an index
        typedef typename FiniteElementSpace<T, Dim, numElementDOFs, ElementType, QuadratureType, FieldLHS,
                                            FieldRHS>::index_t index_t;  // look at ippl::Index

        // A vector with the position of the element in the mesh in each dimension
        typedef typename FiniteElementSpace<T, Dim, numElementDOFs, ElementType, QuadratureType, FieldLHS,
                                            FieldRHS>::ndindex_t ndindex_t;

        // A point in the global coordinate system
        typedef typename FiniteElementSpace<T, Dim, numElementDOFs, ElementType, QuadratureType, FieldLHS,
                                            FieldRHS>::point_t point_t;

        // A gradient vector in the global coordinate system
        typedef typename FiniteElementSpace<T, Dim, numElementDOFs, ElementType, QuadratureType, FieldLHS,
                                            FieldRHS>::gradient_vec_t gradient_vec_t;

        // A vector of vertex indices of the mesh
        typedef typename FiniteElementSpace<T, Dim, numElementDOFs, ElementType, QuadratureType, FieldLHS,
                                            FieldRHS>::mesh_element_vertex_index_vec_t
            mesh_element_vertex_index_vec_t;

        // Vector of vertex points of the mesh
        typedef typename FiniteElementSpace<T, Dim, numElementDOFs, ElementType, QuadratureType, FieldLHS,
                                            FieldRHS>::mesh_element_vertex_point_vec_t
            mesh_element_vertex_point_vec_t;

        typedef typename FiniteElementSpace<T, Dim, numElementDOFs, ElementType, QuadratureType, FieldLHS,
                                            FieldRHS>::mesh_element_vertex_ndindex_vec_t
            mesh_element_vertex_ndindex_vec_t;

        // Field layout type for domain decomposition info
        typedef FieldLayout<Dim> Layout_t;

        // View types
        typedef typename detail::ViewType<T, Dim>::view_type ViewType;
        typedef typename detail::ViewType<T, Dim, Kokkos::MemoryTraits<Kokkos::Atomic>>::view_type AtomicViewType;

        ///////////////////////////////////////////////////////////////////////
        // Constructors ///////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Construct a new LagrangeSpace object
         *
         * @param mesh Reference to the mesh
         * @param ref_element Reference to the reference element
         * @param quadrature Reference to the quadrature rule
         */
        LagrangeSpace(const Mesh<T, Dim>& mesh, ElementType& ref_element,
                      const QuadratureType& quadrature, const Layout_t& layout);

        ///////////////////////////////////////////////////////////////////////
        /**
         * @brief Initialize a Kokkos view containing the element indices
         */
        void initializeElementIndices(const Layout_t& layout);

        /// Degree of Freedom operations //////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Get the number of global degrees of freedom in the space
         *
         * @return size_t - unsigned integer number of global degrees of freedom
         */
        KOKKOS_FUNCTION size_t numGlobalDOFs() const override;

        /**
         * @brief Get the elements local DOF from the element index and global DOF
         * index
         *
         * @param elementIndex index_t (size_t) - The index of the element
         * @param globalDOFIndex index_t (size_t) - The global DOF index
         *
         * @return index_t (size_t) - The local DOF index
         */
        /*
        KOKKOS_FUNCTION index_t getLocalDOFIndex(const index_t& elementIndex,
                                 const index_t& globalDOFIndex) const override;
        */

        /**
         * @brief Get the global DOF index from the element index and local DOF
         *
         * @param elementIndex index_t (size_t) - The index of the element
         * @param localDOFIndex index_t (size_t) - The local DOF index
         *
         * @return index_t (size_t) - The global DOF index
         */
        KOKKOS_FUNCTION index_t getGlobalDOFIndex(const index_t& elementIndex,
                                  const index_t& localDOFIndex) const override;

        /**
         * @brief Get the local DOF indices (vector of local DOF indices)
         * They are independent of the specific element because it only depends on
         * the reference element type
         *
         * @return Vector<index_t, NumElementDOFs> - The local DOF indices
         */
        KOKKOS_FUNCTION Vector<index_t, numElementDOFs> getLocalDOFIndices() const override;

        /**
         * @brief Get the global DOF indices (vector of global DOF indices) of an element
         *
         * @param elementIndex index_t (size_t) - The index of the element
         *
         * @return Vector<index_t, NumElementDOFs> - The global DOF indices
         */
        KOKKOS_FUNCTION Vector<index_t, numElementDOFs> getGlobalDOFIndices(
            const index_t& element_index) const override;

        /**
         * @brief Get the global DOF NDIndices (vector of global DOF NDIndices) of an element
         *
         * @param elementIndex index_t (size_t) - The index of the element
         *
         * @return Vector<ndindex_t, NumElementDOFs> - The global DOF NDIndices
         */
        /*
        KOKKOS_FUNCTION Vector<ndindex_t, numElementDOFs> getGlobalDOFNDIndices(
            const index_t& element_index) const override;
        */

        ///////////////////////////////////////////////////////////////////////
        /// Basis functions and gradients /////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Evaluate the shape function of a local degree of freedom at a given point in the
         * reference element
         *
         * @param localDOF index_t (size_t) - The local degree of freedom index
         * @param localPoint point_t (Vector<T, Dim>) - The point in the reference element
         *
         * @return T - The value of the shape function at the given point
         */
        KOKKOS_FUNCTION T evaluateRefElementShapeFunction(const index_t& localDOF,
                                          const point_t& localPoint) const override;

        /**
         * @brief Evaluate the gradient of the shape function of a local degree of freedom at a
         * given point in the reference element
         *
         * @param localDOF index_t (size_t) - The local degree of freedom index
         * @param localPoint point_t (Vector<T, Dim>) - The point in the reference element
         *
         * @return gradient_vec_t (Vector<T, Dim>) - The gradient of the shape function at the given
         * point
         */
        KOKKOS_FUNCTION gradient_vec_t evaluateRefElementShapeFunctionGradient(
            const index_t& localDOF, const point_t& localPoint) const override;

        ///////////////////////////////////////////////////////////////////////
        /// Assembly operations ///////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Assemble the left stiffness matrix A of the system Ax = b
         *
         * @param field The field to assemble the matrix for
         *
         * @return FieldLHS - The LHS field containing A*x
         */
        template <typename F>
        FieldLHS evaluateAx(
            const FieldLHS& field, F& evalFunction) const;
            //const std::function<T(
            //    const index_t&, const index_t&,
            //    const Vector<Vector<T, Dim>, LagrangeSpace<T, Dim, Order, ElementType, QuadratureType,
            //                                               FieldLHS, FieldRHS>::numElementDOFs>&)>&
            //    evalFunction) const override;

        KOKKOS_FUNCTION
        T evalFunc(const T absDetDPhi,
                   const index_t elementIndex, const index_t& i, const point_t& q_k,
                   const Vector<T, numElementDOFs>& basis_q) const;

        /**
         * @brief Assemble the load vector b of the system Ax = b
         *
         * @param rhs_field The field to set with the load vector
         * @param f The source function
         *
         * @return FieldRHS - The RHS field containing b
         */
        void evaluateLoadVector(FieldRHS& rhs_field) const override;        

    private:        
        /**
         * @brief Check if a DOF is on the boundary of the mesh
         *
         * @param ndindex The NDIndex of the DOF
         *
         * @return true - If the DOF is on the boundary
         * @return false - If the DOF is not on the boundary
         */
        KOKKOS_FUNCTION bool isDOFOnBoundary(const ndindex_t& ndindex) const {
            for (index_t d = 0; d < Dim; ++d) {
                if (ndindex[d] <= 0 || ndindex[d] >= this->nr_m[d] - 1) {
                    return true;
                }
            }
            return false;
        }

        Kokkos::View<size_t*> elementIndices;
    };

}  // namespace ippl

#include "FEM/LagrangeSpace.hpp"

#endif
