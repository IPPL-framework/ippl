// FiniteElementSpace Class
//   This class is the interface for the solvers with the finite element methods implemented in
//   IPPL. The constructor takes in an IPPL mesh, a reference element and a quadrature rule. The
//   class is templated on the floating point type T, the dimension 'Dim' (1 for 1D, 2 for 2D, 3 for
//   3D), the number of degrees of freedom per element 'NumElementDOFs', the quadrature rule type
//   'QuadratureType', the left hand side field type 'FieldLHS' and the right hand side field type
//   'FieldRHS'.

#ifndef IPPL_FEMSPACE_H
#define IPPL_FEMSPACE_H

#include "Types/ViewTypes.h"

#include "FEM/Elements/Element.h"
#include "FEM/Quadrature/Quadrature.h"
#include "Meshes/Mesh.h"

constexpr unsigned calculateNumElementVertices(unsigned Dim) {
    return 1 << Dim;  // 2^Dim
}

namespace ippl {

    /**
     * @brief The FiniteElementSpace class handles the mesh index mapping to vertices and elements
     * and is the base class for other FiniteElementSpace classes (e.g. LagrangeSpace)
     *
     * @tparam T The floating point type
     * @tparam Dim The dimension of the mesh (same dimension as the space)
     * @tparam NumElementDOFs The number of degrees of freedom per element
     * @tparam QuadratureType The type of the quadrature rule (e.g. MidpointQuadrature,
     * GaussJacobiQuadrature)
     * @tparam FieldLHS The type of the left hand side field
     * @tparam FieldRHS The type of the right hand side field (can be the same as FieldLHS)
     */
    template <typename T, unsigned Dim, unsigned NumElementDOFs, typename QuadratureType,
              typename FieldLHS, typename FieldRHS>
    // requires IsElement<QuadratureType>
    class FiniteElementSpace {
    public:
        static constexpr unsigned dim = Dim;

        // the number of mesh vertices per element (not necessarily the same as degrees of freedom,
        // e.g. a 2D element has 4 vertices)
        static constexpr unsigned numElementVertices = calculateNumElementVertices(Dim);

        static constexpr unsigned numElementDOFs = NumElementDOFs;

        typedef Element<T, Dim, numElementVertices> ElementType;

        // An unsigned integer number representing an index
        typedef std::size_t index_t;  // look at ippl::Index

        // A vector with the position of the element in the mesh in each dimension
        typedef Vector<index_t, Dim> ndindex_t;

        // A point in the global coordinate system
        typedef Vector<T, Dim> point_t;

        // A gradient vector in the global coordinate system
        typedef Vector<T, Dim> gradient_vec_t;

        // A vector of vertex indices of the mesh
        typedef Vector<index_t, numElementVertices> mesh_element_vertex_index_vec_t;

        typedef Vector<ndindex_t, numElementVertices> mesh_element_vertex_ndindex_vec_t;

        typedef Vector<point_t, numElementVertices> mesh_element_vertex_point_vec_t;

        ///////////////////////////////////////////////////////////////////////
        // Constructors ///////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Construct a new FiniteElementSpace object
         *
         * @param mesh The mesh object
         * @param ref_element The reference element object
         * @param quadrature The quadrature rule object
         */
        FiniteElementSpace(const Mesh<T, Dim>& mesh, const ElementType& ref_element,
                           const QuadratureType& quadrature);

        ///////////////////////////////////////////////////////////////////////
        /// Mesh and Element operations ///////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Get the number of elements in the mesh of the space
         *
         * @return std::size_t - unsigned integer number of elements
         */
        std::size_t numElements() const;

        /**
         * @brief Get the number of elements in a given dimension
         *
         * @param dim index_t (std::size_t) - representing the dimension
         *
         * @return std::size_t - unsigned integer number of elements in the given dimension
         */
        std::size_t numElementsInDim(const index_t& dim) const;

        /**
         * @brief Get the NDIndex of a mesh vertex.
         *
         * @param vertex_index index_t (std::size_t) - The index of the vertex
         *
         * @return ndindex_t (Vector<std::size_t, Dim>) - Returns the NDIndex (vector of indices for
         * each dimension)
         */
        ndindex_t getMeshVertexNDIndex(const index_t& vertex_index) const;

        /**
         * @brief Get the global index of a mesh vertex given its NDIndex
         *
         * @param vertex_nd_index ndindex_t (Vector<std::size_t, Dim>) - The NDIndex of the vertex
         * (vector of indices for each dimension).
         *
         * @return index_t (std::size_t) - unsigned integer index of the mesh vertex
         */
        index_t getMeshVertexIndex(const ndindex_t& vertex_nd_index) const;

        /**
         * @brief Get the NDIndex (vector of indices for each dimension) of a mesh element.
         *
         * @param elementIndex ndindex_t (Vector<std::size_t, Dim>) - The index of the element
         *
         * @return ndindex_t (Vector<std::size_t, Dim>) - vector of indices for each dimension
         */
        ndindex_t getElementNDIndex(const index_t& elementIndex) const;

        /**
         * @brief Get the global index of a mesh element given the NDIndex.
         *
         * @param ndindex ndindex_t (Vector<std::size_t, Dim>) - vector of indices for each direction
         *
         * @return index_t - the index of the element
         */
        index_t getElementIndex(const ndindex_t& ndindex) const;

        /**
         * @brief Get all the global vertex indices of an element (given by its NDIndex).
         *
         * @param elementNDIndex The NDIndex of the element
         *
         * @return mesh_element_vertex_index_vec_t (Vector<std::size_t, numElementVertices>) -
         * vector of vertex indices
         */
        mesh_element_vertex_index_vec_t getElementMeshVertexIndices(
            const ndindex_t& elementNDIndex) const;

        /**
         * @brief Get all the NDIndices of the vertices of an element (given by its NDIndex).
         *
         * @param elementNDIndex The NDIndex of the element
         *
         * @return mesh_element_vertex_ndindex_vec_t (Vector<Vector<std::size_t, Dim>,
         * numElementVertices>) - vector of vertex NDIndices
         */
        mesh_element_vertex_ndindex_vec_t getElementMeshVertexNDIndices(
            const ndindex_t& elementNDIndex) const;

        /**
         * @brief Get all the global vertex points of an element (given by its NDIndex).
         *
         * @param elementNDIndex The NDIndex of the element
         *
         * @return mesh_element_vertex_point_vec_t (Vector<Vector<T, Dim>, numElementVertices>) -
         */
        mesh_element_vertex_point_vec_t getElementMeshVertexPoints(
            const ndindex_t& elementNDIndex) const;

        ///////////////////////////////////////////////////////////////////////
        /// Degree of Freedom operations //////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Get the number of global degrees of freedom in the space
         *
         * @return std::size_t - unsigned integer number of global degrees of freedom
         */
        virtual std::size_t numGlobalDOFs() const = 0;

        /**
         * @brief Get the elements local DOF from the element index and global DOF
         * index
         *
         * @param elementIndex index_t (std::size_t) - The index of the element
         * @param globalDOFIndex index_t (std::size_t) - The global DOF index
         *
         * @return index_t (std::size_t) - The local DOF index
         */
        virtual index_t getLocalDOFIndex(const index_t& elementIndex,
                                         const index_t& globalDOFIndex) const = 0;

        /**
         * @brief Get the global DOF index from the element index and local DOF
         *
         * @param elementIndex index_t (std::size_t) - The index of the element
         * @param localDOFIndex index_t (std::size_t) - The local DOF index
         *
         * @return index_t (std::size_t) - The global DOF index
         */
        virtual index_t getGlobalDOFIndex(const index_t& elementIndex,
                                          const index_t& localDOFIndex) const = 0;

        /**
         * @brief Get the local DOF indices (vector of local DOF indices)
         * They are independent of the specific element because it only depends on
         * the reference element type
         *
         * @return Vector<index_t, NumElementDOFs> - The local DOF indices
         */
        virtual Vector<index_t, NumElementDOFs> getLocalDOFIndices() const = 0;

        /**
         * @brief Get the global DOF indices (vector of global DOF indices) of an element
         *
         * @param elementIndex index_t (std::size_t) - The index of the element
         *
         * @return Vector<index_t, NumElementDOFs> - The global DOF indices
         */
        virtual Vector<index_t, NumElementDOFs> getGlobalDOFIndices(
            const index_t& elementIndex) const = 0;

        /**
         * @brief Get the global DOF NDIndices (vector of global DOF NDIndices) of an element
         *
         * @param elementIndex index_t (std::size_t) - The index of the element
         *
         * @return Vector<ndindex_t, NumElementDOFs> - The global DOF NDIndices
         */
        virtual Vector<ndindex_t, NumElementDOFs> getGlobalDOFNDIndices(
            const index_t& elementIndex) const = 0;

        ///////////////////////////////////////////////////////////////////////
        /// Basis functions and gradients /////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Evaluate the shape function of a local degree of freedom at a given point in the
         * reference element
         *
         * @param localDOF index_t (std::size_t) - The local degree of freedom index
         * @param localPoint point_t (Vector<T, Dim>) - The point in the reference element
         *
         * @return T - The value of the shape function at the given point
         */
        virtual T evaluateRefElementShapeFunction(const index_t& localDOF,
                                                  const point_t& localPoint) const = 0;

        /**
         * @brief Evaluate the gradient of the shape function of a local degree of freedom at a
         * given point in the reference element
         *
         * @param localDOF index_t (std::size_t) - The local degree of freedom index
         * @param localPoint point_t (Vector<T, Dim>) - The point in the reference element
         *
         * @return gradient_vec_t (Vector<T, Dim>) - The gradient of the shape function at the given
         * point
         */
        virtual gradient_vec_t evaluateRefElementShapeFunctionGradient(
            const index_t& localDOF, const point_t& localPoint) const = 0;

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
        virtual FieldLHS evaluateAx(
            const FieldLHS& field,
            const std::function<T(const index_t&, const index_t&,
                                  const Vector<Vector<T, Dim>, NumElementDOFs>&)>& evalFunction)
            const = 0;

        /**
         * @brief Assemble the load vector b of the system Ax = b
         *
         * @param rhs_field The field to set with the load vector
         * @param f The source function
         *
         * @return FieldRHS - The RHS field containing b
         */
        virtual void evaluateLoadVector(FieldRHS& rhs_field,
                                        const std::function<T(const point_t&)>& f) const = 0;

        ///////////////////////////////////////////////////////////////////////
        /// Member variables //////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        const Mesh<T, Dim>& mesh_m;
        const ElementType& ref_element_m;
        const QuadratureType& quadrature_m;
    };

}  // namespace ippl

#include "FEM/FiniteElementSpace.hpp"

#endif
