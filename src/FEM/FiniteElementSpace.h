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

#include "Utility/IpplException.h"

#include "FEM/Elements/Element.h"
#include "FEM/Quadrature/Quadrature.h"
#include "Meshes/UniformCartesian.h"

namespace ippl {

    constexpr unsigned calculateNumElementVertices(unsigned Dim) {
        return 1 << Dim;  // 2^Dim
    }

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
    template <typename T, unsigned Dim, unsigned NumElementDOFs, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    // requires IsElement<QuadratureType>
    class FiniteElementSpace {
    public:
        static constexpr unsigned dim = Dim;

        // the number of mesh vertices per element (not necessarily the same as degrees of freedom,
        // e.g. a 2D element has 4 vertices)
        static constexpr unsigned numElementVertices = calculateNumElementVertices(Dim);
        static constexpr unsigned numElementDOFs     = NumElementDOFs;

        // A vector with the position of the element in the mesh in each dimension
        typedef Vector<size_t, Dim> indices_t;

        // A point in the global coordinate system
        typedef Vector<T, Dim> point_t;

        // A vector of vertex indices of the mesh
        typedef Vector<size_t, numElementVertices> vertex_indices_t;

        typedef Vector<indices_t, numElementVertices> indices_list_t;

        typedef Vector<point_t, numElementVertices> vertex_points_t;

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
        FiniteElementSpace(UniformCartesian<T, Dim>& mesh, ElementType& ref_element,
                           const QuadratureType& quadrature);

        void setMesh(UniformCartesian<T, Dim>& mesh);

        ///////////////////////////////////////////////////////////////////////
        /// Mesh and Element operations ///////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Get the number of elements in the mesh of the space
         *
         * @return size_t - unsigned integer number of elements
         */
        KOKKOS_FUNCTION size_t numElements() const;

        /**
         * @brief Get the number of elements in a given dimension
         *
         * @param dim size_t - representing the dimension
         *
         * @return size_t - unsigned integer number of elements in the given dimension
         */
        KOKKOS_FUNCTION size_t numElementsInDim(const size_t& dim) const;

        /**
         * @brief Get the NDIndex of a mesh vertex.
         *
         * @param vertex_index size_t - The index of the vertex
         *
         * @return indices_t (Vector<size_t, Dim>) - Returns the NDIndex (vector of indices for
         * each dimension)
         */
        KOKKOS_FUNCTION indices_t getMeshVertexNDIndex(const size_t& vertex_index) const;

        /**
         * @brief Get the global index of a mesh vertex given its NDIndex
         *
         * @param vertex_nd_index indices_t (Vector<size_t, Dim>) - The NDIndex of the vertex
         * (vector of indices for each dimension).
         *
         * @return size_t - unsigned integer index of the mesh vertex
         */
        KOKKOS_FUNCTION size_t getMeshVertexIndex(const indices_t& vertex_nd_index) const;

        /**
         * @brief Get the NDIndex (vector of indices for each dimension) of a mesh element.
         *
         * @param elementIndex indices_t (Vector<size_t, Dim>) - The index of the element
         *
         * @return indices_t (Vector<size_t, Dim>) - vector of indices for each dimension
         */
        KOKKOS_FUNCTION indices_t getElementNDIndex(const size_t& elementIndex) const;

        /**
         * @brief Get the global index of a mesh element given the NDIndex.
         *
         * @param ndindex indices_t (Vector<size_t, Dim>) - vector of indices for each direction
         *
         * @return size_t - the index of the element
         */
        KOKKOS_FUNCTION size_t getElementIndex(const indices_t& ndindex) const;

        /**
         * @brief Get all the global vertex indices of an element (given by its NDIndex).
         *
         * @param elementNDIndex The NDIndex of the element
         *
         * @return vertex_indices_t (Vector<size_t, numElementVertices>) -
         * vector of vertex indices
         */
        KOKKOS_FUNCTION vertex_indices_t
        getElementMeshVertexIndices(const indices_t& elementNDIndex) const;

        /**
         * @brief Get all the NDIndices of the vertices of an element (given by its NDIndex).
         *
         * @param elementNDIndex The NDIndex of the element
         *
         * @return indices_list_t (Vector<Vector<size_t, Dim>,
         * numElementVertices>) - vector of vertex NDIndices
         */
        KOKKOS_FUNCTION indices_list_t
        getElementMeshVertexNDIndices(const indices_t& elementNDIndex) const;

        /**
         * @brief Get all the global vertex points of an element (given by its NDIndex).
         *
         * @param elementNDIndex The NDIndex of the element
         *
         * @return vertex_points_t (Vector<Vector<T, Dim>, numElementVertices>) -
         */
        KOKKOS_FUNCTION vertex_points_t
        getElementMeshVertexPoints(const indices_t& elementNDIndex) const;

        ///////////////////////////////////////////////////////////////////////
        /// Degree of Freedom operations //////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Get the number of global degrees of freedom in the space
         *
         * @return size_t - unsigned integer number of global degrees of freedom
         */
        KOKKOS_FUNCTION virtual size_t numGlobalDOFs() const = 0;

        /**
         * @brief Get the elements local DOF from the element index and global DOF
         * index
         *
         * @param elementIndex size_t - The index of the element
         * @param globalDOFIndex size_t - The global DOF index
         *
         * @return size_t - The local DOF index
         */
        KOKKOS_FUNCTION virtual size_t getLocalDOFIndex(const size_t& elementIndex,
                                         const size_t& globalDOFIndex) const = 0;

        /**
         * @brief Get the global DOF index from the element index and local DOF
         *
         * @param elementIndex size_t - The index of the element
         * @param localDOFIndex size_t  - The local DOF index
         *
         * @return size_t - The global DOF index
         */
        KOKKOS_FUNCTION virtual size_t getGlobalDOFIndex(const size_t& elementIndex,
                                                         const size_t& localDOFIndex) const = 0;

        /**
         * @brief Get the local DOF indices (vector of local DOF indices)
         * They are independent of the specific element because it only depends on
         * the reference element type
         *
         * @return Vector<size_t, NumElementDOFs> - The local DOF indices
         */
        KOKKOS_FUNCTION virtual Vector<size_t, NumElementDOFs> getLocalDOFIndices() const = 0;

        /**
         * @brief Get the global DOF indices (vector of global DOF indices) of an element
         *
         * @param elementIndex size_t - The index of the element
         *
         * @return Vector<size_t, NumElementDOFs> - The global DOF indices
         */
        KOKKOS_FUNCTION virtual Vector<size_t, NumElementDOFs> getGlobalDOFIndices(
            const size_t& elementIndex) const = 0;

        ///////////////////////////////////////////////////////////////////////
        /// Basis functions and gradients /////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Evaluate the shape function of a local degree of freedom at a given point in the
         * reference element
         *
         * @param localDOF size_t - The local degree of freedom index
         * @param localPoint point_t (Vector<T, Dim>) - The point in the reference element
         *
         * @return T - The value of the shape function at the given point
         */
        KOKKOS_FUNCTION virtual T evaluateRefElementShapeFunction(
            const size_t& localDOF, const point_t& localPoint) const = 0;

        /**
         * @brief Evaluate the gradient of the shape function of a local degree of freedom at a
         * given point in the reference element
         *
         * @param localDOF size_t - The local degree of freedom index
         * @param localPoint point_t (Vector<T, Dim>) - The point in the reference element
         *
         * @return point_t (Vector<T, Dim>) - The gradient of the shape function at the given
         * point
         */
        KOKKOS_FUNCTION virtual point_t evaluateRefElementShapeFunctionGradient(
            const size_t& localDOF, const point_t& localPoint) const = 0;

        ///////////////////////////////////////////////////////////////////////
        /// Assembly operations ///////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Assemble the load vector b of the system Ax = b
         *
         * @param rhs_field The field to set with the load vector
         * @param f The source function
         *
         * @return FieldRHS - The RHS field containing b
         */
        virtual void evaluateLoadVector(FieldRHS& rhs_field) const = 0;

        ///////////////////////////////////////////////////////////////////////
        /// Member variables //////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        UniformCartesian<T, Dim>& mesh_m;
        ElementType ref_element_m;
        const QuadratureType& quadrature_m;
        Vector<size_t, Dim> nr_m;
        Vector<double, Dim> hr_m;
        Vector<double, Dim> origin_m;
    };

}  // namespace ippl

#include "FEM/FiniteElementSpace.hpp"

#endif
