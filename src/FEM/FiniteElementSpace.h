// FiniteElementSpace Class
//   This class is the interface for the solvers with the finite element methods implemented in IPPL
//   The constructor takes in an IPPL mesh, a quadrature rule and an optional degree (with the
//   default being 1 at the moment) The class is templated on the floating point type T and the
//   dimension 'Dim' (1 for 1D, 2 for 2D, 3 for 3D)
//   When calling the constructor the class first tries to create a FiniteElementMesh using the IPPL
//   Mesh and select the best supported element type for the mesh. If the mesh is not supported for
//   FEM, an exception is thrown. The class also takes in a quadrature rule which is used to
//   evaluate the stiffness matrix and the load vector. Setting the degree of the finite element
//   space will change the degree of the quadrature rule.

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

    // template <typename T>
    //  concept IsQuadrature = std::is_base_of<Quadrature, T>::value;

    template <typename T, unsigned Dim, unsigned NumElementDOFs, typename QuadratureType,
              typename FieldLHS, typename FieldRHS>
    // requires IsElement<QuadratureType>
    class FiniteElementSpace {
    public:
        static constexpr unsigned dim                = Dim;
        static constexpr unsigned numElementVertices = calculateNumElementVertices(Dim);
        static constexpr unsigned numElementDOFs     = NumElementDOFs;

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

        FiniteElementSpace(const Mesh<T, Dim>& mesh, const ElementType& ref_element,
                           const QuadratureType& quadrature);

        ///////////////////////////////////////////////////////////////////////
        /// Mesh and Element operations ///////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        std::size_t numElements() const;

        std::size_t numElementsInDim(const index_t& dim) const;

        ndindex_t getMeshVertexNDIndex(const index_t& vertex_index) const;

        index_t getMeshVertexIndex(const ndindex_t& vertex_nd_index) const;

        ndindex_t getElementNDIndex(const index_t& elementIndex) const;

        mesh_element_vertex_index_vec_t getElementMeshVertexIndices(
            const ndindex_t& elementNDIndex) const;

        mesh_element_vertex_ndindex_vec_t getElementMeshVertexNDIndices(
            const ndindex_t& elementNDIndex) const;

        mesh_element_vertex_point_vec_t getElementMeshVertexPoints(
            const ndindex_t& elementNDIndex) const;

        ///////////////////////////////////////////////////////////////////////
        /// Degree of Freedom operations //////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        virtual std::size_t numGlobalDOFs(const unsigned& nghosts = 0) const = 0;

        // virtual point_t getCoordsOfDOF(const index_t& dof_index) const = 0;

        virtual index_t getLocalDOFIndex(const index_t& elementIndex,
                                         const index_t& globalDOFIndex) const = 0;

        virtual index_t getGlobalDOFIndex(const index_t& elementIndex,
                                          const index_t& localDOFIndex) const = 0;

        virtual Vector<index_t, NumElementDOFs> getLocalDOFIndices() const = 0;

        virtual Vector<index_t, NumElementDOFs> getGlobalDOFIndices(
            const index_t& elementIndex) const = 0;

        virtual Vector<ndindex_t, NumElementDOFs> getGlobalDOFNDIndices(
            const index_t& elementIndex) const = 0;

        ///////////////////////////////////////////////////////////////////////
        /// Basis functions and gradients /////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        virtual T evaluateRefElementBasis(const index_t& localDOF,
                                          const point_t& localPoint) const = 0;

        virtual gradient_vec_t evaluateRefElementBasisGradient(const index_t& localDOF,
                                                               const point_t& localPoint) const = 0;

        ///////////////////////////////////////////////////////////////////////
        /// Assembly operations ///////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        virtual FieldLHS evaluateAx(
            const FieldLHS& field,
            const std::function<T(const index_t&, const index_t&,
                                  const Vector<Vector<T, Dim>, NumElementDOFs>&)>& evalFunction)
            const = 0;

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