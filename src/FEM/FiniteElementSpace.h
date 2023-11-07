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

namespace ippl {

    /**
     * @brief This abstract class represents an finite element space.
     *
     * @tparam T The floating point type used
     * @tparam Dim The geometric dimension of the space
     * @tparam NumElementVertices The number of vertices of the element
     * @tparam NumIntegrationPoints The number of integration nodes of the quadrature rule
     */
    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints,
              unsigned NumElementDOFs>
    class FiniteElementSpace {
    public:
        // An unsigned integer number representing an index
        typedef std::size_t index_t;  // look at ippl::Index

        // A vector with the position of the element in the mesh in each dimension
        typedef Vector<index_t, Dim> nd_index_t;  // TODO look ad NDINDEX

        // A point in the global coordinate system
        typedef Vector<T, Dim> point_t;

        // A gradient vector in the global coordinate system
        typedef Vector<T, Dim> gradient_vec_t;

        // A vector of vertex indices of the mesh
        typedef Vector<index_t, NumElementVertices> mesh_element_vertex_vec_t;

        ///////////////////////////////////////////////////////////////////////
        // Constructors ///////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        FiniteElementSpace(const Mesh<T, Dim>& mesh,
                           const Element<T, Dim, NumElementVertices>& ref_element,
                           const Quadrature<T, NumIntegrationPoints>& quadrature);

        ///////////////////////////////////////////////////////////////////////
        /// Assembly operations ///////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        template <unsigned NumGlobalDOFs>
        virtual void evaluateAx(const Vector<T, NumGlobalDOFs>& x,
                                Vector<T, NumGlobalDOFs>& z) const = 0;

        template <unsigned NumGlobalDOFs>
        virtual void evaluateLoadVector(Vector<T, NumGlobalDOFs>& b) const = 0;

    protected:
        ///////////////////////////////////////////////////////////////////////
        /// Mesh and Element operations ///////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        std::size_t numElements() const;

        nd_index_t getMeshVertexNDIndex(const index_t& vertex_index) const;

        nd_index_t getElementNDIndex(const index_t& element_index) const;

        mesh_element_vertex_vec_t getElementMeshVertices(const index_t& element_index) const;

        mesh_element_vertex_vec_t getElementMeshVertices(const nd_index_t& element_indices) const;

        ///////////////////////////////////////////////////////////////////////
        /// Degree of Freedom operations //////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        std::size_t numGlobalDOFs() const;

        virtual point_t getCoordsOfDOF(const index_t& dof_index) const = 0;

        virtual index_t getLocalDOFIndex(const index_t& global_dof_index,
                                         const index_t& element_index) const = 0;

        virtual index_t getGlobalDOFIndex(const index_t& local_dof_index,
                                          const index_t& element_index) const = 0;

        virtual Vector<index_t, NumElementDOFs> getLocalDOFIndices() const = 0;

        virtual Vector<index_t, NumElementDOFs> getGlobalDOFIndices(
            const index_t& element_index) const = 0;

        ///////////////////////////////////////////////////////////////////////
        /// Basis functions and gradients /////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        virtual T evaluateRefElementBasis(const index_t& localDOF,
                                          const point_t& localPoint) const = 0;

        virtual gradient_vec_t evaluateRefElementBasisGradient(const index_t& localDOF,
                                                               const point_t& localPoint) const = 0;

        ///////////////////////////////////////////////////////////////////////
        /// Member variables //////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        const Mesh<T, Dim>& mesh_m;
        const Element<T, Dim, NumElementVertices>& ref_element_m;
        const Quadrature<T, NumIntegrationPoints>& quadrature_m;
    };

}  // namespace ippl

#include "FEM/FiniteElementSpace.hpp"

#endif