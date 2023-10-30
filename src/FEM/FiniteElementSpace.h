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
              unsigned NumDoFs>
    class FiniteElementSpace {
    public:
        ///////////////////////////////////////////////////////////////////////
        // Mesh types /////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        // A point in the global coordinate system
        typedef Vector<T, Dim> point_t;

        // A gradient vector in the global coordinate system
        typedef Vector<T, Dim> gradient_vec_t;

        // An index of a vertex in the mesh
        typedef std::size_t mesh_vertex_index_t;

        // A vector of vertex indices of the mesh
        typedef Vector<mesh_vertex_index_t, NumElementVertices> mesh_vertex_vec_t;

        // A vector with the position of the vertex in the mesh in each dimension
        typedef Vector<mesh_vertex_index_t, Dim> mesh_vertex_pos_t;

        ///////////////////////////////////////////////////////////////////////
        // Element types //////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        // A point in the local coordinate system of an element
        typedef Vector<T, Dim> local_point_t;

        // An index of an element in the mesh
        typedef std::size_t element_index_t;

        // A vector with the position of the element in the mesh in each dimension
        typedef Vector<element_index_t, Dim> element_pos_t;

        ///////////////////////////////////////////////////////////////////////
        // DoF types //////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        // An index of a global degree of freedom of the finite element space
        typedef std::size_t global_dof_index_t;

        // An index of a local degree of freedom of an element
        typedef std::size_t local_dof_index_t;

        // A vector of storing a value for all degrees of freedom
        typedef Vector<T, NumDoFs> dof_val_vec_t;

        ///////////////////////////////////////////////////////////////////////
        // Constructors ///////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        FiniteElementSpace(const Mesh<T, Dim>& mesh,
                           const Element<T, Dim, NumElementVertices>& ref_element,
                           const Quadrature<T, NumIntegrationPoints>& quadrature);

        ///////////////////////////////////////////////////////////////////////
        /// Mesh and Element operations ///////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        virtual mesh_vertex_pos_t getMeshVertexPositionFromIndex(
            const mesh_vertex_index_t& vertex_index) const = 0;

        virtual element_pos_t getElementPositionFromIndex(
            const element_index_t& element_index) const = 0;

        mesh_vertex_vec_t getMeshVerticesForElement(const element_index_t& element_index) const;

        virtual mesh_vertex_vec_t getMeshVerticesForElement(
            const element_pos_t& element_indices) const = 0;

        ///////////////////////////////////////////////////////////////////////
        /// Degree of Freedom operations //////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////
        /// Basis functions and gradients /////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        virtual T evaluateBasis(const local_dof_index_t& local_vertex_index,
                                const local_point_t& local_coordinates) const = 0;

        virtual gradient_vec_t evaluateBasisGradient(
            const local_dof_index_t& local_vertex_index,
            const local_point_t& local_coordinates) const = 0;

        ///////////////////////////////////////////////////////////////////////
        /// Assembly operations ///////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        virtual void evaluateLoadVector(dof_val_vec_t& b) const = 0;

        virtual void evaluateAx(const dof_val_vec_t& x, dof_val_vec_t& Ax) const = 0;

    protected:
        const Mesh<T, Dim>& mesh_m;
        const Element<T, Dim, NumElementVertices>& ref_element_m;
        const Quadrature<T, NumIntegrationPoints>& quadrature_m;
    };

}  // namespace ippl

#include "FEM/FiniteElementSpace.hpp"

#endif