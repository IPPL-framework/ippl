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
    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints>
    class FiniteElementSpace {
    public:
        typedef Vector<std::size_t, Dim> index_vector_t;
        typedef Vector<std::size_t, NumElementVertices> vertex_vector_t;

        /**
         * @brief Construct a new Finite Element Space object with a given mesh and
         * quadrature rule.
         *
         * @param mesh Mesh that represents the domain of the problem
         * @param ref_element Pointer to singleton instance of the reference element
         * @param quadrature Pointer to the singleton instance of the quadrature rule
         */
        FiniteElementSpace(const Mesh<T, Dim>& mesh,
                           const Element<T, Dim, Dim, NumElementVertices>& ref_element,
                           const Quadrature<T, NumIntegrationPoints>& quadrature);

        /**
         * @brief Eveluate the load vector at the given index.
         *
         * @param j The index of the load vector
         * @return T The value of the load vector at the given index
         */
        virtual T evaluateLoadVector(const std::size_t& j) const = 0;

        /**
         * @brief Evaluate the stiffness matrix at the given indices.
         *
         * @param i The row index of the stiffness matrix
         * @param j The column index of the stiffness matrix
         * @return T The value of the stiffness matrix at the given indices
         */
        virtual T evaluateStiffnessMatrix(const std::size_t& i, const std::size_t& j) const = 0;

        /**
         * @brief Get the dimension indices for element object
         *
         * @param element_index
         * @return Vector<std::size_t, Dim>
         */
        virtual index_vector_t getDimensionIndicesForElement(
            const std::size_t& element_index) const = 0;

        /**
         * @brief Get the vertices for an element given the element index.
         *
         * @param element_index The index of the element.
         * @return Vector<std::size_t, NumVertices>
         */
        vertex_vector_t getGlobalVerticesForElement(const std::size_t& element_index) const;

        /**
         * @brief Get the vertices for an elment given the element indices in each dimension of the
         * mesh.
         *
         * @param element_indices The indices of the element in each dimension of the mesh.
         * @return Vector<std::size_t, NumVertices>
         */
        virtual vertex_vector_t getGlobalVerticesForElement(
            const index_vector_t& element_indices) const = 0;

    protected:
        const Mesh<T, Dim>& mesh_m;
        const Element<T, Dim, Dim, NumElementVertices>& ref_element_m;
        const Quadrature<T, NumIntegrationPoints>& quadrature_m;
    };

}  // namespace ippl

#include "FEM/FiniteElementSpace.hpp"

#endif