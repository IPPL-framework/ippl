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
     * @tparam GeometricDim The geometric dimension of the space
     * @tparam TopologicalDim The topological dimension of the space
     * @tparam NumElementVertices The number of vertices of the element
     * @tparam NumIntegrationPoints The number of integration nodes of the quadrature rule
     */
    template <typename T, unsigned Dim, unsigned NumElementVertices, unsigned NumIntegrationPoints>
    class FiniteElementSpace {
    public:
        /**
         * @brief Construct a new Finite Element Space object with a given mesh and
         * quadrature rule.
         *
         * @param mesh Mesh that represents the domain of the problem
         * @param ref_element Pointer to singleton instance of the reference element
         * @param quadrature Pointer to the singleton instance of the quadrature rule
         */
        FiniteElementSpace(const Mesh<T, Dim>* mesh,
                           const Element<T, Dim, Dim, NumElementVertices>* ref_element,
                           const Quadrature<T, NumIntegrationPoints>* quadrature);

        // virtual matrix_type getStiffnessMatrix() const = 0;

        // virtual matrix_type getLoadVector() const = 0;

        // virtual T evaluateA(const std::size_t& i, const std::size_t& j) const = 0;

        // virtual T evaluateLoadVector(const std::size_t& j) const = 0;

        // template <typename Func>
        // virtual T evaluateAx(const std::size_t& j, const Func& x) const = 0;

    protected:
        const Mesh<T, Dim>* mesh_m;
        const Element<T, Dim, Dim, NumElementVertices>* ref_element_m;
        const Quadrature<T, NumIntegrationPoints>* quadrature_m;
    };

}  // namespace ippl

#include "FEM/FiniteElementSpace.hpp"

#endif