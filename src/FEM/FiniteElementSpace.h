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

#include "FEM/Element.h"
#include "FEM/FEMesh.h"
#include "FEM/Quadrature.h"
#include "Meshes/Mesh.h"
#include "Meshes/UniformCartesian.h"

namespace ippl {

    template <typename T, unsigned Dim>
    class FiniteElementSpace {
    public:
        /**
         * @brief Construct a new Finite Element Space object with a given mesh and
         * quadrature rule.
         *
         * @param mesh Mesh
         * @param quadrature quadrature rule
         */
        template <typename MeshType, typename QuadratureType>
        FiniteElementSpace(const MeshType& mesh, QuadratureType& quadrature, unsigned degree = 1);

        /**
         * @brief Construct a new Finite Element Space object with a given uniform Cartesian mesh
         * and quadrature rule.
         *
         * @param mesh Uniform Cartesian mesh
         * @param quadrature Quadrature rule
         */
        template <typename QuadratureType>
        FiniteElementSpace(const UniformCartesian<T, Dim>& mesh, QuadratureType& quadrature,
                           unsigned degree = 1);

        ~FiniteElementSpace();

        /**
         * @brief Set the degree of the finite element space
         *
         * @param degree
         */
        virtual void setDegree(unsigned degree);

        /**
         * @brief Get the degree of the finite element space
         *
         * @return unsigned
         */
        virtual unsigned getDegree() const;

        // virtual Element<Dim> getElement(std::size_t element_index) const = 0;

        /**
         * @brief Evaluate the stiffness matrix A_ij for the element with
         * index row i and column j.
         *
         * @param i row index
         * @param j column index
         * @return T
         */
        virtual T evaluateA(const std::size_t& i, const std::size_t& j) const;

        /**
         * @brief Evaluate the load vector b_j for the element with index j.
         *
         * @param j index
         * @return T Returns the value of the load vector at index j
         */
        virtual T evaluateLoadVector(const std::size_t& j) const;

        /**
         * @brief Evaluate the product Ax for the element with index j.
         *
         * @param j index
         * @param x_j value of x at index j
         * @tparam Func function type that returns a value of type T with argument of type
         * std::size_t
         * @return T
         */
        template <typename Func>
        virtual T evaluateAx(const std::size_t& j, const Func& x) const;

    protected:
        unsigned degree_m;
        FEMesh* femesh_m;  // TODO use safe pointer
        Quadrature& quadrature_rule_m;
    };

}  // namespace ippl

#include "FEM/FiniteElementSpace.hpp"

#endif