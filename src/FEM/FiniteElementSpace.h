// FiniteElementSpace Class
//
#ifndef IPPL_FEMSPACE_H
#define IPPL_FEMSPACE_H

#include "FEM/Element.h"
#include "FEM/FEMesh.h"
#include "FEM/Quadrature.h"

namespace ippl {

    template <typename T, unsigned Dim, Element<Dim, 8> ElementType>
    class FiniteElementSpace {
    public:
        FiniteElementSpace(Mesh<T, Dim>& mesh, Quadrature& quadrature);

        virtual void setDegree(unsigned degree);

        virtual unsigned getDegree() const;

        // virtual Element<Dim> getElement(std::size_t element_index) const = 0;

        virtual T evaluateStiffnessMatrix(std::size_t i, std::size_t j) const = 0;

        virtual T evaluateAx(std::size_t i) const = 0;

        virtual T evaluateLoadVector(std::size_t i) const = 0;

    protected:
        FEMesh<Dim, ElementType> mesh;
        Quadrature quadrature;
    };

}  // namespace ippl

#endif