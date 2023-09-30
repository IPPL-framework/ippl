// FiniteElementSpace Class
//
#ifndef IPPL_FEMSPACE_H
#define IPPL_FEMSPACE_H

#include "FEM/Element.h"
#include "FEM/FEMesh.h"
#include "FEM/Quadrature.h"

namespace ippl {

    template <unsigned Dim, Element<Dim> ElementType>
    class FiniteElementSpace {
    public:
        void setDegree(unsigned degree);
        unsigned getDegree() const;

        virtual Element<Dim> getElement(std::size_t element_index) = 0;

    protected:
        FEMesh<Dim, ElementType> mesh;
        Quadrature quadrature;
    };

}  // namespace ippl

#endif