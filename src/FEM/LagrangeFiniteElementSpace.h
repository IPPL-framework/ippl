// Class LagrangeFiniteElementSpace
//   This class represents a Lagrangian finite element space S_p^0 with degree
//   p.

#ifndef IPPL_LAGRANGEFINITEELEMENTSPACE_H
#define IPPL_LAGRANGEFINITEELEMENTSPACE_H

#include "FEM/FiniteElementSpace.h"

namespace ippl {

    template <unsigned Dim, Element<Dim> ElementType>
    class LagrangeFiniteElementSpace : FiniteElementSpace<Dim, ElementType> {};

}  // namespace ippl

#endif