//
// Class Element
//   The Element base class. This is an abstract base class representing an
//   element type for finite element methods.
//
#ifndef IPPL_ELEMENT_H
#define IPPL_ELEMENT_H

#include "Types/Vector.h"

namespace ippl {

template <unsigned Dim>
class Element {
   public:
    template <unsigned NumNodes>
    virtual Vector<Vector<unsigned, Dim>, NumNodes> getLocalNodes() = 0;

    void localToGlobal();

    void globalToLocal();

    void getJacobian();
};

}  // namespace ippl

#include "FEM/Element.hpp"

#endif