//
// Class Hexahedron
//   The Hexahedron class. This is a class representing a hexahedron element
//   for finite element methods.
#ifndef IPPL_HEXAHEDRON_H
#define IPPL_HEXAHEDRON_H

#include "FEM/Element.h"

namespace ippl {

class Hexahedron : public Element<3> {};

Vector<Vector<unsigned, 3>, 8> getNodes();

}  // namespace ippl

#include "FEM/Hexahedron.hpp"

#endif