//
// Class Hexahedron
//   The Hexahedron class. This is a class representing a hexahedron element
//   for finite element methods.
#ifndef IPPL_HEXAHEDRON_H
#define IPPL_HEXAHEDRON_H

#include "FEM/Element.h"

namespace ippl {

    class Hexahedron : public Element<3, 8> {
    public:
        Hexahedron(std::size_t global_index, Vector<std::size_t, 8> global_indices_of_vertices);
    };

}  // namespace ippl

#include "FEM/Hexahedron.hpp"

#endif