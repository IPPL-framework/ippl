// Class FEMMesh
//   The FEMMesh class. This is a class representing a finite element mesh.
//   It is templated on the number of dimensions and the Element type.
#ifndef IPPL_FEMMESH_H
#define IPPL_FEMMESH_H

#include "FEM/Element.h"
#include "Meshes/Mesh.h"

namespace ippl {

template <typename T, unsigned Dim, Element<Dim> ElementType>
class FEMMesh : Mesh<T, Dim> {
   public:
    FEMMesh::FEMMesh();

    /**
     * Get the elements in the mesh
     *
     * @return std::vector<Element>
     */
    std::vector<Element> getElements();

    /**
     * @brief Get the nodes in the mesh
     *
     * @return std::vector<unsigned int>
     */
    std::vector<unsigned int> getNodes();
};

}  // namespace ippl

#include "FEM/FEMMesh.hpp"

#endif  // IPPL_FEMMESH_H