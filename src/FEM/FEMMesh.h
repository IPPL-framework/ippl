// Class FEMMesh
//   The FEMMesh class. This is a class representing a finite element mesh.
#ifndef IPPL_FEMMESH_H
#define IPPL_FEMMESH_H

#include "FEM/Element.h"
#include "Meshes/Mesh.h"

namespace ippl {

class FEMMesh : Mesh {
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

#endif  // IPPL_FEMMESH_H