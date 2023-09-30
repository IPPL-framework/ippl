// Class FEMesh
//   The FEMesh class. This is a class representing a finite element mesh.
//   It is templated on the number of dimensions and the Element type.
#ifndef IPPL_FEMESH_H
#define IPPL_FEMESH_H

#include "FEM/Element.h"
#include "Meshes/Mesh.h"

namespace ippl {

    // template <unsigned Dim, Element<Dim> ElementType>
    // /**
    //  * @brief Helper class to iterate over elements in a mesh.
    //  *
    //  * @tparam Dim The number of dimensions of the mesh
    //  * @tparam ElementType The element type for the finite element mesh
    //  */
    // class MeshElementIterator {
    // public:
    //     MeshElementIterator::MeshElementIterator() = 0;

    //     // Dereference operatorDim
    //     ElementType& operator*() = 0;

    //     // Pre-increment operator
    //     MeshElementIterator& operator++() = 0;

    //     // Post-increment operator
    //     MeshElementIterator operator++(int) = 0;

    //     // Comparison operator
    //     bool operator==(const MeshElementIterator& other) const = 0;

    //     // Inequality operator
    //     bool operator!=(const MeshElementIterator& other) const = 0;

    // private:
    //     std::size_t current_index_m;
    //     ElementType* current_m;
    // };

    template <typename T, unsigned Dim, Element<Dim> ElementType>
    class FEMesh : Mesh<T, Dim> {
    public:
        virtual std::size_t getNumberOfVertices() = 0;

        virtual std::size_t getNumberOfElements() = 0;

        virtual ElementType getElement(std::size_t element_index) = 0;

        /**
         * @brief Get all the nodes in the mesh
         *
         * @return std::vector<unsigned int>
         */
        virtual std::vector<Vector<T, Dim>> getVertices() = 0;
    };

}  // namespace ippl

#endif  // IPPL_FEMESH_H