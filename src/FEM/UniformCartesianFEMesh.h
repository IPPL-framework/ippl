// Class UniformCartesianFEMesh
//   This class represents a Cartesian finite element mesh.

// https://en.wikipedia.org/wiki/Regular_grid#/media/File:Cartesian_grid.svg

#ifndef IPPL_UNIFORMCARTESIANFEMESH_H
#define IPPL_UNIFORMCARTESIANFEMESH_H

#include "FEM/FEMesh.h"
#include "FEM/Hexahedron.h"
#include "Meshes/UniformCartesian.h"

namespace ippl {

    // template <typename T, unsigned Dim>
    // /**
    //  * @brief Helper class to iterator over the elements in the Uniform Cartesian mesh
    //  *
    //  * @tparam T floating point number type used
    //  * @tparam Dim number of dimensions
    //  */
    // class HexahedronIterator : public ElementIterator {
    // public:
    //     HexahedronIterator();

    //     // Dereference operator
    //     Hexahedron& operator*();

    //     // Pre-increment operator
    //     HexahedronIterator& operator++();

    //     // Post-increment operator
    //     HexahedronIterator operator++(int);

    //     // Comparison operator
    //     bool operator==(const HexahedronIterator& other) const;

    //     // Inequality operator
    //     bool operator!=(const HexahedronIterator& other) const;

    // private:
    //     Hexahedron* current_m;
    // };

    template <typename T, unsigned Dim>
    class UniformCartesianFEMesh : public FEMesh<T, Dim, Hexahedron> {
    public:
        typedef typename Mesh<T, Dim>::vector_type vector_type;
        typedef Cell DefaultCentering;

        UniformCartesianFEMesh();

        UniformCartesianFEMesh(UniformCartesian<T, Dim> mesh);

        virtual ElementType getElement(std::size_t element_index) = 0;
    };
}  // namespace ippl

#include "FEM/UniformCartesianFEMesh.hpp"

#endif  // IPPL_UNIFORMCARTESIANFEMESH_H