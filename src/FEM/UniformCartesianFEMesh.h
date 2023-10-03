// Class UniformCartesianFEMesh
//   This class represents a Cartesian finite element mesh.

// https://en.wikipedia.org/wiki/Regular_grid#/media/File:Cartesian_grid.svg

#ifndef IPPL_UNIFORMCARTESIANFEMESH_H
#define IPPL_UNIFORMCARTESIANFEMESH_H

#include "FEM/Elements/HexahedralElement.h"
#include "FEM/Elements/LineElement.h"
#include "FEM/Elements/QuadrilateralElement.h"
#include "FEM/FEMesh.h"
#include "Meshes/UniformCartesian.h"

namespace ippl {

    template <typename T, unsigned Dim>
    class UniformCartesianFEMesh : public FEMesh<T, Dim, HexahedralElement> {
    public:
        typedef typename Mesh<T, Dim>::vector_type vector_type;
        typedef Cell DefaultCentering;

        static_assert(Dim >= 1 && Dim <= 3,
                      "UniformCartesianFEMesh only supports 1D, 2D and 3D meshes");

        using ElementType = std::conditional_t<
            Dim == 1, LineElement,
            std::conditional_t<Dim == 2, QuadrilateralElement,
                               std::conditional_t<Dim == 3, HexahedralElement, void>>>;

        UniformCartesianFEMesh(const UniformCartesian<T, Dim>& mesh);

        virtual ElementType getElement(std::size_t element_index) = 0;
    };
}  // namespace ippl

#include "FEM/UniformCartesianFEMesh.hpp"

#endif  // IPPL_UNIFORMCARTESIANFEMESH_H