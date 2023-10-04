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
    class UniformCartesianFEMesh : public FEMesh<T, Dim> {
    public:
        typedef typename Mesh<T, Dim>::vector_type vector_type;
        typedef Cell DefaultCentering;

        static_assert(Dim >= 1 && Dim <= 3,
                      "UniformCartesianFEMesh only supports 1D, 2D and 3D meshes");

        using ElementType = std::conditional_t<
            Dim == 1, LineElement,
            std::conditional_t<Dim == 2, QuadrilateralElement,
                               std::conditional_t<Dim == 3, HexahedralElement, void>>>;

        typedef unsigned NumVertices =
            std::conditional_t<Dim == 1, 2, std::conditional_t<Dim == 2, 4, 8>>

            UniformCartesianFEMesh(const UniformCartesian<T, Dim>& mesh);

        ElementType getElement(const std::size_t& element_index) const override;

    private:
        /**
         * @brief Get the indices of the element in each dimension
         *
         * @return Vector<std::size_t, Dim>
         */
        Vector<std::size_t, Dim> getElementDimIndices(const std::size_t& element_index) const;

        Vector<std::size_t, NumVertices> getVerticesForElementIndex(
            const std::size_t& element_index) const;
    };
}  // namespace ippl

#include "FEM/UniformCartesianFEMesh.hpp"

#endif  // IPPL_UNIFORMCARTESIANFEMESH_H