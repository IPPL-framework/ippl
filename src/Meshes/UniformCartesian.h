//
// Class UniformCartesian
//   UniformCartesian class - represents uniform-spacing cartesian meshes.
//
#ifndef IPPL_UNIFORM_CARTESIAN_H
#define IPPL_UNIFORM_CARTESIAN_H

#include "Meshes/CartesianCentering.h"
#include "Meshes/Mesh.h"

namespace ippl {

    template <typename T, unsigned Dim>
    class UniformCartesian : public Mesh<T, Dim> {
    public:
        typedef typename Mesh<T, Dim>::vector_type vector_type;
        typedef Cell DefaultCentering;

        KOKKOS_INLINE_FUNCTION UniformCartesian();

        KOKKOS_INLINE_FUNCTION UniformCartesian(const NDIndex<Dim>& ndi, const vector_type& hx,
                                                const vector_type& origin);

        KOKKOS_INLINE_FUNCTION ~UniformCartesian() = default;

        KOKKOS_INLINE_FUNCTION void initialize(const NDIndex<Dim>& ndi, const vector_type& hx,
                                               const vector_type& origin);

        // Set the spacings of mesh vertex positions (recompute Dvc, cell volume):
        KOKKOS_INLINE_FUNCTION void setMeshSpacing(const vector_type& meshSpacing);

        // Get the spacings of mesh vertex positions along specified direction
        KOKKOS_INLINE_FUNCTION T getMeshSpacing(unsigned dim) const;

        KOKKOS_INLINE_FUNCTION const vector_type& getMeshSpacing() const override;

        KOKKOS_INLINE_FUNCTION T getCellVolume() const override;

        KOKKOS_INLINE_FUNCTION T getMeshVolume() const override;

        KOKKOS_INLINE_FUNCTION void updateCellVolume_m();

        // (x,y,z) coordinates of indexed vertex:
        KOKKOS_INLINE_FUNCTION vector_type
        getVertexPosition(const NDIndex<Dim>& ndi) const override {
            //printf("inside getVertexPosition");
            vector_type vertexPosition;
            for (unsigned int d = 0; d < Dim; d++) {
                vertexPosition(d) = ndi[d].first() * meshSpacing_m[d] + this->origin_m(d);
                //printf("vertexPos = %lf", vertexPosition(d));
            }
            return vertexPosition;
        }

        // Vertex-vertex grid spacing of indexed cell:
        KOKKOS_INLINE_FUNCTION vector_type getDeltaVertex(const NDIndex<Dim>& ndi) const override {
            vector_type vertexVertexSpacing;
            for (unsigned int d = 0; d < Dim; d++)
                vertexVertexSpacing[d] = meshSpacing_m[d] * ndi[d].length();
            return vertexVertexSpacing;
        }

    private:
        vector_type meshSpacing_m;  // delta-x, delta-y (>1D), delta-z (>2D)
        T volume_m;                 // Cell length(1D), area(2D), or volume (>2D)
    };

}  // namespace ippl

#include "Meshes/UniformCartesian.hpp"

#endif
