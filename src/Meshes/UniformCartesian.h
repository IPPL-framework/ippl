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

        UniformCartesian();

        UniformCartesian(const NDIndex<Dim>& ndi, const vector_type& hx, const vector_type& origin);

        ~UniformCartesian() = default;

        void initialize(const NDIndex<Dim>& ndi, const vector_type& hx, const vector_type& origin);

        // Set the spacings of mesh vertex positions (recompute Dvc, cell volume):
        void setMeshSpacing(const vector_type& meshSpacing);

        // Get the spacings of mesh vertex positions along specified direction
        T getMeshSpacing(unsigned dim) const;

        const vector_type& getMeshSpacing() const;

        T getCellVolume() const override;
        T getMeshVolume() const override;

        void updateCellVolume_m();

        // (x,y,z) coordinates of indexed vertex:
        vector_type getVertexPosition(const NDIndex<Dim>& ndi) const {
            vector_type vertexPosition;
            for (unsigned int d = 0; d < Dim; d++)
                vertexPosition(d) = ndi[d].first() * meshSpacing_m[d] + this->origin_m(d);
            return vertexPosition;
        }

        // Vertex-vertex grid spacing of indexed cell:
        vector_type getDeltaVertex(const NDIndex<Dim>& ndi) const {
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
