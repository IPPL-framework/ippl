//
// Class UniformCartesian
//   UniformCartesian class - represents uniform-spacing cartesian meshes.
//
#include "Utility/IpplInfo.h"
#include "Utility/PAssert.h"

#include "Field/BareField.h"
#include "Field/Field.h"

namespace ippl {

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION UniformCartesian<T, Dim>::UniformCartesian()
        : Mesh<T, Dim>()
        , volume_m(0.0) {}

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION UniformCartesian<T, Dim>::UniformCartesian(const NDIndex<Dim>& ndi,
                                                                      const vector_type& hx,
                                                                      const vector_type& origin) {
        this->initialize(ndi, hx, origin);
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION void UniformCartesian<T, Dim>::initialize(const NDIndex<Dim>& ndi,
                                                                     const vector_type& hx,
                                                                     const vector_type& origin) {
        meshSpacing_m = hx;

        volume_m = 1.0;
        for (unsigned d = 0; d < Dim; ++d) {
            this->gridSizes_m[d] = ndi[d].length();
            volume_m *= meshSpacing_m[d];
        }

        this->setOrigin(origin);
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION void UniformCartesian<T, Dim>::setMeshSpacing(
        const vector_type& meshSpacing) {
        meshSpacing_m = meshSpacing;
        this->updateCellVolume_m();
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION T UniformCartesian<T, Dim>::getMeshSpacing(unsigned dim) const {
        PAssert_LT(dim, Dim);
        return meshSpacing_m[dim];
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION const typename UniformCartesian<T, Dim>::vector_type&
    UniformCartesian<T, Dim>::getMeshSpacing() const {
        return meshSpacing_m;
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION T UniformCartesian<T, Dim>::getCellVolume() const {
        return volume_m;
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION T UniformCartesian<T, Dim>::getMeshVolume() const {
        T ret = 1;
        for (unsigned int d = 0; d < Dim; ++d) {
            ret *= this->getGridsize(d) * this->getMeshSpacing(d);
        }
        return ret;
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION void UniformCartesian<T, Dim>::updateCellVolume_m() {
        // update cell volume
        volume_m = 1.0;
        for (unsigned i = 0; i < Dim; ++i) {
            volume_m *= meshSpacing_m[i];
        }
    }

}  // namespace ippl
