//
// Class Mesh
//   The Mesh base class. Right now, this mainly acts as a standard base
//   class for all meshes so that other objects can register as users of
//   the mesh and can be notified if the mesh changes (e.g., it is rescaled
//   or restructured entirely).
//
namespace ippl {
    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION typename Mesh<T, Dim>::vector_type Mesh<T, Dim>::getOrigin() const {
        return origin_m;
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION void Mesh<T, Dim>::setOrigin(const vector_type& origin) {
        origin_m = origin;
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION const typename Mesh<T, Dim>::vector_type& Mesh<T, Dim>::getGridsize()
        const {
        return gridSizes_m;
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION T Mesh<T, Dim>::getGridsize(size_t dim) const {
        return gridSizes_m[dim];
    }
}  // namespace ippl
