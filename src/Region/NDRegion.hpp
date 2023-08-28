//
// Class NDRegion
//   NDRegion is a simple container of N PRegion objects. It is templated
//   on the type of data (T) and the number of PRegions (Dim).
//
#include <iostream>

namespace ippl {
    template <typename T, unsigned Dim>
    template <class... Args>
    KOKKOS_FUNCTION NDRegion<T, Dim>::NDRegion(const Args&... args)
        : NDRegion({args...}) {
        static_assert(Dim == sizeof...(args), "Wrong number of arguments.");
    }

    template <typename T, unsigned Dim>
    KOKKOS_FUNCTION NDRegion<T, Dim>::NDRegion(std::initializer_list<PRegion<T>> regions) {
        unsigned int i = 0;
        for (auto& r : regions) {
            regions_m[i] = r;
            ++i;
        }
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION NDRegion<T, Dim>::NDRegion(const NDRegion<T, Dim>& nr) {
        for (unsigned int i = 0; i < Dim; i++) {
            regions_m[i] = nr.regions_m[i];
        }
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION NDRegion<T, Dim>& NDRegion<T, Dim>::operator=(
        const NDRegion<T, Dim>& nr) {
        for (unsigned int i = 0; i < Dim; i++) {
            regions_m[i] = nr.regions_m[i];
        }
        return *this;
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION const PRegion<T>& NDRegion<T, Dim>::operator[](unsigned d) const {
        return regions_m[d];
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION PRegion<T>& NDRegion<T, Dim>::operator[](unsigned d) {
        return regions_m[d];
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION NDRegion<T, Dim>& NDRegion<T, Dim>::operator+=(const T t) {
        for (unsigned int i = 0; i < Dim; i++) {
            regions_m[i] += t;
        }
        return *this;
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION NDRegion<T, Dim>& NDRegion<T, Dim>::operator-=(const T t) {
        for (unsigned int i = 0; i < Dim; i++) {
            regions_m[i] -= t;
        }
        return *this;
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION NDRegion<T, Dim>& NDRegion<T, Dim>::operator*=(const T t) {
        for (unsigned int i = 0; i < Dim; i++) {
            regions_m[i] *= t;
        }
        return *this;
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION NDRegion<T, Dim>& NDRegion<T, Dim>::operator/=(const T t) {
        if (t != 0) {
            for (unsigned int i = 0; i < Dim; i++) {
                regions_m[i] /= t;
            }
        }
        return *this;
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION bool NDRegion<T, Dim>::empty() const {
        bool isEmpty = true;
        for (unsigned int i = 0; i < Dim; i++) {
            isEmpty &= regions_m[i].empty();
        }
        return isEmpty;
    }

    template <typename T, unsigned Dim>
    inline std::ostream& operator<<(std::ostream& out, const NDRegion<T, Dim>& idx) {
        out << '{';
        for (unsigned d = 0; d < Dim; ++d) {
            out << idx[d] << ((d == Dim - 1) ? '}' : ',');
        }
        return out;
    }
}  // namespace ippl