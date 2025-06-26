//
// Class NDIndex
//   This is a simple wrapper around Index that just keeps track of
//   N of them and passes along requests for intersect, etc.
//
#include <iostream>

namespace ippl {
    template <unsigned Dim>
    template <class... Args>
    KOKKOS_FUNCTION NDIndex<Dim>::NDIndex(const Args&... args)
        : NDIndex({args...}) {
        static_assert(Dim == sizeof...(args), "Wrong number of arguments.");
    }

    template <unsigned Dim>
    KOKKOS_FUNCTION NDIndex<Dim>::NDIndex(std::initializer_list<Index> indices) {
        unsigned int i = 0;
        for (auto& index : indices) {
            indices_m[i] = index;
            ++i;
        }
    }

    template <unsigned Dim>
    KOKKOS_FUNCTION NDIndex<Dim>::NDIndex(const Vector<unsigned, Dim>& sizes) {
        for (unsigned int d = 0; d < Dim; ++d) {
            indices_m[d] = Index(sizes[d]);
        }
    }

    template <unsigned Dim>
    KOKKOS_INLINE_FUNCTION const Index& NDIndex<Dim>::operator[](unsigned d) const noexcept {
        return indices_m[d];
    }

    template <unsigned Dim>
    KOKKOS_INLINE_FUNCTION Index& NDIndex<Dim>::operator[](unsigned d) noexcept {
        return indices_m[d];
    }

    template <unsigned Dim>
    KOKKOS_INLINE_FUNCTION unsigned NDIndex<Dim>::size() const noexcept {
        unsigned s = indices_m[0].length();
        for (unsigned int d = 1; d < Dim; ++d) {
            s *= indices_m[d].length();
        }
        return s;
    }

    template <unsigned Dim>
    KOKKOS_INLINE_FUNCTION bool NDIndex<Dim>::empty() const noexcept {
        bool r = false;
        for (unsigned d = 0; d < Dim; ++d) {
            r = r || indices_m[d].empty();
        }
        return r;
    }

    template <unsigned Dim>
    inline std::ostream& operator<<(std::ostream& out, const NDIndex<Dim>& idx) {
        out << '{';
        for (unsigned d = 0; d < Dim; ++d) {
            out << idx[d] << ((d == Dim - 1) ? '}' : ',');
        }
        return out;
    }

    template <unsigned Dim>
    KOKKOS_INLINE_FUNCTION NDIndex<Dim> NDIndex<Dim>::intersect(const NDIndex<Dim>& ndi) const {
        NDIndex<Dim> r;
        for (unsigned d = 0; d < Dim; ++d) {
            r[d] = indices_m[d].intersect(ndi[d]);
        }
        return r;
    }

    template <unsigned Dim>
    KOKKOS_INLINE_FUNCTION NDIndex<Dim> NDIndex<Dim>::grow(int ncells) const {
        NDIndex<Dim> r;
        for (unsigned d = 0; d < Dim; ++d) {
            r[d] = indices_m[d].grow(ncells);
        }
        return r;
    }

    template <unsigned Dim>
    KOKKOS_INLINE_FUNCTION NDIndex<Dim> NDIndex<Dim>::grow(int ncells, unsigned int dim) const {
        NDIndex<Dim> r = *this;
        r[dim]         = indices_m[dim].grow(ncells);
        return r;
    }

    template <unsigned Dim>
    KOKKOS_INLINE_FUNCTION bool NDIndex<Dim>::touches(const NDIndex<Dim>& a) const {
        bool touch = true;
        for (unsigned int d = 0; (d < Dim) && touch; ++d) {
            touch = touch && indices_m[d].touches(a.indices_m[d]);
        }
        return touch;
    }

    template <unsigned Dim>
    KOKKOS_INLINE_FUNCTION bool NDIndex<Dim>::contains(const NDIndex<Dim>& a) const {
        bool cont = true;
        for (unsigned int d = 0; (d < Dim) && cont; ++d) {
            cont = cont && indices_m[d].contains(a.indices_m[d]);
        }
        return cont;
    }

    template <unsigned Dim>
    KOKKOS_INLINE_FUNCTION bool NDIndex<Dim>::split(NDIndex<Dim>& l, NDIndex<Dim>& r, unsigned d,
                                                    int i) const {
        if (&l != this) {
            l = *this;
        }
        if (&r != this) {
            r = *this;
        }
        return indices_m[d].split(l[d], r[d], i);
    }

    template <unsigned Dim>
    KOKKOS_INLINE_FUNCTION bool NDIndex<Dim>::split(NDIndex<Dim>& l, NDIndex<Dim>& r, unsigned d,
                                                    double a) const {
        if (&l != this) {
            l = *this;
        }
        if (&r != this) {
            r = *this;
        }
        return indices_m[d].split(l[d], r[d], a);
    }

    template <unsigned Dim>
    KOKKOS_INLINE_FUNCTION bool NDIndex<Dim>::split(NDIndex<Dim>& l, NDIndex<Dim>& r,
                                                    unsigned d) const {
        if (&l != this) {
            l = *this;
        }
        if (&r != this) {
            r = *this;
        }
        return indices_m[d].split(l[d], r[d]);
    }

    template <unsigned Dim>
    KOKKOS_INLINE_FUNCTION bool NDIndex<Dim>::split(NDIndex<Dim>& l, NDIndex<Dim>& r) const {
        unsigned int max_dim    = 0;
        unsigned int max_length = 0;
        for (unsigned int d = 0; d < Dim; ++d) {
            if (indices_m[d].length() > max_length) {
                max_dim    = d;
                max_length = indices_m[d].length();
            }
        }
        return split(l, r, max_dim);
    }

    template <unsigned Dim>
    KOKKOS_INLINE_FUNCTION Vector<size_t, Dim> NDIndex<Dim>::length() const {
        auto construct = [&]<size_t... Idx>(const std::index_sequence<Idx...>&) {
            return Vector<size_t, Dim>{indices_m[Idx].length()...};
        };
        return construct(std::make_index_sequence<Dim>{});
    }

    template <unsigned Dim>
    KOKKOS_INLINE_FUNCTION Vector<int, Dim> NDIndex<Dim>::first() const {
        auto construct = [&]<size_t... Idx>(const std::index_sequence<Idx...>&) {
            return Vector<int, Dim>{indices_m[Idx].first()...};
        };
        return construct(std::make_index_sequence<Dim>{});
    }

    template <unsigned Dim>
    KOKKOS_INLINE_FUNCTION Vector<int, Dim> NDIndex<Dim>::last() const {
        auto construct = [&]<size_t... Idx>(const std::index_sequence<Idx...>&) {
            return Vector<int, Dim>{indices_m[Idx].last()...};
        };
        return construct(std::make_index_sequence<Dim>{});
    }

    template <unsigned Dim>
    KOKKOS_INLINE_FUNCTION constexpr typename NDIndex<Dim>::iterator NDIndex<Dim>::begin() {
        return indices_m;
    }

    template <unsigned Dim>
    KOKKOS_INLINE_FUNCTION constexpr typename NDIndex<Dim>::iterator NDIndex<Dim>::end() {
        return indices_m + Dim;
    }

    template <unsigned Dim>
    KOKKOS_INLINE_FUNCTION constexpr typename NDIndex<Dim>::const_iterator NDIndex<Dim>::begin()
        const {
        return indices_m;
    }

    template <unsigned Dim>
    KOKKOS_INLINE_FUNCTION constexpr typename NDIndex<Dim>::const_iterator NDIndex<Dim>::end()
        const {
        return indices_m + Dim;
    }

    template <unsigned Dim>
    bool operator==(const NDIndex<Dim>& nd1, const NDIndex<Dim>& nd2) {
        for (unsigned d = 0; d < Dim; d++) {
            if (nd1[d] != nd2[d]) {
                return false;
            }
        }
        return true;
    }

    template <unsigned Dim>
    bool operator!=(const NDIndex<Dim>& nd1, const NDIndex<Dim>& nd2) {
        return !(nd1 == nd2);
    }
}  // namespace ippl
