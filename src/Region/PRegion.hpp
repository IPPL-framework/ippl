//
// Class PRegion
//   PRegion represents a (possibly continuous) numeric interval.  It is
//   similar to Index, with the following differences:
//      1. It is templated on the data type; Index always uses integers
//      2. A PRegion is defined between two endpoints A and B; the PRegion
//         includes values X where A <= X < B (i.e., X in [A,B) )
//      3. PRegion does not keep track of a base Index, and does not
//         supply the plugBase operation.  It is not designed for use in
//         Field operations like Index is, it is meant instead for use in
//         Particle construction and usage.
//
//   PRegion<T>()      --> make a PRegion on [0,1)
//   PRegion<T>(B)     --> make a PRegion on [0,B)
//   PRegion<T>(A,B)   --> make a PRegion on [A,B)
//
#include <algorithm>
#include <iostream>

#include "Utility/PAssert.h"

namespace ippl {
    template <typename T>
    KOKKOS_FUNCTION PRegion<T>::PRegion()
        : PRegion(0, 1) {}

    template <typename T>
    KOKKOS_FUNCTION PRegion<T>::PRegion(T b)
        : PRegion(0, b) {}

    template <typename T>
    KOKKOS_FUNCTION PRegion<T>::PRegion(T a, T b)
        : a_m(a)
        , b_m(b) {
        PAssert(a_m < b_m);
    }

    template <typename T>
    KOKKOS_FUNCTION PRegion<T>::PRegion(const PRegion<T>& pregion) {
        a_m = pregion.a_m;
        b_m = pregion.b_m;
    }

    template <typename T>
    KOKKOS_INLINE_FUNCTION PRegion<T>& PRegion<T>::operator=(const PRegion<T>& pregion) {
        a_m = pregion.a_m;
        b_m = pregion.b_m;
        return *this;
    }

    template <typename T>
    KOKKOS_INLINE_FUNCTION T PRegion<T>::min() const noexcept {
        return a_m;
    }

    template <typename T>
    KOKKOS_INLINE_FUNCTION T PRegion<T>::max() const noexcept {
        return b_m;
    }

    template <typename T>
    KOKKOS_INLINE_FUNCTION T PRegion<T>::length() const noexcept {
        return b_m - a_m;
    }

    template <typename T>
    KOKKOS_INLINE_FUNCTION bool PRegion<T>::empty() const noexcept {
        return (a_m == b_m);
    }

    template <typename T>
    KOKKOS_INLINE_FUNCTION PRegion<T>& PRegion<T>::operator+=(T t) noexcept {
        a_m += t;
        b_m += t;
        return *this;
    }

    template <typename T>
    KOKKOS_INLINE_FUNCTION PRegion<T>& PRegion<T>::operator-=(T t) noexcept {
        a_m -= t;
        b_m -= t;
        return *this;
    }

    template <typename T>
    KOKKOS_INLINE_FUNCTION PRegion<T>& PRegion<T>::operator*=(T t) noexcept {
        a_m *= t;
        b_m *= t;
        return *this;
    }

    template <typename T>
    KOKKOS_INLINE_FUNCTION PRegion<T>& PRegion<T>::operator/=(T t) noexcept {
        if (t != 0) {
            a_m /= t;
            b_m /= t;
        }
        return *this;
    }

    template <typename T>
    inline std::ostream& operator<<(std::ostream& out, const PRegion<T>& r) {
        out << '[' << r.min();
        out << ',' << r.max();
        out << ')';
        return out;
    }
}  // namespace ippl
