//
// Class Vector
//   Vector class used for vector fields and particle attributes like the coordinate.
//
// #include "Utility/PAssert.h"

#include <iomanip>
#include <iostream>

namespace ippl {
    namespace detail {
        template <typename T, unsigned Dim>
        struct isExpression<Vector<T, Dim>> : std::true_type {};
    }  // namespace detail

    template <typename T, unsigned Dim>
    template <typename... Args, typename std::enable_if<sizeof...(Args) == Dim, bool>::type>
    KOKKOS_FUNCTION Vector<T, Dim>::Vector(const Args&... args)
        : Vector({static_cast<T>(args)...}) {}

    template <typename T, unsigned Dim>
    template <typename E, size_t N>
    KOKKOS_FUNCTION Vector<T, Dim>::Vector(const detail::Expression<E, N>& expr) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] = expr[i];
        }
    }

    template <typename T, unsigned Dim>
    KOKKOS_FUNCTION Vector<T, Dim>::Vector(const T& val) {
        for (unsigned int i = 0; i < Dim; ++i)
            data_m[i] = val;
    }

    template <typename T, unsigned Dim>
    KOKKOS_FUNCTION Vector<T, Dim>::Vector(const std::initializer_list<T>& list) {
        // PAssert(list.size() == Dim);
        unsigned int i = 0;
        for (auto& l : list) {
            data_m[i] = l;
            ++i;
        }
    }

    /*
     *
     * Element access operators
     *
     */
    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION typename Vector<T, Dim>::value_type& Vector<T, Dim>::operator[](
        unsigned int i) {
        // PAssert(i < Dim);
        return data_m[i];
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION typename Vector<T, Dim>::value_type Vector<T, Dim>::operator[](
        unsigned int i) const {
        // PAssert(i < Dim);
        return data_m[i];
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION typename Vector<T, Dim>::value_type& Vector<T, Dim>::operator()(
        unsigned int i) {
        // PAssert(i < Dim);
        return data_m[i];
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION typename Vector<T, Dim>::value_type Vector<T, Dim>::operator()(
        unsigned int i) const {
        // PAssert(i < Dim);
        return data_m[i];
    }

    /*
     *
     * Vector Expression assignment operators
     *
     */
    template <typename T, unsigned Dim>
    template <typename E, size_t N>
    KOKKOS_INLINE_FUNCTION Vector<T, Dim>& Vector<T, Dim>::operator=(
        const detail::Expression<E, N>& expr) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] = expr[i];
        }
        return *this;
    }

    template <typename T, unsigned Dim>
    template <typename E, size_t N>
    KOKKOS_INLINE_FUNCTION Vector<T, Dim>& Vector<T, Dim>::operator+=(
        const detail::Expression<E, N>& expr) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] += expr[i];
        }
        return *this;
    }

    template <typename T, unsigned Dim>
    template <typename E, size_t N>
    KOKKOS_INLINE_FUNCTION Vector<T, Dim>& Vector<T, Dim>::operator-=(
        const detail::Expression<E, N>& expr) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] -= expr[i];
        }
        return *this;
    }

    template <typename T, unsigned Dim>
    template <typename E, size_t N>
    KOKKOS_INLINE_FUNCTION Vector<T, Dim>& Vector<T, Dim>::operator*=(
        const detail::Expression<E, N>& expr) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] *= expr[i];
        }
        return *this;
    }

    template <typename T, unsigned Dim>
    template <typename E, size_t N>
    KOKKOS_INLINE_FUNCTION Vector<T, Dim>& Vector<T, Dim>::operator/=(
        const detail::Expression<E, N>& expr) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] /= expr[i];
        }
        return *this;
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION Vector<T, Dim>& Vector<T, Dim>::operator+=(const T& val) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] += val;
        }
        return *this;
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION Vector<T, Dim>& Vector<T, Dim>::operator-=(const T& val) {
        return this->operator+=(-val);
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION Vector<T, Dim>& Vector<T, Dim>::operator*=(const T& val) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] *= val;
        }
        return *this;
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION Vector<T, Dim>& Vector<T, Dim>::operator/=(const T& val) {
        return this->operator*=(T(1.0) / val);
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION constexpr typename Vector<T, Dim>::iterator Vector<T, Dim>::begin() {
        return data_m;
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION constexpr typename Vector<T, Dim>::iterator Vector<T, Dim>::end() {
        return data_m + Dim;
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION constexpr typename Vector<T, Dim>::const_iterator Vector<T, Dim>::begin()
        const {
        return data_m;
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION constexpr typename Vector<T, Dim>::const_iterator Vector<T, Dim>::end()
        const {
        return data_m + Dim;
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION T Vector<T, Dim>::dot(const Vector<T, Dim>& rhs) const {
        T res = 0.0;
        for (unsigned i = 0; i < Dim; ++i) {
            res += data_m[i] * rhs[i];
        }
        return res;
    }

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION T Vector<T, Dim>::Pnorm(const int p) const {

        T val = 0.0;
        for(unsigned i = 0; i < Dim; ++i) {
            val += Kokkos::pow(Kokkos::abs(data_m[i]), p);
        }

        return Kokkos::pow(val, T(1.0) / T(p));
    }

    template <typename T, unsigned Dim>
    inline std::ostream& operator<<(std::ostream& out, const Vector<T, Dim>& v) {
        std::streamsize sw = out.width();
        out << std::setw(1);
        if constexpr (Dim > 1) {
            out << "( ";
            for (unsigned int i = 0; i < Dim - 1; i++) {
                out << std::setw(sw) << v[i] << " , ";
            }
            out << std::setw(sw) << v[Dim - 1] << " )";
        } else {
            out << "( " << std::setw(sw) << v[0] << " )";
        }
        return out;
    }
    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION Vector<T, Dim> min(const Vector<T, Dim>& a, const Vector<T, Dim>& b) {
        using Kokkos::min;
        Vector<T, Dim> ret;
        for (unsigned d = 0; d < Dim; d++) {
            ret[d] = min(a[d], b[d]);
        }
        return ret;
    }
    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION Vector<T, Dim> max(const Vector<T, Dim>& a, const Vector<T, Dim>& b) {
        using Kokkos::max;
        Vector<T, Dim> ret;
        for (unsigned d = 0; d < Dim; d++) {
            ret[d] = max(a[d], b[d]);
        }
        return ret;
    }
}  // namespace ippl

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
