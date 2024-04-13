//
// Class Vector
//   Vector class used for vector fields and particle attributes like the coordinate.
//
#ifndef IPPL_Vector_H
#define IPPL_Vector_H

#include <initializer_list>

#include "Expression/IpplExpressions.h"

namespace ippl {
    /*!
     * @file Vector.h
     */

    /*!
     * @class Vector
     * @tparam T intrinsic vector data type
     * @tparam Dim vector dimension
     */
    template <typename T, unsigned Dim>
    struct Vector {
        using value_type = T;

        constexpr static unsigned dim = Dim;
        T data[Dim];
        KOKKOS_INLINE_FUNCTION constexpr Vector(const std::initializer_list<T>& list) {
            // PAssert(list.size() == Dim);
            unsigned int i = 0;
            for (auto& l : list) {
                data[i] = l;
                ++i;
            }
        }
        Vector() = default;
        constexpr KOKKOS_INLINE_FUNCTION Vector(T v) { fill(v); }
        KOKKOS_INLINE_FUNCTION value_type dot(const Vector<T, Dim>& v) const noexcept {
            value_type ret = 0;
            for (unsigned i = 0; i < dim; i++) {
                ret += data[i] * v[i];
            }
            return ret;
        }
        template <unsigned N>
        KOKKOS_INLINE_FUNCTION Vector<T, N> tail() const noexcept {
            Vector<T, N> ret;
            static_assert(N <= Dim, "N must be smaller than Dim");
            constexpr unsigned diff = Dim - N;
            for (unsigned i = 0; i < N; i++) {
                ret[i] = (*this)[i + diff];
            }
            return ret;
        }
        template <unsigned N>
        KOKKOS_INLINE_FUNCTION Vector<T, N> head() const noexcept {
            Vector<T, N> ret;
            static_assert(N <= Dim, "N must be smaller than Dim");
            for (unsigned i = 0; i < N; i++) {
                ret[i] = (*this)[i];
            }
            return ret;
        }
        KOKKOS_INLINE_FUNCTION value_type squaredNorm() const noexcept { return dot(*this); }
        KOKKOS_INLINE_FUNCTION value_type norm() {
#ifndef __CUDA_ARCH__
            using std::sqrt;
#endif
            return Kokkos::sqrt(squaredNorm());
        }
        KOKKOS_INLINE_FUNCTION Vector<T, Dim> normalized() const noexcept { return *this / norm(); }
        KOKKOS_INLINE_FUNCTION value_type sum() const noexcept {
            value_type ret = 0;
            for (unsigned i = 0; i < dim; i++) {
                ret += data[i];
            }
            return ret;
        }
        KOKKOS_INLINE_FUNCTION value_type average() const noexcept {
            value_type ret = 0;
            for (unsigned i = 0; i < dim; i++) {
                ret += data[i];
            }
            return ret / dim;
        }
        KOKKOS_INLINE_FUNCTION Vector<T, 3> cross(const Vector<T, Dim>& v) const noexcept
            requires(Dim == 3)
        {
            Vector<T, 3> ret(0);
            ret[0] = data[1] * v[2] - data[2] * v[1];
            ret[1] = data[2] * v[0] - data[0] * v[2];
            ret[2] = data[0] * v[1] - data[1] * v[0];
            return ret;
        }
        KOKKOS_INLINE_FUNCTION bool operator==(const Vector<T, dim>& o) const noexcept {
            for (unsigned i = 0; i < dim; i++) {
                if (data[i] != o[i])
                    return false;
            }
            return true;
        }
        KOKKOS_INLINE_FUNCTION value_type& operator[](unsigned int i) noexcept {
            assert(i < dim);
            return data[i];
        }
        KOKKOS_INLINE_FUNCTION T* begin() noexcept { return data; }
        KOKKOS_INLINE_FUNCTION T* end() noexcept { return data + dim; }
        KOKKOS_INLINE_FUNCTION const T* begin() const noexcept { return data; }
        KOKKOS_INLINE_FUNCTION const T* end() const noexcept { return data + dim; }
        KOKKOS_INLINE_FUNCTION constexpr void fill(value_type x) {
            for (unsigned i = 0; i < dim; i++) {
                data[i] = value_type(x);
            }
        }
        KOKKOS_INLINE_FUNCTION const value_type& operator[](unsigned int i) const noexcept {
            assert(i < dim);
            return data[i];
        }

        KOKKOS_INLINE_FUNCTION value_type& operator()(unsigned int i) noexcept {
            assert(i < dim);
            return data[i];
        }

        KOKKOS_INLINE_FUNCTION const value_type& operator()(unsigned int i) const noexcept {
            assert(i < dim);
            return data[i];
        }
        KOKKOS_INLINE_FUNCTION Vector operator-() const noexcept {
            Vector ret;
            for (unsigned i = 0; i < dim; i++) {
                ret[i] = -(*this)[i];
            }
            return ret;
        }
        KOKKOS_INLINE_FUNCTION Vector<T, dim> decompose(Vector<int, dim>* integral) {
#ifndef __CUDA_ARCH__
            using std::modf;
#endif
            Vector<T, dim> ret;
            for (unsigned i = 0; i < dim; i++) {
                if constexpr (std::is_same_v<T, float>) {
                    float tmp;
                    ret[i]         = modff((*this)[i], &tmp);
                    (*integral)[i] = (int)tmp;
                } else if constexpr (std::is_same_v<T, double>) {
                    double tmp;
                    ret[i]         = modf((*this)[i], &tmp);
                    (*integral)[i] = (int)tmp;
                }
            }
            return ret;
        }
        template <typename O>
        constexpr KOKKOS_INLINE_FUNCTION Vector<O, dim> cast() const noexcept {
            Vector<O, dim> ret;
            for (unsigned i = 0; i < dim; i++) {
                ret.data[i] = (O)(data[i]);
            }
            return ret;
        }
        KOKKOS_INLINE_FUNCTION Vector operator*(const value_type& o) const noexcept {
            Vector ret;
            for (unsigned i = 0; i < dim; i++) {
                ret[i] = (*this)[i] * o;
            }
            return ret;
        }
#define defop_kf(OP)                                                                  \
    constexpr KOKKOS_INLINE_FUNCTION Vector<T, Dim> operator OP(const Vector<T, Dim>& o) const noexcept { \
        Vector<T, Dim> ret;                                                           \
        for (unsigned i = 0; i < dim; i++) {                                          \
            ret.data[i] = data[i] OP o.data[i];                                       \
        }                                                                             \
        return ret;                                                                   \
    }
        defop_kf(+)
        defop_kf(-)
        defop_kf(*)
        defop_kf(/)

#define def_aop_kf(OP)                                                 \
    KOKKOS_INLINE_FUNCTION Vector<T, Dim>& operator OP(const Vector<T, Dim>& o) noexcept { \
        for (unsigned i = 0; i < dim; i++) {                           \
            (*this)[i] OP o[i];                                        \
        }                                                              \
        return *this;                                                  \
    }
            def_aop_kf(+=)
            def_aop_kf(-=)
            def_aop_kf(*=)
            def_aop_kf(/=)
            template <typename stream_t>
            friend stream_t& operator<<(stream_t& str, const Vector<T, Dim>& v) {
            // tr << "{";
            for (unsigned i = 0; i < Dim; i++) {
                str << v[i];
                if (i < Dim - 1)
                    str << ", ";
            }
            // str << "}";
            return str;
        }
        // defop_kf(*)
        // defop_kf(/)
    };

    //template <typename T, unsigned Dim>
    //KOKKOS_INLINE_FUNCTION Vector<T, Dim> min(const Vector<T, Dim>& a, const Vector<T, Dim>& b);
    //template <typename T, unsigned Dim>
    //KOKKOS_INLINE_FUNCTION Vector<T, Dim> max(const Vector<T, Dim>& a, const Vector<T, Dim>& b);
    template<typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION Vector<T, Dim> min(const Vector<T, Dim>& a, const Vector<T, Dim>& b){
        using Kokkos::min;
        Vector<T, Dim> ret;
        for(unsigned d = 0; d < Dim;d++){
            ret[d] = min(a[d], b[d]);
        }
        return ret;
    }
    template<typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION Vector<T, Dim> max(const Vector<T, Dim>& a, const Vector<T, Dim>& b){
        using Kokkos::max;
        Vector<T, Dim> ret;
        for(unsigned d = 0; d < Dim;d++){
            ret[d] = max(a[d], b[d]);
        }
        return ret;
    }
    template<typename O, typename T, unsigned N>
    KOKKOS_INLINE_FUNCTION ippl::Vector<T, N> operator *(const O &o, const ippl::Vector<T, N>& v) noexcept {
        ippl::Vector<T, N> ret;
        for (unsigned i = 0; i < N; i++) {
            ret[i] = v[i] * o;
        }
        return ret;
    }
    template <typename T>
    KOKKOS_INLINE_FUNCTION Vector<T, 3> cross(const ippl::Vector<T, 3>& a, const ippl::Vector<T, 3>& b) {
        ippl::Vector<T, 3> ret{0.0, 0.0, 0.0};
        ret[0] = a[1] * b[2] - a[2] * b[1];
        ret[1] = a[2] * b[0] - a[0] * b[2];
        ret[2] = a[0] * b[1] - a[1] * b[0];
        return ret;
    }
    template <typename T, unsigned dim>
    KOKKOS_INLINE_FUNCTION T dot(const ippl::Vector<T, dim>& a, const ippl::Vector<T, dim>& b) {
        T ret = 0;
        for(unsigned k = 0;k < dim;k++){
            ret += a[k] * b[k];
        }
        return ret;
    }
}  // namespace ippl

//#include "Vector.hpp"

#endif
