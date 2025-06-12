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
    class Vector : public detail::Expression<Vector<T, Dim>, sizeof(T) * Dim> {
    public:
        typedef T value_type;
        static constexpr unsigned dim = Dim;

        KOKKOS_FUNCTION
        Vector()
            : Vector(value_type(0)) {}

        template <typename... Args,
                  typename std::enable_if<sizeof...(Args) == Dim, bool>::type = true>
        explicit KOKKOS_FUNCTION Vector(const Args&... args);

        template <typename E, size_t N>
        KOKKOS_FUNCTION Vector(const detail::Expression<E, N>& expr);

        KOKKOS_DEFAULTED_FUNCTION
        Vector(const Vector<T, Dim>& v) = default;

        KOKKOS_FUNCTION
        Vector(const T& val);

        Vector(const std::array<T, Dim>& a);

        Vector(const std::array<std::vector<T>, Dim>& a);

        /*!
         * @param list of values
         */
        KOKKOS_FUNCTION
        Vector(const std::initializer_list<T>& list);

        KOKKOS_FUNCTION
        ~Vector() {}

        // Get and Set Operations
        KOKKOS_INLINE_FUNCTION value_type& operator[](unsigned int i);

        KOKKOS_INLINE_FUNCTION value_type operator[](unsigned int i) const;

        KOKKOS_INLINE_FUNCTION value_type& operator()(unsigned int i);

        KOKKOS_INLINE_FUNCTION value_type operator()(unsigned int i) const;

        // Assignment Operators
        template <typename E, size_t N>
        KOKKOS_INLINE_FUNCTION Vector<T, Dim>& operator=(const detail::Expression<E, N>& expr);

        template <typename E, size_t N>
        KOKKOS_INLINE_FUNCTION Vector<T, Dim>& operator+=(const detail::Expression<E, N>& expr);

        template <typename E, size_t N>
        KOKKOS_INLINE_FUNCTION Vector<T, Dim>& operator-=(const detail::Expression<E, N>& expr);

        template <typename E, size_t N>
        KOKKOS_INLINE_FUNCTION Vector<T, Dim>& operator*=(const detail::Expression<E, N>& expr);

        template <typename E, size_t N>
        KOKKOS_INLINE_FUNCTION Vector<T, Dim>& operator/=(const detail::Expression<E, N>& expr);

        KOKKOS_INLINE_FUNCTION Vector<T, Dim>& operator+=(const T& val);

        KOKKOS_INLINE_FUNCTION Vector<T, Dim>& operator-=(const T& val);

        KOKKOS_INLINE_FUNCTION Vector<T, Dim>& operator*=(const T& val);

        KOKKOS_INLINE_FUNCTION Vector<T, Dim>& operator/=(const T& val);

        using iterator       = T*;
        using const_iterator = const T*;
        KOKKOS_INLINE_FUNCTION constexpr iterator begin();
        KOKKOS_INLINE_FUNCTION constexpr iterator end();
        KOKKOS_INLINE_FUNCTION constexpr const_iterator begin() const;
        KOKKOS_INLINE_FUNCTION constexpr const_iterator end() const;

        KOKKOS_INLINE_FUNCTION T dot(const Vector<T, Dim>& rhs) const;
        
        KOKKOS_INLINE_FUNCTION T Pnorm(const int p = 2) const;

        // Needs to be public to be a standard-layout type
        // private:
        T data_m[Dim];
    };

    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION Vector<T, Dim> min(const Vector<T, Dim>& a, const Vector<T, Dim>& b);
    template <typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION Vector<T, Dim> max(const Vector<T, Dim>& a, const Vector<T, Dim>& b);
}  // namespace ippl

#include "Vector.hpp"

#endif
