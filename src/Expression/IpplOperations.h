//
// File IpplOperations.h
//   Expression Templates operations.
//
// Copyright (c) 2020, Matthias Frey, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//
#ifndef IPPL_OPERATIONS_H
#define IPPL_OPERATIONS_H

#include "AppTypes/Vector.h"

namespace ippl {
    /*
     * Binary operations for Scalar, Vector and LField classes.
     */
    #define DefineBinaryOperation(fun, name, op1, op2)                      \
    template<typename E1, typename E2>                                      \
    struct fun : public Expression<fun<E1, E2>, sizeof(E1) + sizeof(E2)> {  \
        KOKKOS_FUNCTION                                                     \
        fun(const E1& u, const E2& v) : u_m(u), v_m(v) { }                  \
                                                                            \
        KOKKOS_INLINE_FUNCTION                                              \
        auto operator[](size_t i) const { return op1; }                     \
                                                                            \
        template<typename ...Args>                                          \
        KOKKOS_INLINE_FUNCTION                                              \
        auto operator()(Args... args) const {                               \
            return op2;                                                     \
        }                                                                   \
                                                                            \
    private:                                                                \
        const E1 u_m;                                                       \
        const E2 v_m;                                                       \
    };                                                                      \
                                                                            \
    template<typename E1, size_t N1, typename E2, size_t N2>                \
    KOKKOS_INLINE_FUNCTION                                                  \
    fun<E1, E2> name(const Expression<E1, N1>& u,                           \
                     const Expression<E2, N2>& v) {                         \
        return fun<E1, E2>(*static_cast<const E1*>(&u),                     \
                           *static_cast<const E2*>(&v));                    \
    }                                                                       \
                                                                            \
    template<typename E, size_t N, typename T,                              \
             typename = std::enable_if_t<std::is_scalar<T>::value>>         \
    KOKKOS_INLINE_FUNCTION                                                  \
    fun<E, Scalar<T>> name(const Expression<E, N>& u,                       \
                           const T& v) {                                    \
        return fun<E, Scalar<T>>(*static_cast<const E*>(&u), v);            \
    }                                                                       \
                                                                            \
    template<typename E, size_t N, typename T,                              \
             typename = std::enable_if_t<std::is_scalar<T>::value>>         \
    KOKKOS_INLINE_FUNCTION                                                  \
    fun<E, Scalar<T>> name(const T& u,                                      \
                           const Expression<E, N>& v) {                     \
        return fun<E, Scalar<T>>(*static_cast<const E*>(&v), u);            \
    }


    DefineBinaryOperation(Add,      operator+, u_m[i] + v_m[i], u_m(args...) + v_m(args...))
    DefineBinaryOperation(Subtract, operator-, u_m[i] - v_m[i], u_m(args...) - v_m(args...))
    DefineBinaryOperation(Multiply, operator*, u_m[i] * v_m[i], u_m(args...) * v_m(args...))
    DefineBinaryOperation(Divide,   operator/, u_m[i] / v_m[i], u_m(args...) / v_m(args...))

    /*
     * Cross product. This function is only supported for 3-dimensional vectors.
     */

    template<typename E1, typename E2>
    struct meta_cross : public Expression<meta_cross<E1, E2>, sizeof(E1) + sizeof(E2)> {
        KOKKOS_FUNCTION
        meta_cross(const E1& u, const E2& v) : u_m(u), v_m(v) {
//             static_assert(E1::dim == 3, "meta_cross: Dimension of first argument needs to be 3");
//             static_assert(E2::dim == 3, "meta_cross: Dimension of second argument needs to be 3");
        }

        /*
         * Vector::cross
         */
        KOKKOS_INLINE_FUNCTION
        auto operator[](size_t i) const {
            const size_t j = (i + 1) % 3;
            const size_t k = (i + 2) % 3;
            return  u_m[j] * v_m[k] - u_m[k] * v_m[j];
        }

        /*
         * This is required for LField::cross
         */
        template<typename ...Args>
        KOKKOS_INLINE_FUNCTION
        auto operator()(Args... args) const {
            return cross(u_m(args...), v_m(args...));
        }

    private:
        const E1 u_m;
        const E2 v_m;
    };

    template<typename E1, size_t N1, typename E2, size_t N2>
    KOKKOS_INLINE_FUNCTION
    meta_cross<E1, E2> cross(const Expression<E1, N1>& u, const Expression<E2, N2>& v) {
        return meta_cross<E1, E2>(*static_cast<const E1*>(&u),
                                  *static_cast<const E2*>(&v));
    }


    /*
     * Dot product.
     */
    template<typename E1, typename E2>
    struct meta_dot : public Expression<meta_dot<E1, E2>, sizeof(E1) + sizeof(E2)> {
        KOKKOS_FUNCTION
        meta_dot(const E1& u, const E2& v) : u_m(u), v_m(v) {
//             static_assert(E1::dim == E2::dim, "meta_dot: Dimensions do not agree!");
        }

        /*
         * Vector::dot
         */
        KOKKOS_INLINE_FUNCTION
        typename E1::value_t operator()() const {
            typename E1::value_t res = 0.0;
            for (size_t i = 0; i < E1::dim; ++i) {
                res += u_m[i] * v_m[i];
            }
            return res; //u_m[0] * v_m[0] + u_m[1] * v_m[1] + u_m[2] * v_m[2];
        }

        /*
         * This is required for LField::dot
         */
        template<typename ...Args>
        KOKKOS_INLINE_FUNCTION
        auto operator()(Args... args) const {
            return dot(u_m(args...) * v_m(args...));
        }

    private:
        const E1 u_m;
        const E2 v_m;
    };

    template<typename E1, size_t N1, typename E2, size_t N2>
    KOKKOS_INLINE_FUNCTION
    typename E1::value_t dot(const Expression<E1, N1>& u, const Expression<E2, N2>& v) {
        return meta_dot<E1, E2>(*static_cast<const E1*>(&u),
                                *static_cast<const E2*>(&v))();
    }


    /*
     * Gradient
     */
    template<typename E, typename M>
    struct meta_grad : public Expression<meta_grad<E, M>, sizeof(E)> {
        KOKKOS_FUNCTION
        meta_grad(const E& u, const M& m) : u_m(u) {
            idx[0] = 0.5 / m.getMeshSpacing(0);
            idx[1] = 0.0;
            idx[2] = 0.0;
            idy[0] = 0.0;
            idy[1] = 0.5 / m.getMeshSpacing(1);
            idy[2] = 0.0;
            idz[0] = 0.0;
            idz[1] = 0.0;
            idz[2] = 0.5 / m.getMeshSpacing(2);
        }

        /*
         * This is required for LField::grad
         */
        KOKKOS_INLINE_FUNCTION
        auto operator()(size_t i) const {
            std::cout << "1d meta_grad for LField" << std::endl;
            return u_m(i);
        }

         KOKKOS_INLINE_FUNCTION
        auto operator()(size_t i, size_t j) const {
            std::cout << "2d meta_grad for LField" << std::endl;
            return u_m(i, j);
        }

        KOKKOS_INLINE_FUNCTION
        auto operator()(size_t i, size_t j, size_t k) const {
            return idx * (u_m(i+1, j,   k)   - u_m(i-1, j,   k  )) +
                   idy * (u_m(i  , j+1, k)   - u_m(i  , j-1, k  )) +
                   idz * (u_m(i  , j  , k+1) - u_m(i  , j  , k-1));
        }

    private:
        const E u_m;
        Vector<double, 3> idx;
        Vector<double, 3> idy;
        Vector<double, 3> idz;
    };

    template<typename E, size_t N, typename M>
    KOKKOS_INLINE_FUNCTION
    meta_grad<E, M> grad(const Expression<E, N>& u, const M& m) {
        return meta_grad<E, M>(*static_cast<const E*>(&u), m);
    }
}

#endif