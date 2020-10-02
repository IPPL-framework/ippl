//
// File FieldExpressions.h
//   Local Field class
//
// Copyright (c) 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
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
#ifndef IPPL_FIELD_EXPRESSIONS_H
#define IPPL_FIELD_EXPRESSIONS_H

#include "AppTypes/VektorExpressions.h"

namespace ippl {
    /* Although the template argument T is not used, it is required
     * in order to avoid operator ambiguity error.
     */
    template <typename T, typename E, size_t N = sizeof(E)>
    struct FieldExpr {
        KOKKOS_INLINE_FUNCTION
        auto operator()(size_t i) const {
            return static_cast<const E&>(*this)(i);
        }
    };


    /* Although the template argument T is not used, it is required
     * in order to avoid operator ambiguity error.
     */
    template <typename T, typename E, size_t N = sizeof(E)>
    struct LFieldCaptureExpr {
        template<typename ...Args>
        KOKKOS_INLINE_FUNCTION
        auto operator()(Args... args) const {
            return reinterpret_cast<const E&>(*this)(args...);
        }

        char buffer[N];
    };


    #define DefineFieldFieldOperation(fun, op, expr)                            \
    template<typename T, typename E1, typename E2>                              \
    struct fun : public FieldExpr<T, fun<T, E1, E2>, sizeof(E1) + sizeof(E2)> { \
        fun(const E1& u, const E2& v) : u_m(u), v_m(v) { }                      \
                                                                                \
        template<typename ...Args>                                              \
        KOKKOS_INLINE_FUNCTION                                                  \
        auto operator()(Args... args) const {                                   \
            return expr;                                                        \
        }                                                                       \
                                                                                \
    private:                                                                    \
        const E1 u_m;                                                           \
        const E2 v_m;                                                           \
    };                                                                          \
                                                                                \
    template<typename T, typename E1, typename E2, size_t N1, size_t N2>        \
    fun<T, E1, E2> op(const FieldExpr<T, E1, N1>& u,                            \
                      const FieldExpr<T, E2, N2>& v) {                          \
        return fun<T, E1, E2>(*static_cast<const E1*>(&u),                      \
                              *static_cast<const E2*>(&v));                     \
    }

    DefineFieldFieldOperation(LFieldAdd,      operator+, u_m(args...) + v_m(args...))
    DefineFieldFieldOperation(LFieldSubtract, operator-, u_m(args...) - v_m(args...))
    DefineFieldFieldOperation(LFieldMultiply, operator*, u_m(args...) * v_m(args...))
    DefineFieldFieldOperation(LFieldDivide,   operator/, u_m(args...) / v_m(args...))


    #define DefineScalarFieldOperation(fun, op, expr)                           \
    template<typename T, typename E>                                            \
    struct fun : public FieldExpr<T, fun<T, E>, sizeof(E) + sizeof(T)> {        \
        fun(const T& u, const E& v) : u_m(u), v_m(v) { }                        \
                                                                                \
        template<typename ...Args>                                              \
        KOKKOS_INLINE_FUNCTION                                                  \
        auto operator()(Args... args) const {                                   \
            return expr;                                                        \
        }                                                                       \
                                                                                \
    private:                                                                    \
        const T u_m;                                                            \
        const E v_m;                                                            \
    };                                                                          \
                                                                                \
    template<typename T, typename E, size_t N>                                  \
    fun<T, E> op(const T& u,                                                    \
                 const FieldExpr<T, E, N>& v) {                                 \
        return fun<T, E>(u,                                                     \
                         *static_cast<const E*>(&v));                           \
    }

    DefineScalarFieldOperation(LFieldAddScalarLeft,      operator+, u_m + v_m(args...))
    DefineScalarFieldOperation(LFieldSubtractScalarLeft, operator-, u_m - v_m(args...))
    DefineScalarFieldOperation(LFieldMultiplyScalarLeft, operator*, u_m * v_m(args...))
    DefineScalarFieldOperation(LFieldDivideScalarLeft,   operator/, u_m / v_m(args...))


    #define DefineFieldScalarOperation(fun, op, expr)                           \
    template<typename T, typename E>                                            \
    struct fun : public FieldExpr<T, fun<T, E>, sizeof(E) + sizeof(T)> {        \
        fun(const E& u, const T& v) : u_m(u), v_m(v) { }                        \
                                                                                \
        template<typename ...Args>                                              \
        KOKKOS_INLINE_FUNCTION                                                  \
        auto operator()(Args... args) const {                                   \
            return expr;                                                        \
        }                                                                       \
                                                                                \
    private:                                                                    \
        const E u_m;                                                            \
        const T v_m;                                                            \
    };                                                                          \
                                                                                \
    template<typename T, typename E, size_t N>                                  \
    fun<T, E> op(const FieldExpr<T, E, N>& u,                                   \
                 const T& v) {                                                  \
        return fun<T, E>(*static_cast<const E*>(&u),                            \
                         v);                                                    \
    }

    DefineFieldScalarOperation(LFieldAddScalarRight,      operator+, u_m(args...) + v_m)
    DefineFieldScalarOperation(LFieldSubtractScalarRight, operator-, u_m(args...) - v_m)
    DefineFieldScalarOperation(LFieldMultiplyScalarRight, operator*, u_m(args...) * v_m)
    DefineFieldScalarOperation(LFieldDivideScalarRight,   operator/, u_m(args...) / v_m)


    /*
     * Scalar-Field operations when T is a Vektor.
     *
     */
    #define DefineVectorFieldScalarRightOperation(fun, op, expr)                \
    template<typename T, typename E>                                            \
    struct fun : public FieldExpr<T, fun<T, E>, sizeof(E) + sizeof(T)> {        \
        fun(const typename T::value_t& u, const E& v) : u_m(u), v_m(v) { }      \
                                                                                \
        template<typename ...Args>                                              \
        KOKKOS_INLINE_FUNCTION                                                  \
        auto operator()(Args... args) const {                                   \
            return expr;                                                        \
        }                                                                       \
                                                                                \
    private:                                                                    \
        const typename T::value_t u_m;                                          \
        const E v_m;                                                            \
    };                                                                          \
                                                                                \
    template<typename T, typename E, size_t N>                                  \
    fun<T, E> op(const typename T::value_t& u,                                  \
                 const FieldExpr<T, E, N>& v) {                                 \
        return fun<T, E>(u,                                                     \
                         *static_cast<const E*>(&v));                           \
    }


    DefineVectorFieldScalarRightOperation(LVectorFieldAddScalarRight,      operator+, u_m + v_m(args...))
    DefineVectorFieldScalarRightOperation(LVectorFieldSubractScalarRight,  operator-, u_m - v_m(args...))
    DefineVectorFieldScalarRightOperation(LVectorFieldMultiplyScalarRight, operator*, u_m * v_m(args...))
    DefineVectorFieldScalarRightOperation(LVectorFieldDivideScalarRight,   operator/, u_m / v_m(args...))



    #define DefineVectorFieldScalarLeftOperation(fun, op, expr)                 \
    template<typename E, typename T>                                            \
    struct fun : public FieldExpr<T, fun<E, T>, sizeof(E) + sizeof(T)> {        \
        fun(const E& u, const typename T::value_t& v) : u_m(u), v_m(v) { }      \
                                                                                \
        template<typename ...Args>                                              \
        KOKKOS_INLINE_FUNCTION                                                  \
        auto operator()(Args... args) const {                                   \
            return expr;                                                        \
        }                                                                       \
                                                                                \
    private:                                                                    \
        const E u_m;                                                            \
        const typename T::value_t v_m;                                          \
    };                                                                          \
                                                                                \
    template<typename E, typename T, size_t N>                                  \
    fun<E, T> op(const FieldExpr<T, E, N>& u,                                   \
                 const typename T::value_t& v) {                                \
        return fun<E, T>(*static_cast<const E*>(&u),                            \
                         v);                                                    \
    }


    DefineVectorFieldScalarLeftOperation(LVectorFieldAddScalarLeft,      operator+, u_m(args...) + v_m)
    DefineVectorFieldScalarLeftOperation(LVectorFieldSubractScalarLeft,  operator-, u_m(args...) - v_m)
    DefineVectorFieldScalarLeftOperation(LVectorFieldMultiplyScalarLeft, operator*, u_m(args...) * v_m)
    DefineVectorFieldScalarLeftOperation(LVectorFieldDivideScalarLeft,   operator/, u_m(args...) / v_m)
}

#endif