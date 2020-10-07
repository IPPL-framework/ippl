//
// File IpplFieldOperations.h
//   Expression Templates BareField operations.
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
#ifndef IPPL_FIELD_OPERATIONS_H
#define IPPL_FIELD_OPERATIONS_H

namespace ippl {
    #define DefineBinaryFieldOperation(fun, name, op1, op2)                     \
    template<typename E1, typename E2>                                          \
    struct fun : public FieldExpression<fun<E1, E2>> {                          \
        fun(const E1& u, const E2& v) : u_m(u), v_m(v) { }                      \
                                                                                \
        auto operator[](size_t i) const { return op1; }                         \
                                                                                \
        template<typename ...Args>                                              \
        auto operator()(Args... args) const {                                   \
            return op2;                                                         \
        }                                                                       \
                                                                                \
    private:                                                                    \
        const E1 u_m;                                                           \
        const E2 v_m;                                                           \
    };                                                                          \
                                                                                \
    template<typename E1, typename E2>                                          \
    fun<E1, E2> name(const FieldExpression<E1>& u,                              \
                     const FieldExpression<E2>& v) {                            \
        return fun<E1, E2>(*static_cast<const E1*>(&u),                         \
                           *static_cast<const E2*>(&v));                        \
    }                                                                           \
                                                                                \
    template<typename E, typename T,                                            \
             typename = std::enable_if_t<std::is_scalar<T>::value>>             \
    fun<E, Scalar<T>> name(const FieldExpression<E>& u,                         \
                           const T& v) {                                        \
        return fun<E, Scalar<T>>(*static_cast<const E*>(&u), v);                \
    }                                                                           \
                                                                                \
    template<typename E, typename T,                                            \
             typename = std::enable_if_t<std::is_scalar<T>::value>>             \
    fun<E, Scalar<T>> name(const T& u,                                          \
                           const FieldExpression<E>& v) {                       \
        return fun<E, Scalar<T>>(*static_cast<const E*>(&v), u);                \
    }


    DefineBinaryFieldOperation(FieldAdd,      operator+, u_m[i] + v_m[i], u_m(args...) + v_m(args...))
    DefineBinaryFieldOperation(FieldSubtract, operator-, u_m[i] - v_m[i], u_m(args...) - v_m(args...))
    DefineBinaryFieldOperation(FieldMultiply, operator*, u_m[i] * v_m[i], u_m(args...) * v_m(args...))
    DefineBinaryFieldOperation(FieldDivide,   operator/, u_m[i] / v_m[i], u_m(args...) / v_m(args...))


    /*
     * Cross product. This function is only supported for 3-dimensional vectors.
     */

    template<typename E1, typename E2>
    struct field_meta_cross : public FieldExpression<field_meta_cross<E1, E2>> {
        field_meta_cross(const E1& u, const E2& v) : u_m(u), v_m(v) {
        }

        auto operator[](size_t i) const {
            return  cross(u_m[i], v_m[i]);
        }

    private:
        const E1 u_m;
        const E2 v_m;
    };

    template<typename E1, typename E2>
    field_meta_cross<E1, E2> cross(const FieldExpression<E1>& u, const FieldExpression<E2>& v) {
        return field_meta_cross<E1, E2>(*static_cast<const E1*>(&u),
                                        *static_cast<const E2*>(&v));
    }
}

#endif