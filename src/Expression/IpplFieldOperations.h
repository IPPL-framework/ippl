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
    /*!
     * @file IpplFieldOperations.h
     */

    /*!
     * Macro to overload C++ operators for the Field and BareField class.
     * @param fun name of the expression template function
     * @param name overloaded operator
     * @param op1 operation for single index access
     * @param op2 operation for multipole indices access
     */
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
    fun<E, detail::Scalar<T>> name(const FieldExpression<E>& u,                 \
                                   const T& v) {                                \
        return fun<E, detail::Scalar<T>>(*static_cast<const E*>(&u), v);        \
    }                                                                           \
                                                                                \
    template<typename E, typename T,                                            \
             typename = std::enable_if_t<std::is_scalar<T>::value>>             \
    fun<detail::Scalar<T>, E> name(const T& u,                                  \
                                   const FieldExpression<E>& v) {               \
        return fun<detail::Scalar<T>, E>(u, *static_cast<const E*>(&v));        \
    }


    /// @cond
    DefineBinaryFieldOperation(FieldAdd,      operator+, u_m[i] + v_m[i], u_m(args...) + v_m(args...))
    DefineBinaryFieldOperation(FieldSubtract, operator-, u_m[i] - v_m[i], u_m(args...) - v_m(args...))
    DefineBinaryFieldOperation(FieldMultiply, operator*, u_m[i] * v_m[i], u_m(args...) * v_m(args...))
    DefineBinaryFieldOperation(FieldDivide,   operator/, u_m[i] / v_m[i], u_m(args...) / v_m(args...))
    /// @endcond

    namespace detail {
        /*!
         * Meta function of cross product. This function is only supported for 3-dimensional vectors.
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
    }

    /*!
     * User interface of cross product.
     * @tparam E1 expression type of left-hand side
     * @tparam E2 expression type of right-hand side
     * @param u arbitrary left-hand side vector field expression
     * @param v arbitrary right-hand side vector field expression
     */
    template<typename E1, typename E2>
    detail::field_meta_cross<E1, E2> cross(const FieldExpression<E1>& u,
                                           const FieldExpression<E2>& v) {
        return detail::field_meta_cross<E1, E2>(*static_cast<const E1*>(&u),
                                                *static_cast<const E2*>(&v));
    }
}

#endif