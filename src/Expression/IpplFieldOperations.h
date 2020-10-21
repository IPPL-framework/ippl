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

    #define DefineUnaryFieldOperation(fun, name, op1, op2)                  \
    template<typename E>                                                    \
    struct fun : public FieldExpression<fun<E>> {                           \
        fun(const E& u) : u_m(u) { }                                        \
                                                                            \
        auto operator[](size_t i) const { return op1; }                     \
                                                                            \
        template<typename ...Args>                                          \
        auto operator()(Args... args) const {                               \
            return op2;                                                     \
        }                                                                   \
                                                                            \
    private:                                                                \
        const E u_m;                                                        \
    };                                                                      \
                                                                            \
    template<typename E>                                                    \
    fun<E> name(const FieldExpression<E>& u) {                              \
        return fun<E>(*static_cast<const E*>(&u));                          \
    }                                                                       \

    /// @cond

    DefineUnaryFieldOperation(FieldUnaryMinus, operator-, -u_m[i],  -u_m(args...)) 
    DefineUnaryFieldOperation(FieldUnaryPlus,  operator+, +u_m[i],  +u_m(args...)) 
    DefineUnaryFieldOperation(FieldBitwiseNot, operator~, ~u_m[i],  ~u_m(args...)) 
    DefineUnaryFieldOperation(FieldNot,        operator!, !u_m[i],  !u_m(args...)) 
    
    DefineUnaryFieldOperation(FieldArcCos, acos,  acos(u_m[i]),  acos(u_m(args...))) 
    DefineUnaryFieldOperation(FieldArcSin, asin,  asin(u_m[i]),  asin(u_m(args...))) 
    DefineUnaryFieldOperation(FieldArcTan, atan,  atan(u_m[i]),  atan(u_m(args...))) 
    DefineUnaryFieldOperation(FieldCeil,   ceil,  ceil(u_m[i]),  ceil(u_m(args...))) 
    DefineUnaryFieldOperation(FieldCos,    cos,   cos(u_m[i]),   cos(u_m(args...))) 
    DefineUnaryFieldOperation(FieldHypCos, cosh,  cosh(u_m[i]),  cosh(u_m(args...))) 
    DefineUnaryFieldOperation(FieldExp,    exp,   exp(u_m[i]),   exp(u_m(args...))) 
    DefineUnaryFieldOperation(FieldFabs,   fabs,  fabs(u_m[i]),  fabs(u_m(args...))) 
    DefineUnaryFieldOperation(FieldFloor,  floor, floor(u_m[i]), floor(u_m(args...))) 
    DefineUnaryFieldOperation(FieldLog,    log,   log(u_m[i]),   log(u_m(args...))) 
    DefineUnaryFieldOperation(FieldLog10,  log10, log10(u_m[i]), log10(u_m(args...))) 
    DefineUnaryFieldOperation(FieldSin,    sin,   sin(u_m[i]),   sin(u_m(args...))) 
    DefineUnaryFieldOperation(FieldHypSin, sinh,  sinh(u_m[i]),  sinh(u_m(args...))) 
    DefineUnaryFieldOperation(FieldSqrt,   sqrt,  sqrt(u_m[i]),  sqrt(u_m(args...))) 
    DefineUnaryFieldOperation(FieldTan,    tan,   tan(u_m[i]),   tan(u_m(args...))) 
    DefineUnaryFieldOperation(FieldHypTan, tanh,  tanh(u_m[i]),  tanh(u_m(args...))) 
    DefineUnaryFieldOperation(FieldErf,    erf,   erf(u_m[i]),   erf(u_m(args...))) 
    /// @endcond
    
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
    fun<E, detail::Scalar<T>> name(const T& u,                                  \
                                   const FieldExpression<E>& v) {               \
        return fun<E, detail::Scalar<T>>(*static_cast<const E*>(&v), u);        \
    }


    /// @cond
    DefineBinaryFieldOperation(FieldAdd,      operator+,  u_m[i] + v_m[i],  u_m(args...) + v_m(args...))
    DefineBinaryFieldOperation(FieldSubtract, operator-,  u_m[i] - v_m[i],  u_m(args...) - v_m(args...))
    DefineBinaryFieldOperation(FieldMultiply, operator*,  u_m[i] * v_m[i],  u_m(args...) * v_m(args...))
    DefineBinaryFieldOperation(FieldDivide,   operator/,  u_m[i] / v_m[i],  u_m(args...) / v_m(args...))
    DefineBinaryFieldOperation(FieldMod,      operator%,  u_m[i] % v_m[i],  u_m(args...) % v_m(args...))
    DefineBinaryFieldOperation(FieldLT,       operator<,  u_m[i] < v_m[i],  u_m(args...) < v_m(args...))
    DefineBinaryFieldOperation(FieldLE,       operator<=, u_m[i] <= v_m[i], u_m(args...) <= v_m(args...))
    DefineBinaryFieldOperation(FieldGT,       operator>,  u_m[i] > v_m[i],  u_m(args...) > v_m(args...))
    DefineBinaryFieldOperation(FieldGE,       operator>=, u_m[i] >= v_m[i], u_m(args...) >= v_m(args...))
    DefineBinaryFieldOperation(FieldEQ,       operator==, u_m[i] == v_m[i], u_m(args...) == v_m(args...))
    DefineBinaryFieldOperation(FieldNEQ,      operator!=, u_m[i] != v_m[i], u_m(args...) != v_m(args...))
    DefineBinaryFieldOperation(FieldAnd,      operator&&, u_m[i] && v_m[i], u_m(args...) && v_m(args...))
    DefineBinaryFieldOperation(FieldOr,       operator||, u_m[i] || v_m[i], u_m(args...) || v_m(args...))
    
    DefineBinaryFieldOperation(FieldBitwiseAnd, operator&, u_m[i] & v_m[i], u_m(args...) & v_m(args...))
    DefineBinaryFieldOperation(FieldBitwiseOr,  operator|, u_m[i] | v_m[i], u_m(args...) | v_m(args...))
    DefineBinaryFieldOperation(FieldBitwiseXor, operator^, u_m[i] ^ v_m[i], u_m(args...) ^ v_m(args...))
   
    
    DefineBinaryFieldOperation(FieldCopysign, copysign, copysign(u_m[i],v_m[i]), 
                          copysign(u_m(args...),v_m(args...)))
    DefineBinaryFieldOperation(FieldLdexp, ldexp, ldexp(u_m[i],v_m[i]), 
                          ldexp(u_m(args...),v_m(args...)))
    DefineBinaryFieldOperation(FieldFmod, fmod, fmod(u_m[i],v_m[i]), 
                          fmod(u_m(args...),v_m(args...)))
    DefineBinaryFieldOperation(FieldPow, pow, pow(u_m[i],v_m[i]), pow(u_m(args...),v_m(args...)))
    DefineBinaryFieldOperation(FieldArcTan2, atan2, atan2(u_m[i],v_m[i]), 
                          atan2(u_m(args...),v_m(args...)))
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
    
    namespace detail {
        /*!
         * Meta function of dot product.
         */
        template<typename E1, typename E2>
        struct field_meta_dot : public FieldExpression<field_meta_dot<E1, E2>> {
            field_meta_dot(const E1& u, const E2& v) : u_m(u), v_m(v) { }

            auto operator[](size_t i) const {
                return  dot(u_m[i], v_m[i]);
            }

        private:
            const E1 u_m;
            const E2 v_m;
        };
    }

    template<typename E1, typename E2>
    detail::field_meta_dot<E1, E2> dot(const FieldExpression<E1>& u,
                                 const FieldExpression<E2>& v) {
        return detail::field_meta_dot<E1, E2>(*static_cast<const E1*>(&u),
                                        *static_cast<const E2*>(&v));
    }


}

#endif
