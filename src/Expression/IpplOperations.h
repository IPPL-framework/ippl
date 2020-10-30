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

#include "Types/Vector.h"

namespace ippl {
    /*!
     * @file IpplOperations.h
     */

    #define DefineUnaryOperation(fun, name, op1, op2)                       \
    template<typename E>                                                    \
    struct fun : public Expression<fun<E>, sizeof(E)> {                     \
        KOKKOS_FUNCTION                                                     \
        fun(const E& u) : u_m(u) { }                                        \
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
        const E u_m;                                                        \
    };                                                                      \
                                                                            \
    template<typename E, size_t N>                                          \
    KOKKOS_INLINE_FUNCTION                                                  \
    fun<E> name(const Expression<E, N>& u) {                                \
        return fun<E>(*static_cast<const E*>(&u));                          \
    }                                                                       \

    /// @cond

    DefineUnaryOperation(UnaryMinus, operator-, -u_m[i],  -u_m(args...)) 
    DefineUnaryOperation(UnaryPlus,  operator+, +u_m[i],  +u_m(args...)) 
    DefineUnaryOperation(BitwiseNot, operator~, ~u_m[i],  ~u_m(args...)) 
    DefineUnaryOperation(Not,        operator!, !u_m[i],  !u_m(args...)) 
    
    DefineUnaryOperation(ArcCos, acos,  acos(u_m[i]),  acos(u_m(args...))) 
    DefineUnaryOperation(ArcSin, asin,  asin(u_m[i]),  asin(u_m(args...))) 
    DefineUnaryOperation(ArcTan, atan,  atan(u_m[i]),  atan(u_m(args...))) 
    DefineUnaryOperation(Ceil,   ceil,  ceil(u_m[i]),  ceil(u_m(args...))) 
    DefineUnaryOperation(Cos,    cos,   cos(u_m[i]),   cos(u_m(args...))) 
    DefineUnaryOperation(HypCos, cosh,  cosh(u_m[i]),  cosh(u_m(args...))) 
    DefineUnaryOperation(Exp,    exp,   exp(u_m[i]),   exp(u_m(args...))) 
    DefineUnaryOperation(Fabs,   fabs,  fabs(u_m[i]),  fabs(u_m(args...))) 
    DefineUnaryOperation(Floor,  floor, floor(u_m[i]), floor(u_m(args...))) 
    DefineUnaryOperation(Log,    log,   log(u_m[i]),   log(u_m(args...))) 
    DefineUnaryOperation(Log10,  log10, log10(u_m[i]), log10(u_m(args...))) 
    DefineUnaryOperation(Sin,    sin,   sin(u_m[i]),   sin(u_m(args...))) 
    DefineUnaryOperation(HypSin, sinh,  sinh(u_m[i]),  sinh(u_m(args...))) 
    DefineUnaryOperation(Sqrt,   sqrt,  sqrt(u_m[i]),  sqrt(u_m(args...))) 
    DefineUnaryOperation(Tan,    tan,   tan(u_m[i]),   tan(u_m(args...))) 
    DefineUnaryOperation(HypTan, tanh,  tanh(u_m[i]),  tanh(u_m(args...))) 
    DefineUnaryOperation(Erf,    erf,   erf(u_m[i]),   erf(u_m(args...))) 
    /// @endcond


    /*!
     * Macro to overload C++ operators for the Scalar, LField and Vector class.
     * @param fun name of the expression template function
     * @param name overloaded operator
     * @param op1 operation for single index access
     * @param op2 operation for multiple indices access
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
    fun<E, detail::Scalar<T>> name(const Expression<E, N>& u,               \
                                   const T& v) {                            \
        return fun<E, detail::Scalar<T>>(*static_cast<const E*>(&u), v);    \
    }                                                                       \
                                                                            \
    template<typename E, size_t N, typename T,                              \
             typename = std::enable_if_t<std::is_scalar<T>::value>>         \
    KOKKOS_INLINE_FUNCTION                                                  \
    fun<detail::Scalar<T>, E> name(const T& u,                              \
                                   const Expression<E, N>& v) {             \
        return fun<detail::Scalar<T>, E>(u, *static_cast<const E*>(&v));    \
    }

    /// @cond
    DefineBinaryOperation(Add,      operator+,  u_m[i] + v_m[i],  u_m(args...) + v_m(args...))
    DefineBinaryOperation(Subtract, operator-,  u_m[i] - v_m[i],  u_m(args...) - v_m(args...))
    DefineBinaryOperation(Multiply, operator*,  u_m[i] * v_m[i],  u_m(args...) * v_m(args...))
    DefineBinaryOperation(Divide,   operator/,  u_m[i] / v_m[i],  u_m(args...) / v_m(args...))
    DefineBinaryOperation(Mod,      operator%,  u_m[i] % v_m[i],  u_m(args...) % v_m(args...))
    DefineBinaryOperation(LT,       operator<,  u_m[i] < v_m[i],  u_m(args...) < v_m(args...))
    DefineBinaryOperation(LE,       operator<=, u_m[i] <= v_m[i], u_m(args...) <= v_m(args...))
    DefineBinaryOperation(GT,       operator>,  u_m[i] > v_m[i],  u_m(args...) > v_m(args...))
    DefineBinaryOperation(GE,       operator>=, u_m[i] >= v_m[i], u_m(args...) >= v_m(args...))
    DefineBinaryOperation(EQ,       operator==, u_m[i] == v_m[i], u_m(args...) == v_m(args...))
    DefineBinaryOperation(NEQ,      operator!=, u_m[i] != v_m[i], u_m(args...) != v_m(args...))
    DefineBinaryOperation(And,      operator&&, u_m[i] && v_m[i], u_m(args...) && v_m(args...))
    DefineBinaryOperation(Or,       operator||, u_m[i] || v_m[i], u_m(args...) || v_m(args...))
    
    DefineBinaryOperation(BitwiseAnd, operator&, u_m[i] & v_m[i], u_m(args...) & v_m(args...))
    DefineBinaryOperation(BitwiseOr,  operator|, u_m[i] | v_m[i], u_m(args...) | v_m(args...))
    DefineBinaryOperation(BitwiseXor, operator^, u_m[i] ^ v_m[i], u_m(args...) ^ v_m(args...))
   
    DefineBinaryOperation(Copysign, copysign, copysign(u_m[i],v_m[i]), 
                          copysign(u_m(args...),v_m(args...)))
    DefineBinaryOperation(Ldexp, ldexp, ldexp(u_m[i],v_m[i]), 
                          ldexp(u_m(args...),v_m(args...)))
    DefineBinaryOperation(Fmod, fmod, fmod(u_m[i],v_m[i]), 
                          fmod(u_m(args...),v_m(args...)))
    DefineBinaryOperation(Pow, pow, pow(u_m[i],v_m[i]), pow(u_m(args...),v_m(args...)))
    DefineBinaryOperation(ArcTan2, atan2, atan2(u_m[i],v_m[i]), 
                          atan2(u_m(args...),v_m(args...)))
    /// @endcond


    namespace detail {
        /*!
         * Meta function of cross product. This function is only supported for 3-dimensional vectors.
         */
        template<typename E1, typename E2>
        struct meta_cross : public Expression<meta_cross<E1, E2>, sizeof(E1) + sizeof(E2)> {
            KOKKOS_FUNCTION
            meta_cross(const E1& u, const E2& v) : u_m(u), v_m(v) { }

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
    }

    template<typename E1, size_t N1, typename E2, size_t N2>
    KOKKOS_INLINE_FUNCTION
    detail::meta_cross<E1, E2> cross(const Expression<E1, N1>& u,
                                     const Expression<E2, N2>& v) {
        return detail::meta_cross<E1, E2>(*static_cast<const E1*>(&u),
                                          *static_cast<const E2*>(&v));
    }

    namespace detail {
        /*!
         * Meta function of dot product.
         */
        template<typename E1, typename E2>
        struct meta_dot : public Expression<meta_dot<E1, E2>, sizeof(E1) + sizeof(E2)> {
            KOKKOS_FUNCTION
            meta_dot(const E1& u, const E2& v) : u_m(u), v_m(v) { }

            /*
             * Vector::dot
             */
            KOKKOS_INLINE_FUNCTION
            auto apply() const {
                typename E1::value_type res = 0.0;
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
                return dot(u_m(args...), v_m(args...)).apply();
            }

        private:
            const E1 u_m;
            const E2 v_m;
        };
    }

    template<typename E1, size_t N1, typename E2, size_t N2>
    KOKKOS_INLINE_FUNCTION
    detail::meta_dot<E1, E2> dot(const Expression<E1, N1>& u,
                                 const Expression<E2, N2>& v) {
        return detail::meta_dot<E1, E2>(*static_cast<const E1*>(&u),
                                        *static_cast<const E2*>(&v));
    }

    namespace detail {
        /*!
         * Meta function of gradient
         */

        template<typename E, typename Evec>
        struct meta_grad : public Expression<meta_grad<E, Evec>, sizeof(E) + 3 * sizeof(Evec)> {

            KOKKOS_FUNCTION
            meta_grad(const E& u,
                    const Evec& xvector,
                    const Evec& yvector,
                    const Evec& zvector)
            : u_m(u)
            , xvector_m(xvector)
            , yvector_m(yvector)
            , zvector_m(zvector)
            { }


            /*
             * 3-dimensional grad
             */
            KOKKOS_INLINE_FUNCTION
            auto operator()(size_t i, size_t j, size_t k) const {

                return xvector_m * (u_m(i+1, j,   k)   - u_m(i-1, j,   k  )) +
                       yvector_m * (u_m(i  , j+1, k)   - u_m(i  , j-1, k  )) +
                       zvector_m * (u_m(i  , j  , k+1) - u_m(i  , j  , k-1));
            }

        private:
            const E u_m;
            const Evec xvector_m;
            const Evec yvector_m;
            const Evec zvector_m;
        };
    }

    /*!
     * User interface of gradient in three dimensions.
     * @tparam E expression type of left-hand side
     * @tparam N size of expression
     * @tparam Evec vector as an expression
     * @tparam Nvec size of vector expression
     * @param u expression
     * @param xvector
     * @param yvector
     * @param zvector
     */

    template<typename E, size_t N, typename Evec, size_t Nvec>
    KOKKOS_INLINE_FUNCTION
    detail::meta_grad<E, Evec> grad(const Expression<E, N>& u, 
                                    const Expression<Evec, Nvec>& xvector, 
                                    const Expression<Evec, Nvec>& yvector, 
                                    const Expression<Evec, Nvec>& zvector) {
        return detail::meta_grad<E, Evec>(*static_cast<const E*>(&u), xvector, yvector, zvector);
    }

    namespace detail {

        /*!
         * Meta function of divergence
         */
        template<typename E, typename Evec>
        struct meta_div : public Expression<meta_div<E, Evec>, sizeof(E) + 3 * sizeof(Evec)> {

            KOKKOS_FUNCTION
            meta_div(const E& u,
                    const Evec& xvector,
                    const Evec& yvector,
                    const Evec& zvector)
            : u_m(u)
            , xvector_m(xvector)
            , yvector_m(yvector)
            , zvector_m(zvector)
            { }


            /*
             * 3-dimensional div
             */
            KOKKOS_INLINE_FUNCTION
            auto operator()(size_t i, size_t j, size_t k) const {
                return dot(xvector_m, (u_m(i+1, j,   k)   - u_m(i-1, j,   k  ))).apply() +
                       dot(yvector_m, (u_m(i  , j+1, k)   - u_m(i  , j-1, k  ))).apply() +
                       dot(zvector_m, (u_m(i  , j  , k+1) - u_m(i  , j  , k-1))).apply();
            }

        private:
            const E u_m;
            const Evec xvector_m;
            const Evec yvector_m;
            const Evec zvector_m;
        };
    }


    /*!
     * User interface of divergence in three dimensions.
     * @tparam E expression type of left-hand side
     * @tparam N size of expression
     * @tparam Evec vector as an expression
     * @tparam Nvec size of vector expression
     * @param u expression
     * @param xvector
     * @param yvector
     * @param zvector
     */
    template<typename E, size_t N, typename Evec, size_t Nvec>
    KOKKOS_INLINE_FUNCTION
    detail::meta_div<E, Evec> div(const Expression<E, N>& u, 
                                  const Expression<Evec, Nvec>& xvector, 
                                  const Expression<Evec, Nvec>& yvector, 
                                  const Expression<Evec, Nvec>& zvector) {
        return detail::meta_div<E, Evec>(*static_cast<const E*>(&u), xvector, yvector, zvector);
    }


    namespace detail {

        /*!
         * Meta function of Laplacian 
         */
        template<typename E, typename Evec>
        struct meta_laplace : public Expression<meta_laplace<E, Evec>, sizeof(E) + sizeof(Evec)> {

            KOKKOS_FUNCTION
            meta_laplace(const E& u,
                         const Evec& hvector)
            : u_m(u)
            , hvector_m(hvector)
            { }


            /*
             * 3-dimensional Laplacian
             */
            KOKKOS_INLINE_FUNCTION
            auto operator()(size_t i, size_t j, size_t k) const {
                
                return hvector_m[0] * (u_m(i+1, j,   k)   - 2 * u_m(i, j, k) + u_m(i-1, j,   k  )) +
                       hvector_m[1] * (u_m(i  , j+1, k)   - 2 * u_m(i, j, k) + u_m(i  , j-1, k  )) +
                       hvector_m[2] * (u_m(i  , j  , k+1) - 2 * u_m(i, j, k) + u_m(i  , j  , k-1));
            }

        private:
            const E u_m;
            const Evec hvector_m;
        };
    }


    /*!
     * User interface of Laplacian in three dimensions.
     * @tparam E expression type of left-hand side
     * @tparam N size of expression
     * @tparam Evec vector as an expression
     * @tparam Nvec size of vector expression
     * @param u expression
     * @param hvector
     */
    template<typename E, size_t N, typename Evec, size_t Nvec>
    KOKKOS_INLINE_FUNCTION
    detail::meta_laplace<E, Evec> laplace(const Expression<E, N>& u, const Expression<Evec, Nvec>& hvector) {
        return detail::meta_laplace<E, Evec>(*static_cast<const E*>(&u), hvector);
    }


}

#endif
