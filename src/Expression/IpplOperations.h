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

#include <tuple>

#include "Types/Vector.h"

namespace ippl {
    /*!
     * @file IpplOperations.h
     */

    template <typename View, typename Coords, size_t... Idx>
    KOKKOS_INLINE_FUNCTION constexpr decltype(auto) apply_impl(const View& view,
                                                               const Coords& coords,
                                                               std::index_sequence<Idx...>) {
        return view(coords[Idx]...);
    }

    template <unsigned Dim, typename View, typename Coords>
    KOKKOS_INLINE_FUNCTION constexpr decltype(auto) apply(const View& view, const Coords& coords) {
        using Indices = std::make_index_sequence<Dim>;
        return apply_impl(view, coords, Indices{});
    }

#define DefineUnaryOperation(fun, name, op1, op2)                              \
    template <typename E>                                                      \
    struct fun : public detail::Expression<fun<E>, sizeof(E)> {                \
        KOKKOS_FUNCTION                                                        \
        fun(const E& u)                                                        \
            : u_m(u) {}                                                        \
                                                                               \
        KOKKOS_INLINE_FUNCTION auto operator[](size_t i) const { return op1; } \
                                                                               \
        template <typename... Args>                                            \
        KOKKOS_INLINE_FUNCTION auto operator()(Args... args) const {           \
            return op2;                                                        \
        }                                                                      \
                                                                               \
    private:                                                                   \
        const E u_m;                                                           \
    };                                                                         \
                                                                               \
    template <typename E, size_t N>                                            \
    KOKKOS_INLINE_FUNCTION fun<E> name(const detail::Expression<E, N>& u) {    \
        return fun<E>(*static_cast<const E*>(&u));                             \
    }

    /// @cond

    DefineUnaryOperation(UnaryMinus, operator-, -u_m[i], -u_m(args...))
    DefineUnaryOperation(UnaryPlus, operator+, +u_m[i], +u_m(args...))
    DefineUnaryOperation(BitwiseNot, operator~, ~u_m[i], ~u_m(args...))
    DefineUnaryOperation(Not, operator!, !u_m[i], !u_m(args...))

    DefineUnaryOperation(ArcCos, acos, acos(u_m[i]), acos(u_m(args...)))
    DefineUnaryOperation(ArcSin, asin, asin(u_m[i]), asin(u_m(args...)))
    DefineUnaryOperation(ArcTan, atan, atan(u_m[i]), atan(u_m(args...)))
    DefineUnaryOperation(Ceil, ceil, ceil(u_m[i]), ceil(u_m(args...)))
    DefineUnaryOperation(Cos, cos, cos(u_m[i]), cos(u_m(args...)))
    DefineUnaryOperation(HypCos, cosh, cosh(u_m[i]), cosh(u_m(args...)))
    DefineUnaryOperation(Exp, exp, exp(u_m[i]), exp(u_m(args...)))
    DefineUnaryOperation(Fabs, fabs, fabs(u_m[i]), fabs(u_m(args...)))
    DefineUnaryOperation(Floor, floor, floor(u_m[i]), floor(u_m(args...)))
    DefineUnaryOperation(Log, log, log(u_m[i]), log(u_m(args...)))
    DefineUnaryOperation(Log10, log10, log10(u_m[i]), log10(u_m(args...)))
    DefineUnaryOperation(Sin, sin, sin(u_m[i]), sin(u_m(args...)))
    DefineUnaryOperation(HypSin, sinh, sinh(u_m[i]), sinh(u_m(args...)))
    DefineUnaryOperation(Sqrt, sqrt, sqrt(u_m[i]), sqrt(u_m(args...)))
    DefineUnaryOperation(Tan, tan, tan(u_m[i]), tan(u_m(args...)))
    DefineUnaryOperation(HypTan, tanh, tanh(u_m[i]), tanh(u_m(args...)))
    DefineUnaryOperation(Erf, erf, erf(u_m[i]), erf(u_m(args...)))
/// @endcond

/*!
 * Macro to overload C++ operators for the Scalar, BareField and Vector class.
 * @param fun name of the expression template function
 * @param name overloaded operator
 * @param op1 operation for single index access
 * @param op2 operation for multiple indices access
 */
#define DefineBinaryOperation(fun, name, op1, op2)                                             \
    template <typename E1, typename E2>                                                        \
    struct fun : public detail::Expression<fun<E1, E2>, sizeof(E1) + sizeof(E2)> {             \
        KOKKOS_FUNCTION                                                                        \
        fun(const E1& u, const E2& v)                                                          \
            : u_m(u)                                                                           \
            , v_m(v) {}                                                                        \
                                                                                               \
        KOKKOS_INLINE_FUNCTION auto operator[](size_t i) const { return op1; }                 \
                                                                                               \
        template <typename... Args>                                                            \
        KOKKOS_INLINE_FUNCTION auto operator()(Args... args) const {                           \
            return op2;                                                                        \
        }                                                                                      \
                                                                                               \
    private:                                                                                   \
        const E1 u_m;                                                                          \
        const E2 v_m;                                                                          \
    };                                                                                         \
                                                                                               \
    template <typename E1, size_t N1, typename E2, size_t N2>                                  \
    KOKKOS_INLINE_FUNCTION fun<E1, E2> name(const detail::Expression<E1, N1>& u,               \
                                            const detail::Expression<E2, N2>& v) {             \
        return fun<E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));          \
    }                                                                                          \
                                                                                               \
    template <typename E, size_t N, typename T,                                                \
              typename = std::enable_if_t<std::is_scalar<T>::value>>                           \
    KOKKOS_INLINE_FUNCTION fun<E, detail::Scalar<T>> name(const detail::Expression<E, N>& u,   \
                                                          const T& v) {                        \
        return fun<E, detail::Scalar<T>>(*static_cast<const E*>(&u), v);                       \
    }                                                                                          \
                                                                                               \
    template <typename E, size_t N, typename T,                                                \
              typename = std::enable_if_t<std::is_scalar<T>::value>>                           \
    KOKKOS_INLINE_FUNCTION fun<detail::Scalar<T>, E> name(const T& u,                          \
                                                          const detail::Expression<E, N>& v) { \
        return fun<detail::Scalar<T>, E>(u, *static_cast<const E*>(&v));                       \
    }

    /// @cond
    DefineBinaryOperation(Add, operator+, u_m[i] + v_m[i], u_m(args...) + v_m(args...))
    DefineBinaryOperation(Subtract, operator-, u_m[i] - v_m[i], u_m(args...) - v_m(args...))
    DefineBinaryOperation(Multiply, operator*, u_m[i] * v_m[i], u_m(args...) * v_m(args...))
    DefineBinaryOperation(Divide, operator/, u_m[i] / v_m[i], u_m(args...) / v_m(args...))
    DefineBinaryOperation(Mod, operator%, u_m[i] % v_m[i], u_m(args...) % v_m(args...))
    DefineBinaryOperation(LT, operator<, u_m[i] < v_m[i], u_m(args...) < v_m(args...))
    DefineBinaryOperation(LE, operator<=, u_m[i] <= v_m[i], u_m(args...) <= v_m(args...))
    DefineBinaryOperation(GT, operator>, u_m[i] > v_m[i], u_m(args...) > v_m(args...))
    DefineBinaryOperation(GE, operator>=, u_m[i] >= v_m[i], u_m(args...) >= v_m(args...))
    DefineBinaryOperation(EQ, operator==, u_m[i] == v_m[i], u_m(args...) == v_m(args...))
    DefineBinaryOperation(NEQ, operator!=, u_m[i] != v_m[i], u_m(args...) != v_m(args...))
    DefineBinaryOperation(And, operator&&, u_m[i] && v_m[i], u_m(args...) && v_m(args...))
    DefineBinaryOperation(Or, operator||, u_m[i] || v_m[i], u_m(args...) || v_m(args...))

    DefineBinaryOperation(BitwiseAnd, operator&, u_m[i] & v_m[i], u_m(args...) & v_m(args...))
    DefineBinaryOperation(BitwiseOr, operator|, u_m[i] | v_m[i], u_m(args...) | v_m(args...))
    DefineBinaryOperation(BitwiseXor, operator^, u_m[i] ^ v_m[i], u_m(args...) ^ v_m(args...))

    DefineBinaryOperation(Copysign, copysign, copysign(u_m[i], v_m[i]),
                          copysign(u_m(args...), v_m(args...)))
    DefineBinaryOperation(Ldexp, ldexp, ldexp(u_m[i], v_m[i]), ldexp(u_m(args...), v_m(args...)))
    DefineBinaryOperation(Fmod, fmod, fmod(u_m[i], v_m[i]), fmod(u_m(args...), v_m(args...)))
    DefineBinaryOperation(Pow, pow, pow(u_m[i], v_m[i]), pow(u_m(args...), v_m(args...)))
    DefineBinaryOperation(ArcTan2, atan2, atan2(u_m[i], v_m[i]), atan2(u_m(args...), v_m(args...)))
    /// @endcond

    namespace detail {
        /*!
         * Meta function of cross product. This function is only supported for 3-dimensional
         * vectors.
         */
        template <typename E1, typename E2>
        struct meta_cross : public detail::Expression<meta_cross<E1, E2>, sizeof(E1) + sizeof(E2)> {
            KOKKOS_FUNCTION
            meta_cross(const E1& u, const E2& v)
                : u_m(u)
                , v_m(v) {}

            /*
             * Vector::cross
             */
            KOKKOS_INLINE_FUNCTION auto operator[](size_t i) const {
                const size_t j = (i + 1) % 3;
                const size_t k = (i + 2) % 3;
                return u_m[j] * v_m[k] - u_m[k] * v_m[j];
            }

            /*
             * This is required for BareField::cross
             */
            template <typename... Args>
            KOKKOS_INLINE_FUNCTION auto operator()(Args... args) const {
                return cross(u_m(args...), v_m(args...));
            }

        private:
            const E1 u_m;
            const E2 v_m;
        };
    }  // namespace detail

    template <typename E1, size_t N1, typename E2, size_t N2>
    KOKKOS_INLINE_FUNCTION detail::meta_cross<E1, E2> cross(const detail::Expression<E1, N1>& u,
                                                            const detail::Expression<E2, N2>& v) {
        return detail::meta_cross<E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));
    }

    namespace detail {
        /*!
         * Meta function of dot product.
         */
        template <typename E1, typename E2>
        struct meta_dot : public Expression<meta_dot<E1, E2>, sizeof(E1) + sizeof(E2)> {
            KOKKOS_FUNCTION
            meta_dot(const E1& u, const E2& v)
                : u_m(u)
                , v_m(v) {}

            /*
             * Vector::dot
             */
            KOKKOS_INLINE_FUNCTION auto apply() const {
                typename E1::value_type res = 0.0;
                for (size_t i = 0; i < E1::dim; ++i) {
                    res += u_m[i] * v_m[i];
                }
                return res;  // u_m[0] * v_m[0] + u_m[1] * v_m[1] + u_m[2] * v_m[2];
            }

            /*
             * This is required for BareField::dot
             */
            template <typename... Args>
            KOKKOS_INLINE_FUNCTION auto operator()(Args... args) const {
                return dot(u_m(args...), v_m(args...)).apply();
            }

        private:
            const E1 u_m;
            const E2 v_m;
        };
    }  // namespace detail

    template <typename E1, size_t N1, typename E2, size_t N2>
    KOKKOS_INLINE_FUNCTION detail::meta_dot<E1, E2> dot(const detail::Expression<E1, N1>& u,
                                                        const detail::Expression<E2, N2>& v) {
        return detail::meta_dot<E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));
    }

    namespace detail {
        /*!
         * Meta function of gradient
         */

        template <typename E>
        struct meta_grad
            : public Expression<meta_grad<E>,
                                sizeof(E) + 3 * sizeof(typename E::Mesh_t::vector_type)> {
            KOKKOS_FUNCTION
            meta_grad(const E& u, const typename E::Mesh_t::vector_type& xvector,
                      const typename E::Mesh_t::vector_type& yvector,
                      const typename E::Mesh_t::vector_type& zvector)
                : u_m(u)
                , xvector_m(xvector)
                , yvector_m(yvector)
                , zvector_m(zvector) {}

            /*
             * 3-dimensional grad
             */
            KOKKOS_INLINE_FUNCTION auto operator()(size_t i, size_t j, size_t k) const {
                return xvector_m * (u_m(i + 1, j, k) - u_m(i - 1, j, k))
                       + yvector_m * (u_m(i, j + 1, k) - u_m(i, j - 1, k))
                       + zvector_m * (u_m(i, j, k + 1) - u_m(i, j, k - 1));
            }

        private:
            using Mesh_t      = typename E::Mesh_t;
            using vector_type = typename Mesh_t::vector_type;
            const E u_m;
            const vector_type xvector_m;
            const vector_type yvector_m;
            const vector_type zvector_m;
        };
    }  // namespace detail

    namespace detail {

        /*!
         * Meta function of divergence
         */
        template <typename E>
        struct meta_div
            : public Expression<meta_div<E>,
                                sizeof(E) + 3 * sizeof(typename E::Mesh_t::vector_type)> {
            KOKKOS_FUNCTION
            meta_div(const E& u, const typename E::Mesh_t::vector_type& xvector,
                     const typename E::Mesh_t::vector_type& yvector,
                     const typename E::Mesh_t::vector_type& zvector)
                : u_m(u)
                , xvector_m(xvector)
                , yvector_m(yvector)
                , zvector_m(zvector) {}

            /*
             * 3-dimensional div
             */
            KOKKOS_INLINE_FUNCTION auto operator()(size_t i, size_t j, size_t k) const {
                return dot(xvector_m, (u_m(i + 1, j, k) - u_m(i - 1, j, k))).apply()
                       + dot(yvector_m, (u_m(i, j + 1, k) - u_m(i, j - 1, k))).apply()
                       + dot(zvector_m, (u_m(i, j, k + 1) - u_m(i, j, k - 1))).apply();
            }

        private:
            using Mesh_t      = typename E::Mesh_t;
            using vector_type = typename Mesh_t::vector_type;
            const E u_m;
            const vector_type xvector_m;
            const vector_type yvector_m;
            const vector_type zvector_m;
        };

        /*!
         * Meta function of Laplacian
         */
        template <typename E>
        struct meta_laplace
            : public Expression<meta_laplace<E>,
                                sizeof(E) + sizeof(typename E::Mesh_t::vector_type)> {
            KOKKOS_FUNCTION
            meta_laplace(const E& u, const typename E::Mesh_t::vector_type& hvector)
                : u_m(u)
                , hvector_m(hvector) {}

            /*
             * n-dimensional Laplacian
             */
            template <typename... Idx>
            KOKKOS_INLINE_FUNCTION auto operator()(const Idx... args) const {
                using index_type       = std::tuple_element_t<0, std::tuple<Idx...>>;
                using T                = typename E::Mesh_t::value_type;
                T res                  = 0;
                constexpr unsigned Dim = E::Mesh_t::Dimension;
                for (unsigned d = 0; d < Dim; d++) {
                    index_type coords[Dim] = {args...};
                    auto&& center          = apply<Dim>(u_m, coords);

                    coords[d] -= 1;
                    auto&& left = apply<Dim>(u_m, coords);

                    coords[d] += 2;
                    auto&& right = apply<Dim>(u_m, coords);

                    res += hvector_m[d] * (left - 2 * center + right);
                }
                return res;

                /*
                return hvector_m[0] * (u_m(i+1, j,   k)   - 2 * u_m(i, j, k) + u_m(i-1, j,   k  )) +
                       hvector_m[1] * (u_m(i  , j+1, k)   - 2 * u_m(i, j, k) + u_m(i  , j-1, k  )) +
                       hvector_m[2] * (u_m(i  , j  , k+1) - 2 * u_m(i, j, k) + u_m(i  , j  , k-1));
                       */
            }

        private:
            using Mesh_t      = typename E::Mesh_t;
            using vector_type = typename Mesh_t::vector_type;
            const E u_m;
            const vector_type hvector_m;
        };
    }  // namespace detail

    namespace detail {
        /*!
         * Meta function of curl
         */

        template <typename E>
        struct meta_curl
            : public Expression<meta_curl<E>,
                                sizeof(E) + 4 * sizeof(typename E::Mesh_t::vector_type)> {
            KOKKOS_FUNCTION
            meta_curl(const E& u, const typename E::Mesh_t::vector_type& xvector,
                      const typename E::Mesh_t::vector_type& yvector,
                      const typename E::Mesh_t::vector_type& zvector,
                      const typename E::Mesh_t::vector_type& hvector)
                : u_m(u)
                , xvector_m(xvector)
                , yvector_m(yvector)
                , zvector_m(zvector)
                , hvector_m(hvector) {}

            /*
             * 3-dimensional curl
             */
            KOKKOS_INLINE_FUNCTION auto operator()(size_t i, size_t j, size_t k) const {
                return xvector_m
                           * ((u_m(i, j + 1, k)[2] - u_m(i, j - 1, k)[2]) / (2 * hvector_m[1])
                              - (u_m(i, j, k + 1)[1] - u_m(i, j, k - 1)[1]) / (2 * hvector_m[2]))
                       + yvector_m
                             * ((u_m(i, j, k + 1)[0] - u_m(i, j, k - 1)[0]) / (2 * hvector_m[2])
                                - (u_m(i + 1, j, k)[2] - u_m(i - 1, j, k)[2]) / (2 * hvector_m[0]))
                       + zvector_m
                             * ((u_m(i + 1, j, k)[1] - u_m(i - 1, j, k)[1]) / (2 * hvector_m[0])
                                - (u_m(i, j + 1, k)[0] - u_m(i, j - 1, k)[0]) / (2 * hvector_m[1]));
            }

        private:
            using Mesh_t      = typename E::Mesh_t;
            using vector_type = typename Mesh_t::vector_type;
            const E u_m;
            const vector_type xvector_m;
            const vector_type yvector_m;
            const vector_type zvector_m;
            const vector_type hvector_m;
        };
    }  // namespace detail

    namespace detail {

        /*!
         * Meta function of Hessian
         */
        template <typename E>
        struct meta_hess
            : public Expression<meta_hess<E>,
                                sizeof(E) + 4 * sizeof(typename E::Mesh_t::vector_type)> {
            KOKKOS_FUNCTION
            meta_hess(const E& u, const typename E::Mesh_t::vector_type& xvector,
                      const typename E::Mesh_t::vector_type& yvector,
                      const typename E::Mesh_t::vector_type& zvector,
                      const typename E::Mesh_t::vector_type& hvector)
                : u_m(u)
                , xvector_m(xvector)
                , yvector_m(yvector)
                , zvector_m(zvector)
                , hvector_m(hvector) {}

            /*
             * 3-dimensional hessian (return Vector<Vector<T,3>,3>)
             */
            KOKKOS_INLINE_FUNCTION auto operator()(size_t i, size_t j, size_t k) const {
                vector_type row_1, row_2, row_3;

                row_1 = xvector_m
                            * ((u_m(i + 1, j, k) - 2.0 * u_m(i, j, k) + u_m(i - 1, j, k))
                               / (hvector_m[0] * hvector_m[0]))
                        + yvector_m
                              * ((u_m(i + 1, j + 1, k) - u_m(i - 1, j + 1, k) - u_m(i + 1, j - 1, k)
                                  + u_m(i - 1, j - 1, k))
                                 / (4.0 * hvector_m[0] * hvector_m[1]))
                        + zvector_m
                              * ((u_m(i + 1, j, k + 1) - u_m(i - 1, j, k + 1) - u_m(i + 1, j, k - 1)
                                  + u_m(i - 1, j, k - 1))
                                 / (4.0 * hvector_m[0] * hvector_m[2]));

                row_2 = xvector_m
                            * ((u_m(i + 1, j + 1, k) - u_m(i + 1, j - 1, k) - u_m(i - 1, j + 1, k)
                                + u_m(i - 1, j - 1, k))
                               / (4.0 * hvector_m[1] * hvector_m[0]))
                        + yvector_m
                              * ((u_m(i, j + 1, k) - 2.0 * u_m(i, j, k) + u_m(i, j - 1, k))
                                 / (hvector_m[1] * hvector_m[1]))
                        + zvector_m
                              * ((u_m(i, j + 1, k + 1) - u_m(i, j - 1, k + 1) - u_m(i, j + 1, k - 1)
                                  + u_m(i, j - 1, k - 1))
                                 / (4.0 * hvector_m[1] * hvector_m[2]));

                row_3 = xvector_m
                            * ((u_m(i + 1, j, k + 1) - u_m(i + 1, j, k - 1) - u_m(i - 1, j, k + 1)
                                + u_m(i - 1, j, k - 1))
                               / (4.0 * hvector_m[2] * hvector_m[0]))
                        + yvector_m
                              * ((u_m(i, j + 1, k + 1) - u_m(i, j + 1, k - 1) - u_m(i, j - 1, k + 1)
                                  + u_m(i, j - 1, k - 1))
                                 / (4.0 * hvector_m[2] * hvector_m[1]))
                        + zvector_m
                              * ((u_m(i, j, k + 1) - 2.0 * u_m(i, j, k) + u_m(i, j, k - 1))
                                 / (hvector_m[2] * hvector_m[2]));

                matrix_type hessian = {row_1, row_2, row_3};
                return hessian;
            }

        private:
            using Mesh_t      = typename E::Mesh_t;
            using vector_type = typename Mesh_t::vector_type;
            using matrix_type = typename Mesh_t::matrix_type;
            const E u_m;
            const vector_type xvector_m;
            const vector_type yvector_m;
            const vector_type zvector_m;
            const vector_type hvector_m;
        };
    }  // namespace detail
}  // namespace ippl

#endif
