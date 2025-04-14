//
// File IpplOperations.h
//   Expression Templates operations.
//
#ifndef IPPL_OPERATIONS_H
#define IPPL_OPERATIONS_H

#include <Kokkos_MathematicalFunctions.hpp>
#include <tuple>

namespace ippl {
    /*!
     * @file IpplOperations.h
     */

    /*!
     * Utility function for apply (see its docstring)
     * @tparam Idx... indices of the elements to take (in practice, always the sequence of natural
     * numbers up to the dimension of the view)
     */
    template <typename View, typename Coords, size_t... Idx>
    KOKKOS_INLINE_FUNCTION constexpr decltype(auto) apply_impl(const View& view,
                                                               const Coords& coords,
                                                               const std::index_sequence<Idx...>&) {
        return view(coords[Idx]...);
    }

    /*!
     * Extracts the mathematical rank of an expression (i.e. the number of dimensions) based on
     * its type
     */
    struct ExtractExpressionRank {
        /*!
         * Extracts the extent of an array-like expression
         * @tparam Coords the array type
         * @return The array's rank
         */
        template <typename Coords, std::enable_if_t<std::is_array_v<Coords>, int> = 0>
        KOKKOS_INLINE_FUNCTION constexpr static unsigned getRank() {
            return std::extent_v<Coords>;
        }

        /*!
         * Extracts the rank of an expression type
         * @tparam Coords the expression type that evaluates to a set of coordinates
         * @return The number of dimensions in the expression
         */
        template <typename Coords, std::enable_if_t<std::is_class_v<Coords>, int> = 0>
        KOKKOS_INLINE_FUNCTION constexpr static unsigned getRank() {
            return Coords::dim;
        }
    };

    /*!
     * Accesses the element of a view at the indices contained in an array-like structure
     * instead of having the indices being separate arguments
     * @tparam View the view type
     * @tparam Coords an array-like container of indices
     * @param view the view to access
     * @param coords the indices
     * @return The element in the view at the given location
     */
    template <typename View, typename Coords>
    KOKKOS_INLINE_FUNCTION constexpr decltype(auto) apply(const View& view, const Coords& coords) {
        using Indices = std::make_index_sequence<ExtractExpressionRank::getRank<Coords>()>;
        return apply_impl(view, coords, Indices{});
    }

#define DefineUnaryOperation(fun, name, op1, op2)                              \
    template <typename E>                                                      \
    struct fun : public detail::Expression<fun<E>, sizeof(E)> {                \
        constexpr static unsigned dim = E::dim;                                \
        using value_type              = typename E::value_type;                \
                                                                               \
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

    // clang-format off
    DefineUnaryOperation(UnaryMinus, operator-, -u_m[i],  -u_m(args...))
    DefineUnaryOperation(UnaryPlus,  operator+, +u_m[i],  +u_m(args...))
    DefineUnaryOperation(BitwiseNot, operator~, ~u_m[i],  ~u_m(args...))
    DefineUnaryOperation(Not,        operator!, !u_m[i],  !u_m(args...))

    DefineUnaryOperation(ArcCos, acos,  Kokkos::acos(u_m[i]),  Kokkos::acos(u_m(args...)))
    DefineUnaryOperation(ArcSin, asin,  Kokkos::asin(u_m[i]),  Kokkos::asin(u_m(args...)))
    DefineUnaryOperation(ArcTan, atan,  Kokkos::atan(u_m[i]),  Kokkos::atan(u_m(args...)))
    DefineUnaryOperation(Ceil,   ceil,  Kokkos::ceil(u_m[i]),  Kokkos::ceil(u_m(args...)))
    DefineUnaryOperation(Cos,    cos,   Kokkos::cos(u_m[i]),   Kokkos::cos(u_m(args...)))
    DefineUnaryOperation(HypCos, cosh,  Kokkos::cosh(u_m[i]),  Kokkos::cosh(u_m(args...)))
    DefineUnaryOperation(Exp,    exp,   Kokkos::exp(u_m[i]),   Kokkos::exp(u_m(args...)))
    DefineUnaryOperation(Fabs,   fabs,  Kokkos::fabs(u_m[i]),  Kokkos::fabs(u_m(args...)))
    DefineUnaryOperation(Floor,  floor, Kokkos::floor(u_m[i]), Kokkos::floor(u_m(args...)))
    DefineUnaryOperation(Log,    log,   Kokkos::log(u_m[i]),   Kokkos::log(u_m(args...)))
    DefineUnaryOperation(Log10,  log10, Kokkos::log10(u_m[i]), Kokkos::log10(u_m(args...)))
    DefineUnaryOperation(Sin,    sin,   Kokkos::sin(u_m[i]),   Kokkos::sin(u_m(args...)))
    DefineUnaryOperation(HypSin, sinh,  Kokkos::sinh(u_m[i]),  Kokkos::sinh(u_m(args...)))
    DefineUnaryOperation(Sqrt,   sqrt,  Kokkos::sqrt(u_m[i]),  Kokkos::sqrt(u_m(args...)))
    DefineUnaryOperation(Tan,    tan,   Kokkos::tan(u_m[i]),   Kokkos::tan(u_m(args...)))
    DefineUnaryOperation(HypTan, tanh,  Kokkos::tanh(u_m[i]),  Kokkos::tanh(u_m(args...)))
    DefineUnaryOperation(Erf,    erf,   Kokkos::erf(u_m[i]),   Kokkos::erf(u_m(args...)))
// clang-format on
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
        constexpr static unsigned dim = std::max(E1::dim, E2::dim);                            \
        using value_type              = typename E1::value_type;                               \
                                                                                               \
        KOKKOS_FUNCTION                                                                        \
        fun(const E1& u, const E2& v)                                                          \
            : u_m(u)                                                                           \
            , v_m(v) {}                                                                        \
                                                                                               \
        KOKKOS_INLINE_FUNCTION auto operator[](size_t i) const { return op1; }                 \
                                                                                               \
        template <typename... Args>                                                            \
        KOKKOS_INLINE_FUNCTION auto operator()(Args... args) const {                           \
            static_assert(sizeof...(Args) == dim || dim == 0);                                 \
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
    // clang-format off
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
    // clang-format on

    DefineBinaryOperation(Copysign, copysign, Kokkos::copysign(u_m[i], v_m[i]),
                          Kokkos::copysign(u_m(args...), v_m(args...)))
    // ldexp not provided by Kokkos
    DefineBinaryOperation(Ldexp, ldexp, ldexp(u_m[i], v_m[i]), ldexp(u_m(args...), v_m(args...)))
    DefineBinaryOperation(Fmod, fmod, Kokkos::fmod(u_m[i], v_m[i]),
                          Kokkos::fmod(u_m(args...), v_m(args...)))
    DefineBinaryOperation(Pow, pow, Kokkos::pow(u_m[i], v_m[i]),
                          Kokkos::pow(u_m(args...), v_m(args...)))
    DefineBinaryOperation(ArcTan2, atan2, Kokkos::atan2(u_m[i], v_m[i]),
                          Kokkos::atan2(u_m(args...), v_m(args...)))
    /// @endcond

    namespace detail {
        /*!
         * Meta function of cross product. This function is only supported for 3-dimensional
         * vectors.
         */
        template <typename E1, typename E2>
        struct meta_cross : public detail::Expression<meta_cross<E1, E2>, sizeof(E1) + sizeof(E2)> {
            constexpr static unsigned dim = E1::dim;
            static_assert(E1::dim == E2::dim);

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
            constexpr static unsigned dim = E1::dim;
            static_assert(E1::dim == E2::dim);

            KOKKOS_FUNCTION
            meta_dot(const E1& u, const E2& v)
                : u_m(u)
                , v_m(v) {}

            /*
             * Vector::dot
             */
            KOKKOS_INLINE_FUNCTION auto apply() const {
                typename E1::value_type res = 0.0;
                // Equivalent computation in 3D:
                // u_m[0] * v_m[0] + u_m[1] * v_m[1] + u_m[2] * v_m[2]
                for (size_t i = 0; i < E1::dim; ++i) {
                    res += u_m[i] * v_m[i];
                }
                return res;
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
            : public Expression<
                  meta_grad<E>,
                  sizeof(E) + sizeof(typename E::Mesh_t::vector_type[E::Mesh_t::Dimension])> {
            constexpr static unsigned dim = E::dim;
            using value_type              = typename E::value_type;

            KOKKOS_FUNCTION
            meta_grad(const E& u, const typename E::Mesh_t::vector_type vectors[])
                : u_m(u) {
                for (unsigned d = 0; d < E::Mesh_t::Dimension; d++) {
                    vectors_m[d] = vectors[d];
                }
            }

            /*
             * n-dimensional grad
             */
            template <typename... Idx>
            KOKKOS_INLINE_FUNCTION auto operator()(const Idx... args) const {
                using index_type = std::tuple_element_t<0, std::tuple<Idx...>>;

                /*
                 * Equivalent computation in 3D:
                 *     xvector_m * (u_m(i + 1, j, k) - u_m(i - 1, j, k))
                 *   + yvector_m * (u_m(i, j + 1, k) - u_m(i, j - 1, k))
                 *   + zvector_m * (u_m(i, j, k + 1) - u_m(i, j, k - 1))
                 */

                vector_type res(0);
                for (unsigned d = 0; d < dim; d++) {
                    index_type coords[dim] = {args...};

                    coords[d] += 1;
                    auto&& right = apply(u_m, coords);

                    coords[d] -= 2;
                    auto&& left = apply(u_m, coords);

                    res += vectors_m[d] * (right - left);
                }
                return res;
            }

        private:
            using Mesh_t      = typename E::Mesh_t;
            using vector_type = typename Mesh_t::vector_type;
            const E u_m;
            vector_type vectors_m[dim];
        };
    }  // namespace detail

    namespace detail {

        /*!
         * Meta function of divergence
         */
        template <typename E>
        struct meta_div
            : public Expression<
                  meta_div<E>,
                  sizeof(E) + sizeof(typename E::Mesh_t::vector_type[E::Mesh_t::Dimension])> {
            constexpr static unsigned dim = E::dim;

            KOKKOS_FUNCTION
            meta_div(const E& u, const typename E::Mesh_t::vector_type vectors[])
                : u_m(u) {
                for (unsigned d = 0; d < E::Mesh_t::Dimension; d++) {
                    vectors_m[d] = vectors[d];
                }
            }

            /*
             * n-dimensional div
             */
            template <typename... Idx>
            KOKKOS_INLINE_FUNCTION auto operator()(const Idx... args) const {
                using index_type = std::tuple_element_t<0, std::tuple<Idx...>>;

                /*
                 * Equivalent computation in 3D:
                 *     dot(xvector_m, (u_m(i + 1, j, k) - u_m(i - 1, j, k))).apply()
                 *   + dot(yvector_m, (u_m(i, j + 1, k) - u_m(i, j - 1, k))).apply()
                 *   + dot(zvector_m, (u_m(i, j, k + 1) - u_m(i, j, k - 1))).apply()
                 */
                typename E::Mesh_t::value_type res = 0;
                for (unsigned d = 0; d < dim; d++) {
                    index_type coords[dim] = {args...};

                    coords[d] += 1;
                    auto&& right = apply(u_m, coords);

                    coords[d] -= 2;
                    auto&& left = apply(u_m, coords);

                    res += dot(vectors_m[d], right - left).apply();
                }
                return res;
            }

        private:
            using Mesh_t      = typename E::Mesh_t;
            using vector_type = typename Mesh_t::vector_type;
            const E u_m;
            vector_type vectors_m[dim];
        };

        /*!
         * Meta function of Laplacian
         */
        template <typename E>
        struct meta_laplace
            : public Expression<meta_laplace<E>,
                                sizeof(E) + sizeof(typename E::Mesh_t::vector_type)> {
            constexpr static unsigned dim = E::dim;
            using value_type              = typename E::value_type;

            KOKKOS_FUNCTION
            meta_laplace(const E& u, const typename E::Mesh_t::vector_type& hvector)
                : u_m(u)
                , hvector_m(hvector) {}

            /*
             * n-dimensional Laplacian
             */
            template <typename... Idx>
            KOKKOS_INLINE_FUNCTION auto operator()(const Idx... args) const {
                using index_type = std::tuple_element_t<0, std::tuple<Idx...>>;
                using T          = typename E::Mesh_t::value_type;

                /*
                 * Equivalent computation in 3D:
                 *     hvector_m[0] * (u_m(i+1, j,   k)   - 2 * u_m(i, j, k) + u_m(i-1, j,   k  ))
                 *   + hvector_m[1] * (u_m(i  , j+1, k)   - 2 * u_m(i, j, k) + u_m(i  , j-1, k  ))
                 *   + hvector_m[2] * (u_m(i  , j  , k+1) - 2 * u_m(i, j, k) + u_m(i  , j  , k-1))
                 */
                T res = 0;
                for (unsigned d = 0; d < dim; d++) {
                    index_type coords[dim] = {args...};
                    auto&& center          = apply(u_m, coords);

                    coords[d] -= 1;
                    auto&& left = apply(u_m, coords);

                    coords[d] += 2;
                    auto&& right = apply(u_m, coords);

                    res += hvector_m[d] * (left - 2 * center + right);
                }
                return res;
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
            constexpr static unsigned dim = E::dim;

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
                                sizeof(E)
                                    + sizeof(typename E::Mesh_t::vector_type[E::Mesh_t::Dimension])
                                    + sizeof(typename E::Mesh_t::vector_type)> {
            constexpr static unsigned dim = E::dim;

            KOKKOS_FUNCTION
            meta_hess(const E& u, const typename E::Mesh_t::vector_type vectors[],
                      const typename E::Mesh_t::vector_type& hvector)
                : u_m(u)
                , hvector_m(hvector) {
                for (unsigned d = 0; d < E::Mesh_t::Dimension; d++) {
                    vectors_m[d] = vectors[d];
                }
            }

            /*
             * n-dimensional hessian (return Vector<Vector<T,n>,n>)
             */
            template <typename... Idx>
            KOKKOS_INLINE_FUNCTION auto operator()(const Idx... args) const {
                matrix_type hessian;
                computeHessian(std::make_index_sequence<dim>{}, hessian, args...);
                return hessian;
            }

        private:
            using Mesh_t      = typename E::Mesh_t;
            using vector_type = typename Mesh_t::vector_type;
            using matrix_type = typename Mesh_t::matrix_type;

            const E u_m;
            vector_type vectors_m[dim];
            const vector_type hvector_m;

            /*!
             * Utility function for computing the Hessian. Computes the rows of the matrix
             * one by one via fold expression.
             * @tparam row... the row indices (in practice, the sequence 0...Dim - 1)
             * @tparam Idx... the indices at which to access the field view
             * @param is index sequence (reused for row computation)
             * @param hessian matrix in which to store the Hessian
             * @param args... the indices
             */
            template <size_t... row, typename... Idx>
            KOKKOS_INLINE_FUNCTION constexpr void computeHessian(
                const std::index_sequence<row...>& is, matrix_type& hessian,
                const Idx... args) const {
                // The comma operator forces left-to-right evaluation order, which reduces
                // performance; therefore we apply a dummy operation to dummy values and discard the
                // result
                [[maybe_unused]] auto _ = (hessianRow<row>(is, hessian, args...) + ...);
            }

            /*!
             * Utility function for computing the Hessian. Computes the entries in a single
             * row of the matrix via fold expression.
             * @tparam row the row index
             * @tparam col... the column indices (in practice, the sequence 0...Dim - 1)
             * @tparam Idx... the indices at which to access the field view
             * @param hessian matrix in which to store the hessian
             * @param args... the indices
             * @return An unused dummy value (required to allow use of a more performant fold
             * expression)
             */
            template <size_t row, size_t... col, typename... Idx>
            KOKKOS_INLINE_FUNCTION constexpr int hessianRow(const std::index_sequence<col...>&,
                                                            matrix_type& hessian,
                                                            const Idx... args) const {
                hessian[row] = (hessianEntry<row, col>(args...) + ...);
                return 0;
            }

            /*!
             * Utility function for computing the Hessian. Computes a single entry
             * of the matrix
             * @tparam row the row index
             * @tparam col the column index
             * @tparam Idx... the indices at which to access the field view
             * @param args... the indices
             * @return The entry of the Hessian at the given row and column
             */
            template <size_t row, size_t col, typename... Idx>
            KOKKOS_INLINE_FUNCTION constexpr vector_type hessianEntry(const Idx... args) const {
                using index_type       = std::tuple_element_t<0, std::tuple<Idx...>>;
                index_type coords[dim] = {args...};
                if constexpr (row == col) {
                    auto&& center = apply(u_m, coords);

                    coords[row] += 1;
                    auto&& right = apply(u_m, coords);

                    coords[row] -= 2;
                    auto&& left = apply(u_m, coords);

                    // The diagonal elements correspond to second derivatives w.r.t. a single
                    // variable
                    return vectors_m[row] * (right - 2. * center + left)
                           / (hvector_m[row] * hvector_m[row]);
                } else {
                    coords[row] += 1;
                    coords[col] += 1;
                    auto&& uu = apply(u_m, coords);

                    coords[col] -= 2;
                    auto&& ud = apply(u_m, coords);

                    coords[row] -= 2;
                    auto&& dd = apply(u_m, coords);

                    coords[col] += 2;
                    auto&& du = apply(u_m, coords);

                    // The non-diagonal elements are mixed derivatives, whose finite difference form
                    // is slightly different from above
                    return vectors_m[col] * (uu - du - ud + dd)
                           / (4. * hvector_m[row] * hvector_m[col]);
                }
                // Silences incorrect nvcc warning: missing return statement at end of non-void
                // function
                return vector_type{};
            }
        };
    }  // namespace detail
}  // namespace ippl

#endif
