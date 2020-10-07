#ifndef IPPL_OPERATIONS_H
#define IPPL_OPERATIONS_H

namespace ippl {
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

        KOKKOS_INLINE_FUNCTION
        auto operator[](size_t i) const {
            const size_t j = (i + 1) % 3;
            const size_t k = (i + 2) % 3;
            return  u_m[j] * v_m[k] - u_m[k] * v_m[j];
        }

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

        KOKKOS_INLINE_FUNCTION
        typename E1::value_t operator()() const {
            typename E1::value_t res = 0.0;
            for (size_t i = 0; i < E1::dim; ++i) {
                res += u_m[i] * v_m[i];
            }
            return res; //u_m[0] * v_m[0] + u_m[1] * v_m[1] + u_m[2] * v_m[2];
        }

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
}

#endif