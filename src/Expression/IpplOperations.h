#ifndef IPPL_OPERATIONS_H
#define IPPL_OPERATIONS_H

namespace ippl {
    #define DefineBinaryOperation(fun, name, op)                            \
    template<typename E1, typename E2>                                      \
    struct fun : public Expression<fun<E1, E2>, sizeof(E1) + sizeof(E2)> {  \
        KOKKOS_FUNCTION                                                     \
        fun(const E1& u, const E2& v) : u_m(u), v_m(v) { }                  \
                                                                            \
        KOKKOS_INLINE_FUNCTION                                              \
        auto operator[](size_t i) const { return op(u_m[i], v_m[i]); }      \
                                                                            \
        template<typename ...Args>                                          \
        KOKKOS_INLINE_FUNCTION                                              \
        auto operator()(Args... args) const {                               \
            return op(u_m(args...), v_m(args...));                          \
        }                                                                   \
                                                                            \
    private:                                                                \
        const typename ExprType<E1>::value_type u_m;                        \
        const typename ExprType<E2>::value_type v_m;                        \
    };                                                                      \
                                                                            \
    template<typename E1, typename E2>                                      \
    KOKKOS_FUNCTION                                                         \
    fun<E1, E2> name(const E1& u, const E2& v) {                            \
        return fun<E1, E2>(u, v);                                           \
    }


    DefineBinaryOperation(Add,      operator+, std::plus<>())
    DefineBinaryOperation(Subtract, operator-, std::minus<>())
    DefineBinaryOperation(Multiply, operator*, std::multiplies<>())
    DefineBinaryOperation(Divide,   operator/, std::divides<>())


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

    private:
        const typename ExprType<E1>::value_type u_m;
        const typename ExprType<E2>::value_type v_m;
    };

    template<typename E1, typename E2>
    KOKKOS_INLINE_FUNCTION
    meta_cross<E1, E2> cross(const E1& u, const E2& v) {
        return meta_cross<E1, E2>(u, v);
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

    private:
        const typename ExprType<E1>::value_type u_m;
        const typename ExprType<E2>::value_type v_m;
    };

    template<typename E1, typename E2>
    KOKKOS_INLINE_FUNCTION
    typename E1::value_t dot(const E1& u, const E2& v) {
        return meta_dot<E1, E2>(u, v)();
    }
}

#endif