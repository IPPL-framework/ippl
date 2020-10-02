#ifndef IPPL_VEKTOR_EXPRESSIONS_H
#define IPPL_VEKTOR_EXPRESSIONS_H

namespace ippl {
    template<typename T, typename E>
    struct VektorExpr {
        KOKKOS_INLINE_FUNCTION
        T operator[](size_t i) const {
            return static_cast<const E&>(*this)[i];
        }
    };

    #define DefineVektorVektorOperation(fun, op, expr)                  \
    template<typename T, typename E1, typename E2>                      \
    struct fun : public VektorExpr<T, fun<T, E1, E2>> {                 \
        KOKKOS_FUNCTION                                                 \
        fun(const E1& u, const E2& v) : u_m(u), v_m(v) { }              \
                                                                        \
        KOKKOS_INLINE_FUNCTION                                          \
        T operator[](size_t i) const { return expr; }                   \
                                                                        \
    private:                                                            \
        const E1 u_m;                                                   \
        const E2 v_m;                                                   \
    };                                                                  \
                                                                        \
    template<typename T, typename E1, typename E2>                      \
    KOKKOS_FUNCTION                                                     \
    fun<T, E1, E2> op(const VektorExpr<T, E1>& u,                       \
                             const VektorExpr<T, E2>& v) {              \
        return fun<T, E1, E2>(*static_cast<const E1*>(&u),              \
                              *static_cast<const E2*>(&v));             \
    }

    DefineVektorVektorOperation(VektorAdd,      operator+, u_m[i] + v_m[i])
    DefineVektorVektorOperation(VektorSubtract, operator-, u_m[i] - v_m[i])
    DefineVektorVektorOperation(VektorMultiply, operator*, u_m[i] * v_m[i])
    DefineVektorVektorOperation(VektorDivide,   operator/, u_m[i] / v_m[i])


    #define DefineScalarVektorOperation(fun, op, expr)                  \
    template<typename T, typename E>                                    \
    struct fun : public VektorExpr<T, fun<T, E>> {                      \
        KOKKOS_FUNCTION                                                 \
        fun(const T& u, const E& v) : u_m(u), v_m(v) { }                \
                                                                        \
        KOKKOS_INLINE_FUNCTION                                          \
        T operator[](size_t i) const { return expr; }                   \
                                                                        \
    private:                                                            \
        const T u_m;                                                    \
        const E v_m;                                                    \
    };                                                                  \
                                                                        \
    template<typename T, typename E>                                    \
    KOKKOS_FUNCTION                                                     \
    fun<T, E> op(const T& u,                                            \
                 const VektorExpr<T, E>& v) {                           \
        return fun<T, E>(u,                                             \
                         *static_cast<const E*>(&v));                   \
    }

    DefineScalarVektorOperation(VektorAddScalarLeft,      operator+, u_m + v_m[i])
    DefineScalarVektorOperation(VektorSubtractScalarLeft, operator-, u_m - v_m[i])
    DefineScalarVektorOperation(VektorDivideScalarLeft,   operator*, u_m * v_m[i])
    DefineScalarVektorOperation(VektorMultiplyScalarLeft, operator/, u_m / v_m[i])


    #define DefineVektorScalarOperation(fun, op, expr)                  \
    template<typename T, typename E>                                    \
    struct fun : public VektorExpr<T, fun<T, E>> {                      \
        KOKKOS_FUNCTION                                                 \
        fun(const E& u, const T& v) : u_m(u), v_m(v) { }                \
                                                                        \
        KOKKOS_INLINE_FUNCTION                                          \
        T operator[](size_t i) const { return expr; }                   \
                                                                        \
    private:                                                            \
        const E u_m;                                                    \
        const T v_m;                                                    \
    };                                                                  \
                                                                        \
    template<typename T, typename E>                                    \
    KOKKOS_FUNCTION                                                     \
    fun<T, E> op(const VektorExpr<T, E>& u,                             \
                 const T& v) {                                          \
        return fun<T, E>(*static_cast<const E*>(&u),                    \
                         v);                                            \
    }

    DefineVektorScalarOperation(VektorAddScalarRight,      operator+, u_m[i] + v_m)
    DefineVektorScalarOperation(VektorSubtractScalarRight, operator-, u_m[i] - v_m)
    DefineVektorScalarOperation(VektorMultiplyScalarRight, operator*, u_m[i] * v_m)
    DefineVektorScalarOperation(VektorDivideScalarRight,   operator/, u_m[i] / v_m)


    /*
     * Cross product. This function is only supported for 3-dimensional vectors.
     */

    template<typename T, typename E1, typename E2>
    struct meta_cross : public VektorExpr<T, meta_cross<T, E1, E2> > {
        KOKKOS_FUNCTION
        meta_cross(const E1& u, const E2& v) : u_m(u), v_m(v) {
//             static_assert(E1::dim == 3, "meta_cross: Dimension of first argument needs to be 3");
//             static_assert(E2::dim == 3, "meta_cross: Dimension of second argument needs to be 3");
        }

        KOKKOS_INLINE_FUNCTION
        T operator[](size_t i) const {
            const size_t j = (i + 1) % 3;
            const size_t k = (i + 2) % 3;
            return  u_m[j] * v_m[k] - u_m[k] * v_m[j];
        }

    private:
        const E1 u_m;
        const E2 v_m;
    };

    template<typename T, typename E1, typename E2>
    KOKKOS_INLINE_FUNCTION
    meta_cross<T, E1, E2> cross(const VektorExpr<T, E1> &u, const VektorExpr<T, E2> &v) {
        return meta_cross<T, E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));
    }



    /*
     * Dot product.
     */

    template<typename T, typename E1, typename E2>
    struct meta_dot : public VektorExpr<T, meta_dot<T, E1, E2> > {
        KOKKOS_FUNCTION
        meta_dot(const E1& u, const E2& v) : u_m(u), v_m(v) {
//             static_assert(E1::dim == E2::dim, "meta_dot: Dimensions do not agree!");
        }

        KOKKOS_INLINE_FUNCTION
        T operator()() const {
            T res = 0.0;
            for (size_t i = 0; i < E1::dim; ++i) {
                res += u_m[i] * v_m[i];
            }
            return res; //u_m[0] * v_m[0] + u_m[1] * v_m[1] + u_m[2] * v_m[2];
        }

    private:
        const E1 u_m;
        const E2 v_m;
    };

    template<typename T, typename E1, typename E2>
    KOKKOS_INLINE_FUNCTION
    T dot(const VektorExpr<T, E1>& u, const VektorExpr<T, E2>& v) {
        return meta_dot<T, E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v))();
    }
}

#endif
