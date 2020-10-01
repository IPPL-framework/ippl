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


    
    DefineScalarVektorOperation(VektorAddScalarLeft, operator+, u_m + v_m[i])


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

    DefineVektorScalarOperation(VektorAddScalarRight, operator+, u_m[i] + v_m)
}

#endif
