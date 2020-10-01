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


    /*
     * Addition
     */

    template<typename T, typename E1, typename E2>
    struct VektorAdd : public VektorExpr<T, VektorAdd<T, E1, E2>> {
        KOKKOS_FUNCTION
        VektorAdd(const E1& u, const E2& v) : u_m(u), v_m(v) { }

        KOKKOS_INLINE_FUNCTION
        T operator[](size_t i) const { return u_m[i] + v_m[i]; }

    private:
        const E1 u_m;
        const E2 v_m;
    };

    template<typename T, typename E1, typename E2>
    KOKKOS_FUNCTION
    VektorAdd<T, E1, E2> operator+(const VektorExpr<T, E1>& u, const VektorExpr<T, E2>& v) {
        return VektorAdd<T, E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));
    }

    
    template<typename T, typename E>
    struct VektorAddScalarLeft : public VektorExpr<T, VektorAddScalarLeft<T, E>> {
        KOKKOS_FUNCTION
        VektorAddScalarLeft(const T& u, const E& v) : u_m(u), v_m(v) { }

        KOKKOS_INLINE_FUNCTION
        T operator[](size_t i) const { return u_m + v_m[i]; }

    private:
        const T u_m;
        const E v_m;
    };

    template<typename T, typename E>
    KOKKOS_FUNCTION
    VektorAddScalarLeft<T, E> operator+(const T& u, const VektorExpr<T, E>& v) {
        return VektorAddScalarLeft<T, E>(u, *static_cast<const E*>(&v));
    }


    template<typename T, typename E>
    struct VektorAddScalarRight : public VektorExpr<T, VektorAddScalarRight<T, E>> {
        KOKKOS_FUNCTION
        VektorAddScalarRight(const E& u, const T& v) : u_m(u), v_m(v) { }

        KOKKOS_INLINE_FUNCTION
        T operator[](size_t i) const { return u_m[i] + v_m; }

    private:
        const E u_m;
        const T v_m;
    };

    template<typename T, typename E>
    KOKKOS_FUNCTION
    VektorAddScalarRight<T, E> operator+(const VektorExpr<T, E>& u, const T& v) {
        return VektorAddScalarRight<T, E>(*static_cast<const E*>(&u), v);
    }

    /*
     * Subtraction
     */
        
    template<typename T, typename E1, typename E2>
    struct VektorSubtract : public VektorExpr<T, VektorSubtract<T, E1, E2>> {
        KOKKOS_FUNCTION
        VektorSubtract(const E1& u, const E2& v) : u_m(u), v_m(v) { }

        KOKKOS_INLINE_FUNCTION
        T operator[](size_t i) const { return u_m[i] - v_m[i]; }

    private:
        const E1 u_m;
        const E2 v_m;
    };

    template<typename T, typename E1, typename E2>
    KOKKOS_FUNCTION
    VektorSubtract<T, E1, E2> operator-(const VektorExpr<T, E1>& u, const VektorExpr<T, E2>& v) {
        return VektorSubtract<T, E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));
    }

    /*
     * Multiplication
     */

    template<typename T, typename E1, typename E2>
    struct VektorMultiply : public VektorExpr<T, VektorMultiply<T, E1, E2>> {
        KOKKOS_FUNCTION
        VektorMultiply(const E1& u, const E2& v) : u_m(u), v_m(v) { }

        KOKKOS_INLINE_FUNCTION
        T operator[](size_t i) const { return u_m[i] * v_m[i]; }

    private:
        const E1 u_m;
        const E2 v_m;
    };

    template<typename T, typename E1, typename E2>
    KOKKOS_FUNCTION
    VektorMultiply<T, E1, E2> operator*(const VektorExpr<T, E1>& u, const VektorExpr<T, E2>& v) {
        return VektorMultiply<T, E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));
    }

    /*
     * Division
     */

    template<typename T, typename E1, typename E2>
    struct VektorDivide : public VektorExpr<T, VektorDivide<T, E1, E2>> {
        KOKKOS_FUNCTION
        VektorDivide(const E1& u, const E2& v) : u_m(u), v_m(v) { }

        KOKKOS_INLINE_FUNCTION
        T operator[](size_t i) const { return u_m[i] / v_m[i]; }

    private:
        const E1 u_m;
        const E2 v_m;
    };

    template<typename T, typename E1, typename E2>
    KOKKOS_FUNCTION
    VektorDivide<T, E1, E2> operator/(const VektorExpr<T, E1>& u, const VektorExpr<T, E2>& v) {
        return VektorDivide<T, E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));
    }
}

#endif
