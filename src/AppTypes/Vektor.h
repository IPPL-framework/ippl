#ifndef IPPL_VEKTOR_H
#define IPPL_VEKTOR_H

#include "VektorExpressions.h"

#include <initializer_list>

namespace ippl {
    template<typename T, unsigned Dim>
    class Vektor : public VektorExpr<T, Vektor<T, Dim>> {
    public:
        typedef T value_t;
        static constexpr unsigned dim = Dim;
    
        KOKKOS_FUNCTION
        Vektor() : Vektor(value_t(0)) { }

        KOKKOS_FUNCTION
        Vektor(const Vektor<T, Dim>&) = default;

        KOKKOS_FUNCTION
        Vektor(const T& val);

        KOKKOS_FUNCTION
        Vektor(const std::initializer_list<T>& list);

        KOKKOS_FUNCTION
        ~Vektor() { }
        

        // Get and Set Operations
        KOKKOS_INLINE_FUNCTION
        value_t& operator[](unsigned int i);

        KOKKOS_INLINE_FUNCTION
        value_t operator[](unsigned int i) const;

        KOKKOS_INLINE_FUNCTION
        value_t& operator()(unsigned int i);

        KOKKOS_INLINE_FUNCTION
        value_t operator()(unsigned int i) const;

        // Assignment Operators
        template<typename E>
        KOKKOS_INLINE_FUNCTION
        Vektor<T, Dim>& operator=(const VektorExpr<T, E>& rhs);

        template<typename E>
        KOKKOS_INLINE_FUNCTION
        Vektor<T, Dim>& operator+=(const VektorExpr<T, E>& rhs);

        template<typename E>
        KOKKOS_INLINE_FUNCTION
        Vektor<T, Dim>& operator-=(const VektorExpr<T, E>& rhs);

        template<typename E>
        KOKKOS_INLINE_FUNCTION
        Vektor<T, Dim>& operator*=(const VektorExpr<T, E>& rhs);

        template<typename E>
        KOKKOS_INLINE_FUNCTION
        Vektor<T, Dim>& operator/=(const VektorExpr<T, E>& rhs);

    private:
        T data_m[Dim];
    };
}

#include "Vektor.hpp"

#endif // IPPL_VEKTOR_H

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
