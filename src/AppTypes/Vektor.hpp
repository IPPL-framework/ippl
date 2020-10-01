#include "Utility/PAssert.h"

namespace ippl {
    /*
     *
     * Element access operators
     *
     */
    template<typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION
    typename Vektor<T, Dim>::value_t& Vektor<T, Dim>::operator[](unsigned int i) {
        PAssert(i < Dim);
        return data_m[i];
    }


    template<typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION
    typename Vektor<T, Dim>::value_t Vektor<T, Dim>::operator[](unsigned int i) const {
        PAssert(i < Dim);
        return data_m[i];
    }


    template<typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION
    typename Vektor<T, Dim>::value_t& Vektor<T, Dim>::operator()(unsigned int i) {
        PAssert(i < Dim);
        return data_m[i];
    }

    
    template<typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION
    typename Vektor<T, Dim>::value_t Vektor<T, Dim>::operator()(unsigned int i) const {
        PAssert(i < Dim);
        return data_m[i];
    }


    /*
     *
     * Vektor expression assignment operators
     *
     */
    template<typename T, unsigned Dim>
    template<typename E>
    KOKKOS_INLINE_FUNCTION
        Vektor<T, Dim>& Vektor<T, Dim>::operator=(const VektorExpr<T, E>& expr) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] = expr[i];
        }
        return *this;
    }


    template<typename T, unsigned Dim>
    template<typename E>
    KOKKOS_INLINE_FUNCTION
    Vektor<T, Dim>& Vektor<T, Dim>::operator+=(const VektorExpr<T, E>& expr) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] += expr[i];
        }
        return *this;
    }
    

    template<typename T, unsigned Dim>
    template<typename E>
    KOKKOS_INLINE_FUNCTION
    Vektor<T, Dim>& Vektor<T, Dim>::operator-=(const VektorExpr<T, E>& expr) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] -= expr[i];
        }
        return *this;
    }


    template<typename T, unsigned Dim>
    template<typename E>
    KOKKOS_INLINE_FUNCTION
    Vektor<T, Dim>& Vektor<T, Dim>::operator*=(const VektorExpr<T, E>& expr) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] *= expr[i];
        }
        return *this;
    }


    template<typename T, unsigned Dim>
    template<typename E>
    KOKKOS_INLINE_FUNCTION
    Vektor<T, Dim>& Vektor<T, Dim>::operator/=(const VektorExpr<T, E>& expr) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] /= expr[i];
        }
        return *this;
    }
}

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
