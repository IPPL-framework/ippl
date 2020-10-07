#ifndef IPPL_Vector_H
#define IPPL_Vector_H

#include "Expression/IpplExpressions.h"

#include <initializer_list>

namespace ippl {
    template<typename T, unsigned Dim>
    class Vector : public Expression<Vector<T, Dim>, sizeof(T) * Dim> {
    public:
        typedef T value_t;
        static constexpr unsigned dim = Dim;
    
        KOKKOS_FUNCTION
        Vector() : Vector(value_t(0)) { }


        template<typename E, size_t N>
        KOKKOS_FUNCTION
        Vector(const Expression<E, N>& expr);

        KOKKOS_FUNCTION
        Vector(const Vector<T, Dim>&) = default;

        KOKKOS_FUNCTION
        Vector(const T& val);

        KOKKOS_FUNCTION
        Vector(const std::initializer_list<T>& list);

        KOKKOS_FUNCTION
        ~Vector() { }
        

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
        template<typename E, size_t N>
        KOKKOS_INLINE_FUNCTION
        Vector<T, Dim>& operator=(const Expression<E, N>& expr);

        template<typename E, size_t N>
        KOKKOS_INLINE_FUNCTION
        Vector<T, Dim>& operator+=(const Expression<E, N>& expr);

        template<typename E, size_t N>
        KOKKOS_INLINE_FUNCTION
        Vector<T, Dim>& operator-=(const Expression<E, N>& expr);

        template<typename E, size_t N>
        KOKKOS_INLINE_FUNCTION
        Vector<T, Dim>& operator*=(const Expression<E, N>& expr);

        template<typename E, size_t N>
        KOKKOS_INLINE_FUNCTION
        Vector<T, Dim>& operator/=(const Expression<E, N>& expr);

    private:
        T data_m[Dim];
    };
}

#include "Vector.hpp"

#endif // IPPL_Vector_H

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
