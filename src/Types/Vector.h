//
// Class Vector
//   Vector class used for vector fields and particle attributes like the coordinate.
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
#ifndef IPPL_Vector_H
#define IPPL_Vector_H

#include "Expression/IpplExpressions.h"

#include <initializer_list>

namespace ippl {
    /*!
     * @file Vector.h
     */

    /*!
     * @class Vector
     * @tparam T intrinsic vector data type
     * @tparam Dim vector dimension
     */
    template<typename T, unsigned Dim>
    class Vector : public detail::Expression<Vector<T, Dim>, sizeof(T) * Dim> {
    public:
        typedef T value_type;
        static constexpr unsigned dim = Dim;
    
        KOKKOS_FUNCTION
        Vector() : Vector(value_type(0)) { }


        template<typename E, size_t N>
        KOKKOS_FUNCTION
        Vector(const detail::Expression<E, N>& expr);

        KOKKOS_FUNCTION
	    Vector(const Vector<T, Dim>& v) { for(unsigned d = 0; d < Dim; ++d) { data_m[d] = v.data_m[d]; } }

        KOKKOS_FUNCTION
        Vector(const T& val);

        /*!
         * @param list of values
         */
        KOKKOS_FUNCTION
        Vector(const std::initializer_list<T>& list);

        KOKKOS_FUNCTION
        ~Vector() { }
        

        // Get and Set Operations
        KOKKOS_INLINE_FUNCTION
        value_type& operator[](unsigned int i);

        KOKKOS_INLINE_FUNCTION
        value_type operator[](unsigned int i) const;

        KOKKOS_INLINE_FUNCTION
        value_type& operator()(unsigned int i);

        KOKKOS_INLINE_FUNCTION
        value_type operator()(unsigned int i) const;

        // Assignment Operators
        template<typename E, size_t N>
        KOKKOS_INLINE_FUNCTION
        Vector<T, Dim>& operator=(const detail::Expression<E, N>& expr);

        template<typename E, size_t N>
        KOKKOS_INLINE_FUNCTION
        Vector<T, Dim>& operator+=(const detail::Expression<E, N>& expr);

        template<typename E, size_t N>
        KOKKOS_INLINE_FUNCTION
        Vector<T, Dim>& operator-=(const detail::Expression<E, N>& expr);

        template<typename E, size_t N>
        KOKKOS_INLINE_FUNCTION
        Vector<T, Dim>& operator*=(const detail::Expression<E, N>& expr);

        template<typename E, size_t N>
        KOKKOS_INLINE_FUNCTION
        Vector<T, Dim>& operator/=(const detail::Expression<E, N>& expr);

    private:
        T data_m[Dim];
    };
}

#include "Vector.hpp"

#endif
