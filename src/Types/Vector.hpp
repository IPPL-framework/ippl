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
//#include "Utility/PAssert.h"

#include <iomanip>
#include <iostream>

namespace ippl {
    namespace detail {
        template <typename T, unsigned Dim>
        struct isExpression<Vector<T, Dim>> : std::true_type {};
    }


    template<typename T, unsigned Dim>
    template<typename E, size_t N>
    KOKKOS_FUNCTION
    Vector<T, Dim>::Vector(const detail::Expression<E, N>& expr) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] = expr[i];
        }
    }

    /*
 	*
 	*
 	* Constructors with std::array and std::vector to assign
 	* vertex, face and edge neighbors in locateParticles
 	*
 	*/ 
    template<typename T, unsigned Dim>
    Vector<T, Dim>::Vector(const std::array<T, Dim>& a){
	unsigned int idx = 0;

        for(unsigned int i = 0; i< Dim; ++i){
	    if( a[i] < 0)
		continue;
            data_m[i]= a[i];
	    ++idx;
        }
	size_m = idx;
    }

    template<typename T, unsigned Dim>
    Vector<T, Dim>::Vector(const std::array<std::vector<T>, Dim>& a){

	unsigned int idx = 0;
	for(unsigned int i = 0; i < a.size(); i++){
		for(unsigned int j = 0; j < a[i].size(); j++){
			if( a[i].empty() )
				continue;
			data_m[idx] = a[i][j];
			++idx;	
		}
		size_m = idx;
	}
	}


    template<typename T, unsigned Dim>
    KOKKOS_FUNCTION
    Vector<T, Dim>::Vector(const T& val){ 
        for (unsigned int i = 0; i < Dim; ++i)
	  	data_m[i] = val;
    }


    template<typename T, unsigned Dim>
    KOKKOS_FUNCTION
    Vector<T, Dim>::Vector(const std::initializer_list<T>& list) {
        //PAssert(list.size() == Dim);
        unsigned int i = 0;
        for (auto& l : list) {
            data_m[i] = l;
            ++i;
        }
    }


    /*
     *
     * Element access operators
     *
     */
    template<typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION
    typename Vector<T, Dim>::value_type& Vector<T, Dim>::operator[](unsigned int i) {
        //PAssert(i < Dim);
        return data_m[i];
    }


    template<typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION
    typename Vector<T, Dim>::value_type Vector<T, Dim>::operator[](unsigned int i) const {
        //PAssert(i < Dim);
        return data_m[i];
    }


    template<typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION
    typename Vector<T, Dim>::value_type& Vector<T, Dim>::operator()(unsigned int i) {
        //PAssert(i < Dim);
        return data_m[i];
    }


    template<typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION
    typename Vector<T, Dim>::value_type Vector<T, Dim>::operator()(unsigned int i) const {
        //PAssert(i < Dim);
        return data_m[i];
    }

    template<typename T, unsigned Dim>
    KOKKOS_INLINE_FUNCTION
    size_t Vector<T, Dim>::size() const{
    return size_m;
    }


    /*
     *
     * Vector Expression assignment operators
     *
     */
    template<typename T, unsigned Dim>
    template<typename E, size_t N>
    KOKKOS_INLINE_FUNCTION
        Vector<T, Dim>& Vector<T, Dim>::operator=(const detail::Expression<E, N>& expr) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] = expr[i];
        }
        return *this;
    }


    template<typename T, unsigned Dim>
    template<typename E, size_t N>
    KOKKOS_INLINE_FUNCTION
    Vector<T, Dim>& Vector<T, Dim>::operator+=(const detail::Expression<E, N>& expr) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] += expr[i];
        }
        return *this;
    }


    template<typename T, unsigned Dim>
    template<typename E, size_t N>
    KOKKOS_INLINE_FUNCTION
    Vector<T, Dim>& Vector<T, Dim>::operator-=(const detail::Expression<E, N>& expr) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] -= expr[i];
        }
        return *this;
    }


    template<typename T, unsigned Dim>
    template<typename E, size_t N>
    KOKKOS_INLINE_FUNCTION
    Vector<T, Dim>& Vector<T, Dim>::operator*=(const detail::Expression<E, N>& expr) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] *= expr[i];
        }
        return *this;
    }


    template<typename T, unsigned Dim>
    template<typename E, size_t N>
    KOKKOS_INLINE_FUNCTION
    Vector<T, Dim>& Vector<T, Dim>::operator/=(const detail::Expression<E, N>& expr) {
        for (unsigned int i = 0; i < Dim; ++i) {
            data_m[i] /= expr[i];
        }
        return *this;
    }


    template<typename T, unsigned Dim>
    inline std::ostream& operator<<(std::ostream& out, const Vector<T, Dim>& v) {
    std::streamsize sw = out.width();
        out << std::setw(1);
        if (Dim >= 1) {
            out << "( ";
            for (unsigned int i = 0; i < Dim - 1; i++)
            out << std::setw(sw) << v[i] << " , ";
            out << std::setw(sw) << v[Dim - 1] << " )";
        } else {
            out << "( " << std::setw(sw) << v[0] << " )";
        }
        return out;
    }
}

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
