//
// Class ParticleBConds
//   This is a container for a set of particle boundary condition
//   functions. Boundary conditions for particles are not objects, but just
//   functions which map a position X -> X', given the minimum and maximum
//   values of the spatial domain.
//
// Copyright (c) 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
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
#ifndef IPPL_PARTICLE_BCONDS_H
#define IPPL_PARTICLE_BCONDS_H

#include "Region/NDRegion.h"
#include "Index/NDIndex.h"

#include <functional>
#include <array>


namespace ippl {
    namespace detail {
	template<typename T, unsigned Dim, typename ViewType>
	struct PeriodicBC  {
	    ViewType view_m;
	    double minval_m[Dim];
	    double maxval_m[Dim];

	    KOKKOS_INLINE_FUNCTION
	    PeriodicBC(const ViewType& view, const NDRegion<T, Dim>& nr)
		: view_m(view)
	    {
		for (unsigned i = 0; i < Dim; ++i) {
		    minval_m[i] = nr[i].min();
		    maxval_m[i] = nr[i].max();
		}
	    }

        KOKKOS_INLINE_FUNCTION
	void operator()(const size_t& i, const size_t& dim) const {
            if (view_m(i)[dim] < minval_m[dim])
                view_m(i)[dim] = (maxval_m[dim] - (minval_m[dim] - view_m(i)[dim]));
            else if (view_m(i)[dim] >= maxval_m[dim])
                view_m(i)[dim] = (minval_m[dim] + (view_m(i)[dim] - maxval_m[dim]));
        }

        KOKKOS_INLINE_FUNCTION
	    ~PeriodicBC() {}
	};


	template<typename T, unsigned Dim, typename ViewType>
	    struct ReflectiveBC {
            ViewType view_m;
		
            KOKKOS_INLINE_FUNCTION
		ReflectiveBC(const ViewType& view, const NDRegion<T, Dim>& nr)
		: view_m(view)
            {
		for (unsigned i = 0; i < Dim; ++i) {
                    minval_m[i] = nr[i].min();
                    maxval_m[i] = nr[i].max();
		}
            }

        KOKKOS_INLINE_FUNCTION
	    void operator()(const size_t& i, const size_t& dim) const {
            if (view_m(i)[dim] < minval_m[dim])
                view_m(i)[dim] = 2.0 * minval_m[dim] - view_m(i)[dim];
            else if (view_m(i)[dim] >= maxval_m[dim])
                view_m(i)[dim] = 2.0 * maxval_m[dim] - view_m(i)[dim];
        }

        KOKKOS_INLINE_FUNCTION
            ~ReflectiveBC() {}
        };


	template<typename T, unsigned Dim, typename ViewType>
	    struct SinkBC {
            ViewType view_m;
            double minval_m[Dim];
            double maxval_m[Dim];

            KOKKOS_INLINE_FUNCTION
		SinkBC(const ViewType& view, const NDRegion<T, Dim>& nr)
		: view_m(view)
            {
		for (unsigned i = 0; i < Dim; ++i) {
                    minval_m[i] = nr[i].min();
                    maxval_m[i] = nr[i].max();
		}
            }

        KOKKOS_INLINE_FUNCTION
	    void operator()(const size_t& i, const size_t& dim) const {
            if (view_m(i)[dim] < minval_m[dim])
                view_m(i)[dim] = minval_m[dim];
            else if (view_m(i)[dim] >= maxval_m[dim])
                view_m(i)[dim] = maxval_m[dim];
        }

        KOKKOS_INLINE_FUNCTION
            ~SinkBC() {}
        };


    //////////////////////////////////////////////////////////////////////
    // general container for a set of particle boundary conditions
    /*    template<typename T, unsigned Dim, typename ViewType>
    struct ParticleBConds {

	KOKKOS_FUNCTION
	ParticleBConds() {}

    */
	/*!
         * Initialize all BC's to null ones, which do not change
         * the value of the data any
         */
    /*	KOKKOS_FUNCTION
        ParticleBConds(const ViewType& view, const ParticleBC<T, Dim, ViewType> bcs[2*Dim], const NDRegion<T, Dim>& nr) : view_m(view) {
	    for (int d = (2 * Dim - 1); d >= 0; --d)
		bcs_m[d] = bcs[d]; //ParticleNoBCond<T>;
        }

	KOKKOS_FUNCTION
        ParticleBConds(const ParticleBConds<T, Dim, ViewType>&) = default;

    */
        /*!
         * Assignment operator
         */
	/*	KOKKOS_INLINE_FUNCTION
        ParticleBConds<T, Dim>& operator=(const ParticleBConds<T, Dim>& pbc) {
            for (int d = (2 * Dim - 1); d >= 0; --d)
                bcs_m[d] = pbc.bcs_m[d];
            return *this;
	    }*/

    //	KOKKOS_FUNCTION
    //    ~ParticleBConds() { } //= default;

        /*!
         *
         * @returns value of dth boundary condition
         */
    /*	KOKKOS_INLINE_FUNCTION
        ParticleBC<T, Dim, ViewType>& operator[](unsigned d) { return bcs_m[d]; }


	KOKKOS_INLINE_FUNCTION
	void operator()(const size_t& i, const size_t& j, const size_t& d) const {
	    bcs_m[d](i, j);
	}
    */
        /*!
         * for the given value in the given dimension over the given NDRegion,
         * apply the proper BC and return back the new value
         */
	/*	KOKKOS_INLINE_FUNCTION
        T apply(const T t, const unsigned d, const NDRegion<T,Dim>& nr) const {
            return apply(t, d, nr[d].min(), nr[d].max());
	    }*/

        /*!
         * for the given value in the given dimension over the given NDIndex,
         * apply the proper BC and return back the new value.  The extra +1
         * added to the max value is due to the assumption of a cell-centered
         * field.
         */
	/*	KOKKOS_INLINE_FUNCTION
        T apply(const T t, const unsigned d, const NDIndex<Dim>& ni) const {
            return apply(t, d, ni[d].first(), ni[d].last() + 1);
	    }*/

        /*!
         * a different version of apply, where the user just specifies the min
         * and max values of the given dimension
         */
	/*
	KOKKOS_INLINE_FUNCTION
        T apply(const T t, const unsigned d, const T m1, const T m2) const {
            if (t < m1)
                return (bcs_m[d+d])(t, m1, m2);
            else if (t >= m2)                           // here we take into account that
                return (bcs_m[d+d+1])(t, m1, m2);      // Region's store intervals [A,B)
            else
                return t;
		}*/

    //    private:
        //! array storing the function pointers
    /*  ParticleBC<T, Dim, ViewType> bcs_m[2 * Dim];
	ViewType view_m;
	};*/
}

#endif
