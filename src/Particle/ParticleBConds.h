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


namespace ippl {
    //////////////////////////////////////////////////////////////////////
    // particle boundary condition functions ...

    // null BC; value is not changed
    template<typename T>
    inline T ParticleNoBCond(const T t, const T /* minval */, const T /* maxval */) {
        return t;
    }

    // periodic BC; values wrap around at endpoints of the interval
    template<typename T>
    inline T ParticlePeriodicBCond(const T t, const T minval, const T maxval) {
        if (t < minval)
            return (maxval - (minval - t));
        else if (t >= maxval)
            return (minval + (t - maxval));
        else
            return t;
    }

    // reflective BC; values bounce back from endpoints
    template<typename T>
    inline T ParticleReflectiveBCond(const T t, const T minval, const T maxval) {
        if (t < minval)
            return (minval + (minval - t));
        else if (t >= maxval)
            return (maxval - (t - maxval));
        else
            return t;
    }

    // sink BC; particles stick to the selected face
    template<typename T>
    inline T ParticleSinkBCond(const T t, const T minval, const T maxval) {
        if (t < minval)
            return minval;
        else if (t >= maxval)
            return maxval;
        else
            return t;
    }


    //////////////////////////////////////////////////////////////////////
    // general container for a set of particle boundary conditions
    template<typename T, unsigned Dim>
    class ParticleBConds {

    public:
        // typedef for a pointer to boundary condition function
        typedef T (*ParticleBCond)(const T, const T, const T);

    public:
        /*!
         * Initialize all BC's to null ones, which do not change
         * the value of the data any
         */
        ParticleBConds() {
            for (int d = (2 * Dim - 1); d >= 0; --d)
                BCList[d] = ParticleNoBCond;
        }


        /*!
         * Initialize all BC's to null ones, which do not change
         * the value of the data any
         */
        ParticleBConds(const std::initializer_list<ParticleBCond>& /*bcs*/) {
//             for (int d = (2 * Dim - 1); d >= 0; --d)
//                 BCList[d] = bcs[i];
        }


        /*!
         * Assignment operator
         */
        ParticleBConds<T, Dim>& operator=(const ParticleBConds<T, Dim>& pbc) {
            for (int d = (2 * Dim - 1); d >= 0; --d)
                BCList[d] = pbc.BCList[d];
            return *this;
        }

        /*!
         *
         * @returns value of dth boundary condition
         */
        ParticleBCond& operator[](unsigned d) { return BCList[d]; }

        /*!
         * for the given value in the given dimension over the given NDRegion,
         * apply the proper BC and return back the new value
         */
        T apply(const T t, const unsigned d, const NDRegion<T,Dim>& nr) const {
            return apply(t, d, nr[d].min(), nr[d].max());
        }

        /*!
         * for the given value in the given dimension over the given NDIndex,
         * apply the proper BC and return back the new value.  The extra +1
         * added to the max value is due to the assumption of a cell-centered
         * field.
         */
        T apply(const T t, const unsigned d, const NDIndex<Dim>& ni) const {
            return apply(t, d, ni[d].first(), ni[d].last() + 1);
        }

        /*!
         * a different version of apply, where the user just specifies the min
         * and max values of the given dimension
         */
        T apply(const T t, const unsigned d, const T m1, const T m2) const {
            if (t < m1)
                return (BCList[d+d])(t, m1, m2);
            else if (t >= m2)                           // here we take into account that
                return (BCList[d+d+1])(t, m1, m2);      // Region's store intervals [A,B)
            else
                return t;
        }

    private:
        //! array storing the function pointers
        ParticleBCond BCList[2 * Dim];
    };
}

#endif