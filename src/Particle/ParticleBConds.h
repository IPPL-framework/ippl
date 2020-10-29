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
        struct ParticleBC {
            ViewType view_m;
            double minval_m;
            double maxval_m;
            size_t dim_m;

            KOKKOS_INLINE_FUNCTION
            ParticleBC() = default;

            KOKKOS_INLINE_FUNCTION
            ParticleBC(const ViewType& view,
                       const NDRegion<T, Dim>& nr,
                       const unsigned& dim)
            : view_m(view)
            , minval_m(nr[dim].min())
            , maxval_m(nr[dim].max())
            , dim_m(dim)
            { }

            KOKKOS_INLINE_FUNCTION
            ~ParticleBC() = default;
        };


        template<typename T, unsigned Dim, typename ViewType>
        struct PeriodicBC : public ParticleBC<T, Dim, ViewType> {

            KOKKOS_INLINE_FUNCTION
            PeriodicBC() = default;

            KOKKOS_INLINE_FUNCTION
            PeriodicBC(const ViewType& view,
                       const NDRegion<T, Dim>& nr,
                       const unsigned& dim)
            : ParticleBC<T, Dim, ViewType>(view, nr, dim)
            { }

            KOKKOS_INLINE_FUNCTION
            void operator()(const size_t& i) const {
                if (this->view_m(i)[this->dim_m] < this->minval_m)
                    this->view_m(i)[this->dim_m] = (this->maxval_m - (this->minval_m - this->view_m(i)[this->dim_m]));
                else if (this->view_m(i)[this->dim_m] >= this->maxval_m)
                    this->view_m(i)[this->dim_m] = (this->minval_m + (this->view_m(i)[this->dim_m] - this->maxval_m));
            }

            KOKKOS_INLINE_FUNCTION
            ~PeriodicBC() = default;
        };


        template<typename T, unsigned Dim, typename ViewType>
        struct ReflectiveBC : public ParticleBC<T, Dim, ViewType> {

            KOKKOS_INLINE_FUNCTION
            ReflectiveBC() = default;

            KOKKOS_INLINE_FUNCTION
            ReflectiveBC(const ViewType& view,
                         const NDRegion<T, Dim>& nr,
                         const unsigned& dim)
            : ParticleBC<T, Dim, ViewType>(view, nr, dim)
            { }

            KOKKOS_INLINE_FUNCTION
            void operator()(const size_t& i) const {
                if (this->view_m(i)[this->dim_m] < this->minval_m)
                    this->view_m(i)[this->dim_m] = 2.0 * this->minval_m - this->view_m(i)[this->dim_m];
                else if (this->view_m(i)[this->dim_m] >= this->maxval_m)
                    this->view_m(i)[this->dim_m] = 2.0 * this->maxval_m - this->view_m(i)[this->dim_m];
            }

            KOKKOS_INLINE_FUNCTION
            ~ReflectiveBC() = default;
        };


        template<typename T, unsigned Dim, typename ViewType>
        struct SinkBC : public ParticleBC<T, Dim, ViewType> {

            KOKKOS_INLINE_FUNCTION
            SinkBC() = default;

            KOKKOS_INLINE_FUNCTION
            SinkBC(const ViewType& view,
                   const NDRegion<T, Dim>& nr,
                   const unsigned& dim)
            : ParticleBC<T, Dim, ViewType>(view, nr, dim)
            { }

            KOKKOS_INLINE_FUNCTION
            void operator()(const size_t& i) const {
                if (this->view_m(i)[this->dim_m] < this->minval_m)
                    this->view_m(i)[this->dim_m] = this->minval_m;
                else if (this->view_m(i)[this->dim_m] >= this->maxval_m)
                    this->view_m(i)[this->dim_m] = this->maxval_m;
            }

            KOKKOS_INLINE_FUNCTION
            ~SinkBC() = default;
        };
    }


//     template<typename T, unsigned Dim, typename ViewType>
//     class ParticleBConds {
//
//
//     private:
//         std::array<detail::ParticleBC<T, Dim, ViewType>*, 2 * Dim> bcs_m;
//     };
}

#endif
