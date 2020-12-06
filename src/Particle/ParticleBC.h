//
// Functor ParticleBC
//   Functors specifying particle boundary conditions.
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
#ifndef IPPL_PARTICLE_BC_H
#define IPPL_PARTICLE_BC_H

#include "Region/NDRegion.h"

namespace ippl {
    enum BC {
        PERIODIC,
        REFLECTIVE,
        SINK,
        NO
    };

    namespace detail {

        template<typename T, unsigned Dim, class ViewType>
        struct ParticleBC {

            using value_type = typename ViewType::value_type::value_type;

            ViewType view_m;
            double minval_m;
            double maxval_m;
            size_t dim_m;
            bool isUpper_m;

            KOKKOS_INLINE_FUNCTION
            ParticleBC() = default;

            KOKKOS_INLINE_FUNCTION
            ParticleBC(const ViewType& view,
                       const NDRegion<T, Dim>& nr,
                       const unsigned& dim, 
                       const bool& isUpper)
            : view_m(view)
            , minval_m(nr[dim].min())
            , maxval_m(nr[dim].max())
            , dim_m(dim)
            , isUpper_m(isUpper)
            { }

            KOKKOS_INLINE_FUNCTION
            ~ParticleBC() = default;
        };


        template<typename T, unsigned Dim, class ViewType>
        struct PeriodicBC : public ParticleBC<T, Dim, ViewType> {

            using value_type = typename ParticleBC<T, Dim, ViewType>::value_type;

            KOKKOS_INLINE_FUNCTION
            PeriodicBC() = default;

            KOKKOS_INLINE_FUNCTION
            PeriodicBC(const ViewType& view,
                       const NDRegion<T, Dim>& nr,
                       const unsigned& dim, 
                       const bool& isUpper)
            : ParticleBC<T, Dim, ViewType>(view, nr, dim, isUpper)
            { }

            KOKKOS_INLINE_FUNCTION
            void operator()(const size_t& i) const {
                value_type& value = this->view_m(i)[this->dim_m];

                if ((value < this->minval_m) && (!this->isUpper_m))
                    value = (this->maxval_m - (this->minval_m - value));
                else if ((value >= this->maxval_m) && (this->isUpper_m))
                    value = (this->minval_m + (value - this->maxval_m));
            }

            KOKKOS_INLINE_FUNCTION
            ~PeriodicBC() = default;
        };


        template<typename T, unsigned Dim, class ViewType>
        struct ReflectiveBC : public ParticleBC<T, Dim, ViewType> {

            using value_type = typename ParticleBC<T, Dim, ViewType>::value_type;

            KOKKOS_INLINE_FUNCTION
            ReflectiveBC() = default;

            KOKKOS_INLINE_FUNCTION
            ReflectiveBC(const ViewType& view,
                         const NDRegion<T, Dim>& nr,
                         const unsigned& dim, 
                         const bool& isUpper)
            : ParticleBC<T, Dim, ViewType>(view, nr, dim, isUpper)
            { }

            KOKKOS_INLINE_FUNCTION
            void operator()(const size_t& i) const {
                value_type& value = this->view_m(i)[this->dim_m];
                if ((value < this->minval_m) && (!this->isUpper_m))
                    value = 2.0 * this->minval_m - value;
                else if ((value >= this->maxval_m) && (this->isUpper_m))
                    value = 2.0 * this->maxval_m - value;
            }

            KOKKOS_INLINE_FUNCTION
            ~ReflectiveBC() = default;
        };


        template<typename T, unsigned Dim, class ViewType>
        struct SinkBC : public ParticleBC<T, Dim, ViewType> {

            using value_type = typename ParticleBC<T, Dim, ViewType>::value_type;

            KOKKOS_INLINE_FUNCTION
            SinkBC() = default;

            KOKKOS_INLINE_FUNCTION
            SinkBC(const ViewType& view,
                   const NDRegion<T, Dim>& nr,
                   const unsigned& dim, 
                   const bool& isUpper)
            : ParticleBC<T, Dim, ViewType>(view, nr, dim, isUpper)
            { }

            KOKKOS_INLINE_FUNCTION
            void operator()(const size_t& i) const {
                value_type& value = this->view_m(i)[this->dim_m];
                if ((value < this->minval_m) && (!this->isUpper_m))
                    value = this->minval_m;
                else if ((value >= this->maxval_m) && (this->isUpper_m))
                    value = this->maxval_m;
            }

            KOKKOS_INLINE_FUNCTION
            ~SinkBC() = default;
        };
    }
}

#endif
