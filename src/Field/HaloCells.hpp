//
// Class HaloCells
//   The guard / ghost cells of BareField.
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
#include "Field/HaloCells.h"

namespace ippl {
    namespace detail {
        template <typename T, unsigned Dim>
        HaloCells<T, Dim>::HaloCells()
        {
            static_assert(Dim >= 2, "Dimension must be greater than 1.");
        }


        template <typename T, unsigned Dim>
        HaloCells<T, Dim>::HaloCells(int nghost, const view_type& view)
        : HaloCells()
        {
            using Kokkos::subview;
            using Kokkos::ALL;

            lower[0] = subview(view, 0, ALL(), ALL());
            upper[0] = subview(view, view.extent(0), ALL(), ALL());

            if constexpr(Dim > 1) {
                lower[1] = subview(view, ALL(), 0, ALL());
                upper[1] = subview(view, ALL(), view.extent(1), ALL());
            }

            if constexpr(Dim > 2) {
                lower[2] = subview(view, ALL(), ALL(), 0);
                upper[2] = subview(view, ALL(), ALL(), view.extent(2));
            }
        }


        template <typename T, unsigned Dim>
        typename HaloCells<T, Dim>::lower_type&
        HaloCells<T, Dim>::lower(unsigned int dim) {
            return lowerHalo_m[dim];
        }

        template <typename T, unsigned Dim>
        typename HaloCells<T, Dim>::upper_type&
        HaloCells<T, Dim>::upper(unsigned int dim) {
            return upperHalo_m[dim];
        }
    }
}