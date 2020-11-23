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
namespace ippl {
    namespace detail {
        template <typename T, unsigned Dim>
        HaloCells<T, Dim>::HaloCells()
        { }


        template <typename T, unsigned Dim>
        void HaloCells<T, Dim>::resize(const view_type& view, int nghost)
        {
            using Kokkos::subview;
            using Kokkos::ALL;
            using Kokkos::make_pair;

            lower(0) = subview(view, make_pair(0, nghost), ALL(), ALL());

            auto xext = view.extent(0);
            upper(0) = subview(view, make_pair(xext - nghost, xext), ALL(), ALL());

            if constexpr(Dim > 1) {
                auto yext = view.extent(1);
                lower(1) = subview(view, ALL(), make_pair(0, nghost), ALL());
                upper(1) = subview(view, ALL(), make_pair(yext - nghost, yext), ALL());
            }

            if constexpr(Dim > 2) {
                auto zext = view.extent(2);
                lower(2) = subview(view, ALL(), ALL(), make_pair(0, nghost));
                upper(2) = subview(view, ALL(), ALL(), make_pair(zext - nghost, zext));
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