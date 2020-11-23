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
#ifndef IPPL_GUARD_CELLS_H
#define IPPL_GUARD_CELLS_H

#include "Index/NDIndex.h"
#include "Types/ViewTypes.h"
// #include "Communicate/Archive.h"
#include "FieldLayout/FieldLayout.h"

#include <array>


namespace ippl {
    namespace detail {

        template <typename T, unsigned Dim>
        class HaloCells
        {

        public:
            // check Kokkos::LayoutRight or Kokkos::LayoutLeft
            using lower_type = typename ViewType<T, Dim, Kokkos::LayoutStride>::view_type;
            using upper_type = typename ViewType<T, Dim, Kokkos::LayoutStride>::view_type;
            using view_type  = typename detail::ViewType<T, Dim>::view_type;
            using Layout_t   = FieldLayout<Dim>;

            HaloCells();

            void resize(const view_type&, int nghost);

            lower_type& lower(unsigned int dim);

            upper_type& upper(unsigned int dim);

            void fillHalo(const T& value);

            void exchangeHalo(view_type&, const Layout_t* layout, int nghost);


//             void pack(view_type&, const Kokkos::View<int*>&) const;
//
//             void unpack(view_type&);

        private:
            /*! lower halo cells (ordering x, y, z)
             * x --> lower y-z plane
             * y --> lower x-z plane
             * z --> lower x-y plane
             */
            std::array<lower_type, Dim> lowerHalo_m;

            /*! upper halo cells (ordering x, y, z)
             * x --> upper y-z plane
             * y --> upper x-z plane
             * z --> upper x-y plane
             */
            std::array<upper_type, Dim> upperHalo_m;
        };
    }
}

#include "Field/HaloCells.hpp"

#endif