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
#include "Types/ViewType.h"

#include <array>


namespace ippl {
    namespace detail {

        template <typename T, unsigned Dim>
        class HaloCells
        {

        public:
            // check Kokkos::LayoutRight or Kokkos::LayoutLeft
            using lower_type  = typename ViewType::<T, Dim - 1, Kokkos::LayoutStride>::view_type;
            using upper_type = typename ViewType::<T, Dim - 1, Kokkos::LayoutStride>::view_type;

            HaloCells() = delete;

            HaloCells(int nghost);

        private:
            std::array<lower_type, Dim> lowerHalo_m;
            std::array<upper_type, Dim> upperHalo_m;
        };
    }
}

#include "Field/HaloCells.hpp"

#endif