//
// Class GuardCells
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

// #include <iostream>

namespace ippl {
    namespace detail {

        template<unsigned Dim>
        class GuardCells
        {

        public:

            GuardCells() = delete;

            GuardCells(int nghost);

            constexpr GuardCells<Dim>(const GuardCells<Dim>&) = default;

            GuardCells<Dim>& operator=(const GuardCells<Dim>& /*gc*/)
            {
                return *this;
            }

        private:

        };
    }
}

// template<unsigned Dim>
// inline NDIndex<Dim>
// AddGuardCells(const NDIndex<Dim>& idx, const GuardCells<Dim>& g)
// {
//   NDIndex<Dim> ret;
//   for (unsigned int d=0; d<Dim; ++d)
//     ret[d] = ippl::Index(idx[d].min() - g.left(d), idx[d].max() + g.right(d));
//   return ret;
// }
//
//
// template<unsigned Dim>
// // Lexigraphic compare of two GuardCells so we can
// // use them as a Key in a map.
// inline bool
// GuardCells<Dim>::operator<(const GuardCells<Dim>& r) const
// {
//   for (unsigned d=0; d<Dim; ++d) {
//     if ( left(d) != r.left(d) )
//       return ( left(d) < r.left(d) );
//     if ( right(d) != r.right(d) )
//       return ( right(d) < r.right(d) );
//   }
//   // If we get here they're equal.
//   return false;
// }
//
// template<unsigned Dim>
// inline bool
// GuardCells<Dim>::operator==(const GuardCells<Dim>& r) const
// {
//   for (unsigned d=0; d<Dim; ++d) {
//     if ( left(d) != r.left(d) )
//       return false;
//     if ( right(d) != r.right(d) )
//       return false;
//   }
//   // If we get here they're equal.
//   return true;
// }

// template<unsigned Dim>
// std::ostream& operator<<(std::ostream&,const GuardCells<Dim>&);

#include "Field/GuardCells.hpp"

#endif