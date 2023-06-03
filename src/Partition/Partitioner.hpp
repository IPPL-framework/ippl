//
// Class Partitioner
//   Partition a domain into subdomains.
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

#include <algorithm>
#include <numeric>
#include <vector>

namespace ippl {
    namespace detail {

        template <unsigned Dim>
        template <typename view_type>
        void Partitioner<Dim>::split(Communicate* communicate, const NDIndex<Dim>& domain,
                                     view_type& /*view*/, e_dim_tag* /*decomp*/,
                                     int nSplits) const {
            //             using NDIndex_t = NDIndex<Dim>;

            int dims[Dim];

            MPI_Dims_create(nSplits, Dim, dims);

            const int periods[Dim] = {false, false, false};
            int reorder            = false;

            MPI_Comm cart;

            MPI_Comm* world = communicate->getCommunicator();

            MPI_Cart_create(*world, Dim, dims, periods, reorder, &cart);

            int rank = -1;
            MPI_Comm_rank(cart, &rank);

            int coords[Dim];
            MPI_Cart_coords(cart, rank, Dim, coords);

            std::array<pair_type, Dim> bounds;

            for (unsigned d = 0; d < Dim; ++d) {
                int length = domain[d].length();
                bounds[d]  = this->getLocalBounds(length, coords[d], dims[d]);
            }
        }

        template <unsigned Dim>
        Partitioner<Dim>::pair_type Partitioner<Dim>::getLocalBounds(int nglobal, int coords,
                                                                     int dims) const {
            int nlocal    = nglobal / dims;
            int remaining = nglobal - dims * nlocal;

            int first = nlocal * coords;
            int last  = 0;
            if (coords < remaining) {
                nlocal = nlocal + 1;
                first  = first + coords;
                last   = last + coords;
            } else {
                first = first + remaining;
            }

            last = first + nlocal - 1;

            return std::make_pair(first, last);
        }
    }  // namespace detail
}  // namespace ippl
