//
// Class CartesianPartitioner
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

namespace ippl {
    namespace detail {

        template <unsigned Dim>
        mpi::Communicator CartesianPartitioner<Dim>::partition(
            const mpi::Communicator& comm, const NDIndex<Dim>& /*domain*/,
            const std::array<bool, Dim>& isParallel) const {
            //             using NDIndex_t = NDIndex<Dim>;

            const int ndims = std::count(isParallel.cbegin(), isParallel.cend(), true);

            // ndims <= Dim
            int dims[Dim];

            MPI_Dims_create(comm.size(), ndims, dims);

            int periods[ndims];
            for (int i = 0; i < ndims; ++i) {
                periods[i] = 0;
            }

            MPI_Comm tmpcomm;
            MPI_Cart_create(comm, ndims, dims, periods, 0, &tmpcomm);
            mpi::Communicator cartcomm(tmpcomm);

            int coords[Dim];
            MPI_Cart_coords(cartcomm, cartcomm.rank(), ndims, coords);

            return cartcomm;
        }
    }  // namespace detail
}  // namespace ippl
