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
#include <numeric>

namespace ippl {
    namespace detail {

        template <unsigned Dim>
        mpi::Communicator CartesianPartitioner<Dim>::partition(
            const mpi::Communicator& communicator, const NDIndex<Dim>& /*domain*/,
            std::array<bool, Dim>& /*decomp*/) const {
            //             using NDIndex_t = NDIndex<Dim>;

            return communicator;
        }
    }  // namespace detail
}  // namespace ippl
