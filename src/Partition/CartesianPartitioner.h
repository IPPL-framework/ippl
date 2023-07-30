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
#ifndef IPPL_CARTESIAN_PARTITIONER_H
#define IPPL_CARTESIAN_PARTITIONER_H

#include <array>

#include "Communicate/Communicator.h"
#include "Index/NDIndex.h"

namespace ippl {

    namespace detail {

        template <unsigned Dim>
        class CartesianPartitioner {
        public:
            CartesianPartitioner()  = default;
            ~CartesianPartitioner() = default;

            mpi::Communicator partition(const mpi::Communicator& communicator,
                                        const NDIndex<Dim>& domain,
                                        std::array<bool, Dim>& decomp) const;
        };
    }  // namespace detail
}  // namespace ippl

#include "Partition/CartesianPartitioner.hpp"

#endif
