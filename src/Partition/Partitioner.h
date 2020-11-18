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
#ifndef IPPL_PARTITIONER_H
#define IPPL_PARTITIONER_H

#include "Index/NDIndex.h"

namespace ippl {
    namespace detail {

        template <unsigned Dim>
        class Partitioner
        {
        public:
            Partitioner() = default;
            ~Partitioner() = default;

            template <typename view_type>
            void split(const NDIndex<Dim>& domain,
                       view_type& view,
                       e_dim_tag* decomp,
                       int nSplits) const;
        };
    }
}

#include "Partition/Partitioner.hpp"

#endif