//
// Class RegionLayout
//   RegionLayout stores a partitioned set of NDRegion objects, to represent
//   the parallel layout of an encompassing NDRegion.  It also contains
//   functions to find the subsets of the NDRegion partitions which intersect
//   or touch a given NDRegion.  It is similar to FieldLayout, with the
//   following changes:
//   1. It uses NDRegion instead of NDIndex, so it is templated on the position
//      data type (although it can be constructed with an NDIndex and a Mesh
//      as well);
//   2. It does not contain any consideration for guard cells;
//   3. It can store not only the partitioned domain, but periodic copies of
//      the partitioned domain for use by particle periodic boundary conditions
//   4. It also keeps a list of FieldLayoutUser's, so that it can notify them
//      when the internal FieldLayout here is reparitioned or otherwise changed.
//
//   If this is constructed with a FieldLayout, it stores a pointer to it
//   so that if we must repartition the copy of the FieldLayout that
//   is stored here, we will end up repartitioning all the registered Fields.
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
#ifndef IPPL_REGION_LAYOUT_H
#define IPPL_REGION_LAYOUT_H

#include "Region/PRegion.h"

#include "Types/ViewTypes.h"

namespace ippl {
    namespace detail {

        template <typename T, unsigned Dim, class Mesh/* = UniformCartesian<T, Dim> */>
        class RegionLayout
        {
        public:
             using container_type = typename ViewType<PRegion<T, Dim>::view_type;


            // Default constructor.  To make this class actually work, the user
            // will have to later call 'changeDomain' to set the proper Domain
            // and get a new partitioning.
//             RegionLayout()

            // Constructor which takes a FieldLayout and a MeshType
            // This one compares the domain of the FieldLayout and the domain of
            // the MeshType to determine the centering of the index space.
//             RegionLayout(FieldLayout<Dim>&, Mesh&);

//             // Destructor.
//             ~RegionLayout();

        private:
            container_type subdomains_m;
        };
    }
}

#include "Region/Kokkos_RegionLayout.hpp"

#endif