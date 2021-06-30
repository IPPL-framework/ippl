//
// Class ParticleAttribBase
//   Base class for all particle attribute classes.
//
//   This class is used as the generic base class for all (templated) classes
//   which represent a single attribute of a Particle.  An attribute class
//   contains a Kokkos::View of data for N particles, and methods to operate with
//   this data.
//
//   This base class provides virtual methods used to create and destroy
//   elements of the attribute array.
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

#ifndef IPPL_PARTICLE_ATTRIB_BASE_H
#define IPPL_PARTICLE_ATTRIB_BASE_H

#include "Types/ViewTypes.h"

#include "Communicate/Archive.h"

namespace ippl {
    namespace detail {
        template<class... Properties>
        class ParticleAttribBase {

        public:
            typedef typename ViewType<bool, 1, Properties...>::view_type boolean_view_type;

            virtual void create(size_t) = 0;

            virtual void destroy(boolean_view_type, Kokkos::View<int*>, size_t, size_t) = 0;

            virtual size_t packedSize(const int) const = 0;

            virtual void pack(void*, const Kokkos::View<int*>&) const = 0;

            virtual void unpack(void*, int) = 0;

            virtual void serialize(Archive<Properties...>& ar, int nsends) = 0;

            virtual void deserialize(Archive<Properties...>& ar, int nrecvs) = 0;

            virtual size_t size() const = 0;
            virtual size_t totalSize() const = 0;

            virtual ~ParticleAttribBase() = default;
        };
    }
}

#endif
