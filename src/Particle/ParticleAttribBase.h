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

#include "Types/IpplTypes.h"
#include "Types/ViewTypes.h"

#include "Communicate/Archive.h"

namespace ippl {
    namespace detail {
        template <class... Properties>
        class ParticleAttribBase {
        public:
            typedef typename ViewType<bool, 1, Properties...>::view_type boolean_view_type;

            virtual void create(size_type) = 0;

            virtual void destroy(const Kokkos::View<int*>&, const Kokkos::View<int*>&,
                                 size_type)                     = 0;
            virtual size_type packedSize(const size_type) const = 0;

            virtual void pack(void*, const Kokkos::View<int*>&) const = 0;

            virtual void unpack(void*, size_type) = 0;

            virtual void serialize(Archive<Properties...>& ar, size_type nsends) = 0;

            virtual void deserialize(Archive<Properties...>& ar, size_type nrecvs) = 0;

            virtual size_type size() const = 0;

            virtual ~ParticleAttribBase() = default;

            void setParticleCount(size_type& num) { localNum_mp = &num; }
            size_type getParticleCount() const { return *localNum_mp; }

        protected:
            const size_type* localNum_mp;
        };
    }  // namespace detail
}  // namespace ippl

#endif
