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
        template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
        class ParticleAttribBase {
            template <class... Properties>
            struct WithMemSpace {
                using memory_space = typename Kokkos::View<char*, Properties...>::memory_space;
                using type         = ParticleAttribBase<memory_space>;
            };

        public:
            using hash_type       = ippl::detail::hash_type<MemorySpace>;
            using memory_space    = MemorySpace;
            using execution_space = typename memory_space::execution_space;

            template <typename... Properties>
            using with_properties = typename WithMemSpace<Properties...>::type;

            virtual void create(size_type) = 0;

            virtual void destroy(const hash_type&, const hash_type&, size_type) = 0;
            virtual size_type packedSize(const size_type) const                 = 0;

            virtual void pack(void*, const hash_type&) const = 0;

            virtual void unpack(void*, size_type) = 0;

            virtual void serialize(Archive<memory_space>& ar, size_type nsends) = 0;

            virtual void deserialize(Archive<memory_space>& ar, size_type nrecvs) = 0;

            virtual size_type size() const = 0;

            ParticleAttribBase(const std::string& name,
                               const std::string& long_name,
                               const std::string& unit) : name_m(name), long_name_m(long_name), unit_m(unit) {}

            virtual ~ParticleAttribBase() = default;

            void setParticleCount(size_type& num) { localNum_mp = &num; }
            size_type getParticleCount() const { return *localNum_mp; }


            std::string name() const { return name_m; }
            std::string long_name() const { return long_name_m; }
            std::string unit() const { return unit_m; }


        protected:
            const size_type* localNum_mp;
            const std::string name_m;
            const std::string long_name_m;
            const std::string unit_m;
        };
    }  // namespace detail
}  // namespace ippl

#endif
