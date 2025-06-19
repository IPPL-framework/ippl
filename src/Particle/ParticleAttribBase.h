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

            virtual void pack(const hash_type&) = 0;

            virtual void unpack(size_type) = 0;

            virtual void serialize(Archive<memory_space>& ar, size_type nsends) = 0;

            virtual void deserialize(Archive<memory_space>& ar, size_type nrecvs) = 0;

            virtual size_type size() const = 0;

            virtual ~ParticleAttribBase() = default;

            void setParticleCount(size_type& num) { localNum_mp = &num; }
            size_type getParticleCount() const { return *localNum_mp; }

            virtual void applyPermutation(const hash_type&) = 0;
            virtual void internalCopy(const hash_type&) = 0;

        protected:
            const size_type* localNum_mp;
        };
    }  // namespace detail
}  // namespace ippl

#endif
