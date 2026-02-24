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

#include <cstring>

#include "Types/IpplTypes.h"
#include "Types/ViewTypes.h"

#include "Communicate/Archive.h"

namespace ippl {
    namespace detail {
        // Maximum length for attribute names (including null terminator)
        constexpr size_t ATTRIB_NAME_MAX_LEN = 64;

        template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
        class ParticleAttribBase {
            template <class... Properties>
            struct WithMemSpace {
                using memory_space = typename Kokkos::View<char*, Properties...>::memory_space;
                using type         = ParticleAttribBase<memory_space>;
            };

        public:
            using memory_space    = MemorySpace;
            using execution_space = typename memory_space::execution_space;
            using hash_type       = detail::hash_type<MemorySpace>;
            using archive_type    = comms::archive_buffer<memory_space>;

            template <typename... Properties>
            using with_properties = typename WithMemSpace<Properties...>::type;

            KOKKOS_FUNCTION
            ParticleAttribBase() {
                const char* default_name = "UNNAMED_attribute";
                for (size_t i = 0; i < ATTRIB_NAME_MAX_LEN && default_name[i] != '\0'; ++i) {
                    name_m[i] = default_name[i];
                    if (i + 1 < ATTRIB_NAME_MAX_LEN) {
                        name_m[i + 1] = '\0';
                    }
                }
            }

            virtual void set_name(const std::string& name_) = 0;

            virtual std::string get_name() const = 0;

            virtual void create(size_type) = 0;

            virtual void destroy(const hash_type&, const hash_type&, size_type) = 0;
            virtual size_type packedSize(const size_type) const                 = 0;

            virtual void pack(const hash_type&) = 0;

            virtual void unpack(size_type) = 0;

            virtual void serialize(archive_type& ar, size_type nsends) = 0;

            virtual void deserialize(archive_type& ar, size_type nrecvs) = 0;

            virtual size_type size() const = 0;

            virtual ~ParticleAttribBase() = default;

            void setParticleCount(size_type& num) { localNum_mp = &num; }
            size_type getParticleCount() const { return *localNum_mp; }

            virtual void applyPermutation(const hash_type&) = 0;
            virtual void internalCopy(const hash_type&)     = 0;

        protected:
            const size_type* localNum_mp;
            char name_m[ATTRIB_NAME_MAX_LEN];
        };
    }  // namespace detail
}  // namespace ippl

#endif
