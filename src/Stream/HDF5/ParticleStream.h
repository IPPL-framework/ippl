#ifndef IPPL_HDF5_PARTICLE_STREAM_H
#define IPPL_HDF5_PARTICLE_STREAM_H

#include "Stream/HDF5/Stream.h"

namespace ippl {

    namespace hdf5 {

        template <class ParticleContainer>
        class ParticleStream : public Stream<ParticleContainer> {
        public:
            ParticleStream() = delete;

            ParticleStream(std::unique_ptr<ippl::Format> format);

            void operator<<(const ParticleContainer& obj) override;

            void operator>>(ParticleContainer& obj) override;
        };
    }  // namespace hdf5
}  // namespace ippl

#include "Stream/HDF5/ParticleStream.hpp"

#endif
