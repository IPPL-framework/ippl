#ifndef IPPL_HDF5_PARTICLE_STREAM_H
#define IPPL_HDF5_PARTICLE_STREAM_H

#include "Stream/Hdf5Stream.h"

namespace ippl {

    namespace hdf5 {

        template <class ParticleContainer>
        class ParticleStream : public Hdf5Stream<ParticleContainer> {
        public:

            ParticleStream() = default;

            void operator<<(const ParticleContainer& obj) override;

            void operator>>(ParticleContainer& obj) override;

        };
    }
}  // namespace ippl


#include "Stream/Hdf5ParticleStream.hpp"

#endif
