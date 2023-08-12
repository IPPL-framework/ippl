#ifndef IPPL_VTK_STREAM_H
#define IPPL_VTK_STREAM_H

#include "Stream/BasicFileStream.h"

namespace ippl {

    class VtkStream : public BasicFileStream<ParticleBase>, BasicFileStream<FieldContainer> {
    public:
        void operator<<(const ParticleBase& obj) override;

        void operator>>(ParticleBase& obj) override;

        void operator<<(const FieldContainer& obj) override;

        void operator>>(FieldContainer& obj) override;
    };
}  // namespace ippl

#endif
