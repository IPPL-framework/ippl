#ifndef IPPL_VTK_STREAM_H
#define IPPL_VTK_STREAM_H

#include "Stream/BaseStream.h"

namespace ippl {

    class VtkStream : public BaseStream<ParticleBase>, BaseStream<FieldContainer> {
    public:
        void operator<<(const ParticleBase& obj) override;

        void operator>>(ParticleBase& obj) override;

        void operator<<(const FieldContainer& obj) override;

        void operator>>(FieldContainer& obj) override;
    };
}  // namespace ippl

#endif
