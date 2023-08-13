#ifndef IPPL_STREAM_FORMAT_H
#define IPPL_STREAM_FORMAT_H

#include "Utility/ParameterList.h"

namespace ippl {

    class Format {
    public:
        Format() = default;

        virtual ~Format() = default;

        virtual void header(ParameterList* param) const = 0;
    };
}  // namespace ippl

#endif
