#ifndef IPPL_IOS_STANDARD_H
#define IPPL_IOS_STANDARD_H

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
