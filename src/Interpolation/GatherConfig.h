#ifndef IPPL_GATHER_CONFIG_H
#define IPPL_GATHER_CONFIG_H

#include "Interpolation/ScatterConfig.h"

namespace ippl {
namespace Interpolation {

    // GatherConfig is an alias for ScatterConfig since they share the same settings
    using GatherConfig = ScatterConfig;
    using GatherMethod = ScatterMethod;

}  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_GATHER_CONFIG_H
