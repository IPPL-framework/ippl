#ifndef IPPL_UTILITY_DEBUG_H
#define IPPL_UTILITY_DEBUG_H

#include <cstdlib>

namespace ippl {
    namespace detail {
        inline bool debugScatterHaloEnabled() {
            static const bool enabled = [] {
                const char* value = std::getenv("IPPL_DEBUG_SCATTER_HALO");
                return value != nullptr && value[0] != '\0'
                       && !(value[0] == '0' && value[1] == '\0');
            }();
            return enabled;
        }
    }  // namespace detail
}  // namespace ippl

#endif
