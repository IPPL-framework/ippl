//
// Host-side mode parser for isolating backend launch-order issues.
//
#ifndef IPPL_LAUNCH_GUARD_H
#define IPPL_LAUNCH_GUARD_H

#include <cstdlib>
#include <cstring>
#include <iostream>

namespace ippl::detail {
    enum class LaunchGuardMode { Disabled, Fence, Launch, LaunchAndFence };

    inline const char* launchGuardModeName(LaunchGuardMode mode) {
        switch (mode) {
            case LaunchGuardMode::Disabled:
                return "disabled";
            case LaunchGuardMode::Fence:
                return "fence";
            case LaunchGuardMode::Launch:
                return "launch";
            case LaunchGuardMode::LaunchAndFence:
                return "launch+fence";
        }
        return "unknown";
    }

    inline LaunchGuardMode launchGuardMode(const char* envName) {
        const char* value = std::getenv(envName);
        LaunchGuardMode mode = LaunchGuardMode::LaunchAndFence;

        if (value != nullptr) {
            if (value[0] == '0' || value[0] == 'n' || value[0] == 'N'
                || std::strcmp(value, "off") == 0 || std::strcmp(value, "OFF") == 0
                || std::strcmp(value, "false") == 0 || std::strcmp(value, "FALSE") == 0) {
                mode = LaunchGuardMode::Disabled;
            } else if (std::strcmp(value, "fence") == 0 || std::strcmp(value, "FENCE") == 0) {
                mode = LaunchGuardMode::Fence;
            } else if (std::strcmp(value, "launch") == 0 || std::strcmp(value, "LAUNCH") == 0) {
                mode = LaunchGuardMode::Launch;
            }
        }

        return mode;
    }

    inline LaunchGuardMode launchGuardMode(const char* envName, const char* label) {
        const LaunchGuardMode mode = launchGuardMode(envName);

        static bool scatterPrinted    = false;
        static bool bareFieldPrinted  = false;
        static bool haloCellsPrinted  = false;
        static bool fallbackPrinted   = false;
        bool* printed                 = &fallbackPrinted;

        if (std::strcmp(envName, "IPPL_GH200_GUARD_SCATTER_POST_CIC") == 0) {
            printed = &scatterPrinted;
        } else if (std::strcmp(envName, "IPPL_GH200_GUARD_BAREFIELD_ACCUMULATE") == 0) {
            printed = &bareFieldPrinted;
        } else if (std::strcmp(envName, "IPPL_GH200_GUARD_HALOCELLS_ACCUMULATE") == 0) {
            printed = &haloCellsPrinted;
        }

        if (!*printed) {
            const char* value = std::getenv(envName);
            std::cout << "[IPPL launch guard] " << label << ": " << envName << "="
                      << (value != nullptr ? value : "<unset>")
                      << " -> " << launchGuardModeName(mode) << std::endl;
            *printed = true;
        }

        return mode;
    }
}  // namespace ippl::detail

#endif
