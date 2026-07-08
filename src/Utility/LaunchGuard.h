//
// Small host-side launch guard helper for isolating backend launch-order issues.
//
#ifndef IPPL_LAUNCH_GUARD_H
#define IPPL_LAUNCH_GUARD_H

#include <cstdlib>
#include <cstring>

#include <Kokkos_Core.hpp>

namespace ippl::detail {
    enum class LaunchGuardMode { Disabled, Fence, Launch, LaunchAndFence };

    inline LaunchGuardMode launchGuardMode(const char* envName) {
        const char* value = std::getenv(envName);
        if (value == nullptr) {
            return LaunchGuardMode::LaunchAndFence;
        }

        if (value[0] == '0' || value[0] == 'n' || value[0] == 'N'
            || std::strcmp(value, "off") == 0 || std::strcmp(value, "OFF") == 0
            || std::strcmp(value, "false") == 0 || std::strcmp(value, "FALSE") == 0) {
            return LaunchGuardMode::Disabled;
        }
        if (std::strcmp(value, "fence") == 0 || std::strcmp(value, "FENCE") == 0) {
            return LaunchGuardMode::Fence;
        }
        if (std::strcmp(value, "launch") == 0 || std::strcmp(value, "LAUNCH") == 0) {
            return LaunchGuardMode::Launch;
        }

        return LaunchGuardMode::LaunchAndFence;
    }

    template <typename ExecutionSpace>
    inline void launchGuard(const char* envName, const char* label) {
        const LaunchGuardMode mode = launchGuardMode(envName);
        if (mode == LaunchGuardMode::Disabled) {
            return;
        }

        if (mode == LaunchGuardMode::Launch || mode == LaunchGuardMode::LaunchAndFence) {
            using guard_policy_type = Kokkos::RangePolicy<ExecutionSpace>;
            Kokkos::parallel_for(label, guard_policy_type(0, 1), KOKKOS_LAMBDA(const int) {});
        }
        if (mode == LaunchGuardMode::Fence || mode == LaunchGuardMode::LaunchAndFence) {
            Kokkos::fence();
        }
    }
}  // namespace ippl::detail

#endif
