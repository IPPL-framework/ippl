//
// Class Timer
//   This class is used in IpplTimings.
//
//
//
#include "Kokkos_Core.hpp"

#include "Timer.h"

#include <chrono>

bool Timer::enableFences = IPPL_ENABLE_TIMER_FENCES;

namespace {
    using clock_type = std::chrono::high_resolution_clock;
    inline std::int64_t now_ticks() {
        return clock_type::now().time_since_epoch().count();
    }
}  // namespace

Timer::Timer() {
    this->clear();
}

void Timer::clear() {
    elapsed_m = 0.0;
}

void Timer::start() {
    // Fence on start as well: any in-flight kernel from before the timer was
    // armed otherwise leaks its tail latency into the measured interval.
    if (enableFences) {
        Kokkos::fence("Timer Fence (start)");
    }
    start_m = now_ticks();
}

void Timer::stop() {
    if (enableFences) {
        Kokkos::fence("Timer Fence (stop)");
    }
    stop_m = now_ticks();

    std::chrono::duration<double> elapsed =
        clock_type::duration(stop_m) - clock_type::duration(start_m);

    elapsed_m += elapsed.count();
}

double Timer::elapsed() {
    return elapsed_m;
}
