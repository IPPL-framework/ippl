//
// Class Timer
//   This class is used in IpplTimings.
//
//
//
#include "Kokkos_Core.hpp"

#include "Timer.h"

bool Timer::enableFences = IPPL_ENABLE_TIMER_FENCES;

Timer::Timer() {
    this->clear();
}

void Timer::clear() {
    elapsed_m = 0.0;
}

void Timer::start() {
    start_m = std::chrono::high_resolution_clock::now();
}

void Timer::stop() {
    if (enableFences) {
        Kokkos::fence();
    }
    stop_m = std::chrono::high_resolution_clock::now();

    duration_type elapsed = stop_m - start_m;

    elapsed_m += elapsed.count();
}

double Timer::elapsed() {
    return elapsed_m;
}
