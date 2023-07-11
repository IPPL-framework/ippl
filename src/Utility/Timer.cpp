//
// Class Timer
//   This class is used in IpplTimings.
//
// Copyright (c) 2019, Matthias Frey, Paul Scherrer Institut, Villigen PSI, Switzerland
//               2021, Matthias Frey, University of St Andrews, St Andrews, UK
// All rights reserved
//
// Implemented as part of the PhD thesis
// "Precise Simulations of Multibunches in High Intensity Cyclotrons"
//
// This file is part of OPAL.
//
// OPAL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with OPAL. If not, see <https://www.gnu.org/licenses/>.
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
