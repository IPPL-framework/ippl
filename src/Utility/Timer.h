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
#ifndef IPPL_TIMER_H
#define IPPL_TIMER_H

#include <chrono>

class Timer
{
public:
    using timer_type = std::chrono::time_point<std::chrono::high_resolution_clock>;
    using duration_type = std::chrono::duration<double>;

    Timer();

    void clear();               // Set all accumulated times to 0
    void start();               // Start timer
    void stop();                // Stop timer

    double elapsed();         // Report clock time accumulated in seconds

private:
    double elapsed_m;
    timer_type start_m, stop_m;
};

#endif
