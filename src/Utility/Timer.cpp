//
// Class Timer
//   This class is used in IpplTimings.
//   https://www.boost.org/doc/libs/1_70_0/libs/timer/doc/cpu_timers.html
//
// Copyright (c) 2019, Matthias Frey, Paul Scherrer Institut, Villigen PSI, Switzerland
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
#include "Timer.h"

Timer::Timer() {
    this->clear();
}


void Timer::clear() {
    wall_m = user_m = sys_m = 0.0;
}


void Timer::start() {
    timer_m.start();
}


void Timer::stop() {
    timer_m.stop();
    
    boost::timer::cpu_times elapsed = timer_m.elapsed();
    
    wall_m += elapsed.wall;
    user_m += elapsed.user;
    sys_m  += elapsed.system;
}


double Timer::clock_time() {
    return wall_m * 1.0e-9;
}


double Timer::user_time() {
    return user_m * 1.0e-9;
}


double Timer::system_time() {
    return sys_m * 1.0e-9;
}


double Timer::cpu_time() {
    return (user_m + sys_m) * 1.0e-9;
}
