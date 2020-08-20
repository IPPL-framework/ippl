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
#ifndef TIMER_H
#define TIMER_H

#include <boost/timer/timer.hpp>

class Timer
{
public:
    
    Timer();
    
    void clear();               // Set all accumulated times to 0
    void start();               // Start timer
    void stop();                // Stop timer
    
    double clock_time();        // Report clock time accumulated in seconds
    double user_time();         // Report user time accumlated in seconds
    double system_time();       // Report system time accumulated in seconds
    double cpu_time();          // Report total cpu_time which is just user_time + system_time
    
private:
    double wall_m;
    double user_m;
    double sys_m;
    boost::timer::cpu_timer timer_m;
};

#endif
