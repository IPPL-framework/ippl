//
// Class Timer
//   This class is used in IpplTimings.
//
//
//
#ifndef IPPL_TIMER_H
#define IPPL_TIMER_H

#ifndef IPPL_ENABLE_TIMER_FENCES
#warning "IPPL timer fences were not set via CMake! Defaulting to no fences."
#define IPPL_ENABLE_TIMER_FENCES false
#endif

#include <chrono>

class Timer {
public:
    using timer_type    = std::chrono::time_point<std::chrono::high_resolution_clock>;
    using duration_type = std::chrono::duration<double>;

    static bool enableFences;

    Timer();

    void clear();  // Set all accumulated times to 0
    void start();  // Start timer
    void stop();   // Stop timer

    double elapsed();  // Report clock time accumulated in seconds

private:
    double elapsed_m;
    timer_type start_m, stop_m;
};

#endif
