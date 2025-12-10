//
// Class IpplTimings
//   IpplTimings - a simple singleton class which lets the user create and
//   timers that can be printed out at the end of the program.
//
//   General usage
//    1) create a timer:
//       IpplTimings::TimerRef val = IpplTimings::getTimer("timer name");
//    This will either create a new one, or return a ref to an existing one
//
//    2) start a timer:
//       IpplTimings::startTimer(val);
//    This will start the referenced timer running.  If it is already running,
//    it will not change anything.
//
//    3) stop a timer:
//       IpplTimings::stopTimer(val);
//    This will stop the timer, assuming it was running, and add in the
//    time to the accumulating time for that timer.
//
//    4) print out the results:
//       IpplTimings::print();
//
//    5) reset all timers (for warmup):
//       IpplTimings::resetAllTimers();
//    Clears all accumulated times and individual measurements.
//
//    6) export to CSV for violin plots:
//       IpplTimings::dumpToCSV("output.csv");
//    Exports all individual timing measurements for statistical analysis.
//
#ifndef IPPL_TIMINGS_H
#define IPPL_TIMINGS_H

#include <exception>
#include <limits>
#include <map>
#include <stack>
#include <string>
#include <vector>

#include "Utility/PAssert.h"
#include "Utility/Timer.h"
#include "Utility/my_auto_ptr.h"

// a simple class used to store timer values
class IpplTimerInfo {
public:
    // typedef for reference to a timer
    typedef unsigned int TimerRef;

    // constructor
    IpplTimerInfo()
        : name("")
        , wallTime(0.0)
        , indx(std::numeric_limits<TimerRef>::max())
        , measurements()
        , measurement_count(0) {
        clear();
    }

    // destructor
    ~IpplTimerInfo() {}

    // timer operations
    void start() {
        if (!running) {
            running = true;
            t.stop();
            t.clear();
            t.start();
        }
    }

    void stop() {
        if (running) {
            t.stop();
            running = false;
            double elapsed = t.elapsed();
            wallTime += elapsed;

            // NEW: Store individual measurement for CSV export
            measurements.push_back(elapsed);
            measurement_count++;
        }
    }

    void clear() {
        t.stop();
        t.clear();
        running = false;
    }

    // NEW: Clear all data including measurements (for warmup reset)
    void clearAll() {
        clear();
        wallTime = 0.0;
        measurements.clear();
        measurement_count = 0;
    }

    // the IPPL timer that this object manages
    Timer t;

    // the name of this timer
    std::string name;

    // the accumulated time
    double wallTime;

    // is the timer turned on right now?
    bool running;

    // an index value for this timer
    TimerRef indx;

    // NEW: Store individual measurements for CSV/violin plots
    std::vector<double> measurements;
    size_t measurement_count;
};

struct Timing {
    // typedef for reference to a timer
    typedef unsigned int TimerRef;

    // a typedef for the timer information object
    typedef IpplTimerInfo TimerInfo;

    // Default constructor
    Timing();

    // Destructor - clear out the existing timers
    ~Timing();

    // create a timer, or get one that already exists
    TimerRef getTimer(const char*);

    // start a timer
    void startTimer(TimerRef);

    // stop a timer, and accumulate it's values
    void stopTimer(TimerRef);

    // clear a timer, by turning it off and throwing away its time
    void clearTimer(TimerRef);

    // return a TimerInfo struct by asking for the name
    TimerInfo* infoTimer(const char* nm) { return TimerMap[std::string(nm)]; }

    // print the results to standard out
    void print();

    // print the results to a file
    void print(const std::string& fn, const std::map<std::string, unsigned int>& problemSize);

    // NEW: Reset all timers (for warmup purposes)
    void resetAllTimers();

    // NEW: Dump all measurements to CSV for violin plots
    // Format: timer_name,rank,measurement_id,duration
    void dumpToCSV(const std::string& filename);

    // NEW: Dump with custom delimiter and header options
    void dumpToCSV(const std::string& filename, const std::string& delimiter, bool includeHeader);

    // NEW: Get measurements for a specific timer (for programmatic access)
    const std::vector<double>& getMeasurements(TimerRef t) const;
    const std::vector<double>& getMeasurements(const std::string& name) const;

    // NEW: Get number of measurements for a timer
    size_t getMeasurementCount(TimerRef t) const;

    // NEW: Get all timer names
    std::vector<std::string> getTimerNames() const;

    // type of storage for list of TimerInfo
    typedef std::vector<my_auto_ptr<TimerInfo> > TimerList_t;
    typedef std::map<std::string, TimerInfo*> TimerMap_t;

private:
    // a list of timer info structs
    TimerList_t TimerList;

    // a map of timers, keyed by string
    TimerMap_t TimerMap;

    // NEW: Empty vector for returning when timer not found
    static const std::vector<double> emptyMeasurements;
};

class IpplTimings {
public:
    // typedef for reference to a timer
    typedef Timing::TimerRef TimerRef;

    // a typedef for the timer information object
    typedef Timing::TimerInfo TimerInfo;

    // create a timer, or get one that already exists
    static TimerRef getTimer(const char* nm) { return instance->getTimer(nm); }

    // start a timer
    static void startTimer(TimerRef t) { instance->startTimer(t); }

    // stop a timer, and accumulate it's values
    static void stopTimer(TimerRef t) { instance->stopTimer(t); }

    // clear a timer, by turning it off and throwing away its time
    static void clearTimer(TimerRef t) { instance->clearTimer(t); }

    // return a TimerInfo struct by asking for the name
    static TimerInfo* infoTimer(const char* nm) { return instance->infoTimer(nm); }

    // print the results to standard out
    static void print() { instance->print(); }

    // print the results to a file
    static void print(std::string fn, const std::map<std::string, unsigned int>& problemSize =
                                          std::map<std::string, unsigned int>()) {
        instance->print(fn, problemSize);
    }

    static void stash();
    static void pop();

    // NEW: Reset all timers (clears accumulated times and measurements)
    // Use this after warmup iterations to get clean timing data
    static void resetAllTimers() { instance->resetAllTimers(); }

    // NEW: Dump all timing measurements to CSV
    // Format: timer_name,rank,measurement_id,duration_seconds
    // Perfect for creating violin plots in Python/R
    static void dumpToCSV(const std::string& filename) {
        instance->dumpToCSV(filename);
    }

    // NEW: Dump with custom options
    static void dumpToCSV(const std::string& filename,
                         const std::string& delimiter,
                         bool includeHeader = true) {
        instance->dumpToCSV(filename, delimiter, includeHeader);
    }

    // NEW: Get measurements for programmatic access
    static const std::vector<double>& getMeasurements(TimerRef t) {
        return instance->getMeasurements(t);
    }
    static const std::vector<double>& getMeasurements(const std::string& name) {
        return instance->getMeasurements(name);
    }

    // NEW: Get measurement count
    static size_t getMeasurementCount(TimerRef t) {
        return instance->getMeasurementCount(t);
    }

    // NEW: Get all timer names
    static std::vector<std::string> getTimerNames() {
        return instance->getTimerNames();
    }

private:
    // type of storage for list of TimerInfo
    typedef Timing::TimerList_t TimerList_t;
    typedef Timing::TimerMap_t TimerMap_t;

    // Default constructor
    IpplTimings();

    // Destructor - clear out the existing timers
    ~IpplTimings();

    static Timing* instance;
    static std::stack<Timing*> stashedInstance;
};

#endif