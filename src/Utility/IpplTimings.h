// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef IPPL_TIMINGS_H
#define IPPL_TIMINGS_H

/*************************************************************************
 * IpplTimings - a simple singleton class which lets the user create and
 *   timers that can be printed out at the end of the program.
 *
 * General usage
 *  1) create a timer:
 *     IpplTimings::TimerRef val = IpplTimings::getTimer("timer name");
 *  This will either create a new one, or return a ref to an existing one
 *
 *  2) start a timer:
 *     IpplTimings::startTimer(val);
 *  This will start the referenced timer running.  If it is already running,
 *  it will not change anything.
 *
 *  3) stop a timer:
 *     IpplTimings::stopTimer(val);
 *  This will stop the timer, assuming it was running, and add in the
 *  time to the accumulating time for that timer.
 *
 *  4) print out the results:
 *     IpplTimings::print();
 *
 *************************************************************************/

// include files
#include "Utility/Timer.h"
#include "Utility/my_auto_ptr.h"

#include <string>
#include <vector>
#include <map>
#include <limits>
#include <stack>

#ifdef TIMERDEBUG
#include <exception>
#endif

// a simple class used to store timer values
class IpplTimerInfo
{
public:
    // typedef for reference to a timer
    typedef unsigned int TimerRef;

    // constructor
    IpplTimerInfo() : name(""), cpuTime(0.0), wallTime(0.0), indx(std::numeric_limits<TimerRef>::max()) {
        clear();
    }

    // destructor
    ~IpplTimerInfo() { }

    // timer operations
    void start() {
        if (!running) {
            running = true;
            t.stop();
            t.clear();
            t.start();
        }
#ifdef TIMERDEBUG
        else {
            throw std::runtime_error("Timer '" + name + "' already running");
        }
#endif
    }

    void stop() {
        if (running) {
            t.stop();
            running = false;
            cpuTime += t.cpu_time();
            wallTime += t.clock_time();
        }
#ifdef TIMERDEBUG
        else {
            throw std::runtime_error("Timer '" + name + "' already idling");
        }
#endif
    }

    void clear() {
        t.stop();
        t.clear();
        running = false;
    }

    // the IPPL timer that this object manages
    Timer t;

    // the name of this timer
    std::string name;

    // the accumulated time
    double cpuTime;
    double wallTime;

    // is the timer turned on right now?
    bool running;

    // an index value for this timer
    TimerRef indx;
};

struct Timing
{
    // typedef for reference to a timer
    typedef unsigned int TimerRef;

    // a typedef for the timer information object
    typedef IpplTimerInfo TimerInfo;

    // Default constructor
    Timing();

    // Destructor - clear out the existing timers
    ~Timing();

    // create a timer, or get one that already exists
    TimerRef getTimer(const char *);

    // start a timer
    void startTimer(TimerRef);

    // stop a timer, and accumulate it's values
    void stopTimer(TimerRef);

    // clear a timer, by turning it off and throwing away its time
    void clearTimer(TimerRef);

    // return a TimerInfo struct by asking for the name
    TimerInfo *infoTimer(const char *nm) {
        return TimerMap[std::string(nm)];
    }

    // print the results to standard out
    void print();

    // print the results to a file
    void print(const std::string &fn,
               const std::map<std::string, unsigned int> &problemSize);


    // type of storage for list of TimerInfo
    typedef std::vector<my_auto_ptr<TimerInfo> > TimerList_t;
    typedef std::map<std::string, TimerInfo *> TimerMap_t;

private:
    // a list of timer info structs
    TimerList_t TimerList;

    // a map of timers, keyed by string
    TimerMap_t TimerMap;
};



class IpplTimings
{
public:
    // typedef for reference to a timer
    typedef Timing::TimerRef TimerRef;

    // a typedef for the timer information object
    typedef Timing::TimerInfo TimerInfo;

    // create a timer, or get one that already exists
    static TimerRef getTimer(const char * nm) {
        return instance->getTimer(nm);
    }

    // start a timer
    static void startTimer(TimerRef t) {
        instance->startTimer(t);
    }

    // stop a timer, and accumulate it's values
    static void stopTimer(TimerRef t) {
        instance->stopTimer(t);
    }

    // clear a timer, by turning it off and throwing away its time
    static void clearTimer(TimerRef t) {
        instance->clearTimer(t);
    }

    // return a TimerInfo struct by asking for the name
    static TimerInfo *infoTimer(const char *nm) {
        return instance->infoTimer(nm);
    }

    // print the results to standard out
    static void print() {
        instance->print();
    }

    // print the results to a file
    static void print(std::string fn,
                      const std::map<std::string, unsigned int> &problemSize = std::map<std::string, unsigned int>()) {
        instance->print(fn, problemSize);
    }

    static void stash();
    static void pop();

private:
    // type of storage for list of TimerInfo
    typedef Timing::TimerList_t TimerList_t;
    typedef Timing::TimerMap_t TimerMap_t;

    // Default constructor
    IpplTimings();

    // Destructor - clear out the existing timers
    ~IpplTimings();


    static Timing *instance;
    static std::stack<Timing*> stashedInstance;
};

#endif

/***************************************************************************
 * $RCSfile: IpplTimings.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 ***************************************************************************/

/***************************************************************************
 * $RCSfile: addheaderfooter,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:17 $
 * IPPL_VERSION_ID: $Id: addheaderfooter,v 1.1.1.1 2003/01/23 07:40:17 adelmann Exp $
 ***************************************************************************/