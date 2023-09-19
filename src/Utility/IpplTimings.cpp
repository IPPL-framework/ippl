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

#include "Ippl.h"

#include "Utility/IpplTimings.h"

#include <algorithm>
#include <fstream>
#include <iostream>

#include "Utility/Inform.h"
#include "Utility/IpplInfo.h"

Timing* IpplTimings::instance = new Timing();
std::stack<Timing*> IpplTimings::stashedInstance;

Timing::Timing()
    : TimerList()
    , TimerMap() {}

Timing::~Timing() {
    for (TimerMap_t::iterator it = TimerMap.begin(); it != TimerMap.end(); ++it) {
        it->second = 0;
    }
    TimerMap.clear();

    TimerList.clear();
}

// create a timer, or get one that already exists
Timing::TimerRef Timing::getTimer(const char* nm) {
    std::string s(nm);
    TimerInfo* tptr          = 0;
    TimerMap_t::iterator loc = TimerMap.find(s);
    if (loc == TimerMap.end()) {
        tptr       = new TimerInfo;
        tptr->indx = TimerList.size();
        tptr->name = s;
        TimerMap.insert(TimerMap_t::value_type(s, tptr));
        TimerList.push_back(my_auto_ptr<TimerInfo>(tptr));
    } else {
        tptr = (*loc).second;
    }
    return tptr->indx;
}

// start a timer
void Timing::startTimer(TimerRef t) {
    if (t >= TimerList.size())
        return;
    TimerList[t]->start();
}

// stop a timer, and accumulate it's values
void Timing::stopTimer(TimerRef t) {
    if (t >= TimerList.size())
        return;
    TimerList[t]->stop();
}

// clear a timer, by turning it off and throwing away its time
void Timing::clearTimer(TimerRef t) {
    if (t >= TimerList.size())
        return;
    TimerList[t]->clear();
}

// print out the timing results
void Timing::print() {
    if (TimerList.size() < 1)
        return;

    // report the average time for each timer
    Inform msg("Timings");
    msg << level1 << "---------------------------------------------";
    msg << "\n";
    msg << "     Timing results for " << ippl::Comm->size() << " nodes:"
        << "\n";
    msg << "---------------------------------------------";
    msg << "\n";

    {
        TimerInfo* tptr  = TimerList[0].get();
        double walltotal = 0.0;
        MPI_Reduce(&tptr->wallTime, &walltotal, 1, MPI_DOUBLE, MPI_MAX, 0,
                   ippl::Comm->getCommunicator());
        size_t lengthName = std::min(tptr->name.length(), 19lu);
        msg << tptr->name.substr(0, lengthName) << std::string().assign(20 - lengthName, '.')
            << " Wall tot = " << std::setw(10) << walltotal << "\n"
            << "\n";
    }

    for (unsigned int i = 1; i < TimerList.size(); ++i) {
        TimerInfo* tptr = TimerList[i].get();
        double wallmax = 0.0, wallmin = 0.0;
        double wallavg = 0.0;
        MPI_Reduce(&tptr->wallTime, &wallmax, 1, MPI_DOUBLE, MPI_MAX, 0,
                   ippl::Comm->getCommunicator());
        MPI_Reduce(&tptr->wallTime, &wallmin, 1, MPI_DOUBLE, MPI_MIN, 0,
                   ippl::Comm->getCommunicator());
        MPI_Reduce(&tptr->wallTime, &wallavg, 1, MPI_DOUBLE, MPI_SUM, 0,
                   ippl::Comm->getCommunicator());
        size_t lengthName = std::min(tptr->name.length(), 19lu);

        msg << tptr->name.substr(0, lengthName) << std::string().assign(20 - lengthName, '.')
            << " Wall max = " << std::setw(10) << wallmax << "\n"
            << std::string().assign(20, ' ') << " Wall avg = " << std::setw(10)
            << wallavg / ippl::Comm->size() << "\n"
            << std::string().assign(20, ' ') << " Wall min = " << std::setw(10) << wallmin << "\n"
            << "\n";
    }
    msg << "---------------------------------------------";
    msg << endl;
}

// save the timing results into a file
void Timing::print(const std::string& fn, const std::map<std::string, unsigned int>& problemSize) {
    std::ofstream* timer_stream;
    Inform* msg;

    if (TimerList.size() < 1)
        return;

    timer_stream = new std::ofstream;
    timer_stream->open(fn.c_str(), std::ios::out);
    msg = new Inform(0, *timer_stream, 0);

    if (problemSize.size() > 0) {
        *msg << "Problem size:\n";
        for (auto it : problemSize) {
            *msg << "    " << std::setw(10) << it.first << ": " << it.second << "\n";
        }
        *msg << endl;
    }

    *msg << std::setw(27) << "num Nodes" << std::setw(11) << "Wall tot\n"
         << std::string().assign(37, '=') << "\n";
    {
        TimerInfo* tptr  = TimerList[0].get();
        double walltotal = 0.0;
        MPI_Reduce(&tptr->wallTime, &walltotal, 1, MPI_DOUBLE, MPI_MAX, 0,
                   ippl::Comm->getCommunicator());
        size_t lengthName = std::min(tptr->name.length(), 19lu);
        *msg << tptr->name.substr(0, lengthName);
        for (int j = lengthName; j < 20; ++j) {
            *msg << ".";
        }
        *msg << " " << std::setw(6) << ippl::Comm->size() << " " << std::setw(9)
             << std::setprecision(4) << walltotal << "\n";
    }

    *msg << "\n"
         << std::setw(27) << "num Nodes" << std::setw(10) << "Wall max" << std::setw(10)
         << "Wall min" << std::setw(11) << "Wall avg\n"
         << std::string().assign(57, '=') << "\n";
    for (unsigned int i = 0; i < TimerList.size(); ++i) {
        TimerInfo* tptr = TimerList[i].get();
        double wallmax = 0.0, wallmin = 0.0;
        double wallavg = 0.0;
        MPI_Reduce(&tptr->wallTime, &wallmax, 1, MPI_DOUBLE, MPI_MAX, 0,
                   ippl::Comm->getCommunicator());
        MPI_Reduce(&tptr->wallTime, &wallmin, 1, MPI_DOUBLE, MPI_MIN, 0,
                   ippl::Comm->getCommunicator());
        MPI_Reduce(&tptr->wallTime, &wallavg, 1, MPI_DOUBLE, MPI_SUM, 0,
                   ippl::Comm->getCommunicator());
        size_t lengthName = std::min(tptr->name.length(), 19lu);
        *msg << tptr->name.substr(0, lengthName);
        for (int j = lengthName; j < 20; ++j) {
            *msg << ".";
        }
        *msg << " " << std::setw(6) << ippl::Comm->size() << " " << std::setw(9)
             << std::setprecision(4) << wallmax << " " << std::setw(9) << std::setprecision(4)
             << wallmin << " " << std::setw(9) << std::setprecision(4)
             << wallavg / ippl::Comm->size() << endl;
    }
    timer_stream->close();
    delete msg;
    delete timer_stream;
}

IpplTimings::IpplTimings() {}
IpplTimings::~IpplTimings() {}

void IpplTimings::stash() {
    PAssert_EQ(stashedInstance.size(), 0);

    stashedInstance.push(instance);
    instance = new Timing();
}

void IpplTimings::pop() {
    PAssert_GT(stashedInstance.size(), 0);

    delete instance;
    instance = stashedInstance.top();
    stashedInstance.pop();
}
