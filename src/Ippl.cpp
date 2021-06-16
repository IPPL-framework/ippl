//
// Class Ippl
//   Ippl environment.
//
// Copyright (c) 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//
#include "Ippl.h"
#include "Utility/IpplInfo.h"

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

// public static members of IpplInfo, initialized to default values
std::unique_ptr<ippl::Communicate>  Ippl::Comm = 0;
std::unique_ptr<Inform> Ippl::Info = 0;
std::unique_ptr<Inform> Ippl::Warn = 0;
std::unique_ptr<Inform> Ippl::Error = 0;
std::unique_ptr<Inform> Ippl::Debug = 0;

void Ippl::deleteGlobals() {
    Info.reset();
    Warn.reset();
    Error.reset();
    Debug.reset();
}

// print out current state to the given output stream
std::ostream& operator<<(std::ostream& o, const Ippl&) {
    o << "------------------------------------------\n";
    o << "IPPL Framework Application Summary:\n";
    o << "  Running on node " << Ippl::Comm->myNode();
    o << ", out of " << Ippl::Comm->getNodes() << " total.\n";
    o << "  Communication method: " << Ippl::Comm->name() << "\n";
    return o;
}


Ippl::Ippl(int& argc, char**& argv, MPI_Comm mpicomm)
: boost::mpi::environment(argc, argv)
{
    Kokkos::initialize(argc, argv);

    Info = std::make_unique<Inform>("Ippl");
    Warn = std::make_unique<Inform>("Warning", std::cerr);
    Error = std::make_unique<Inform>("Error", std::cerr, INFORM_ALL_NODES);
    Debug = std::make_unique<Inform>("**DEBUG**", std::cerr, INFORM_ALL_NODES);

    Comm = std::make_unique<ippl::Communicate>(mpicomm);

    try {
        namespace po = boost::program_options;

        /*
         * supported options
         */
        po::options_description desc("Ippl options");

        int infoLevel = 0;
        int warnLevel = 0;
        int errorLevel = 0;
        int debugLevel = 0;

        desc.add_options()
            ("ippl-help,h", "print Ippl help")
            ("ippl-version", "print Ippl version")
            ("summary", "print summary of Ippl library")

            ("info",
                po::value<int>(&infoLevel)->default_value(infoLevel),
                "set info message level")

            ("warn",
                po::value<int>(&warnLevel)->default_value(warnLevel),
                "set warning message level")

            ("error",
                po::value<int>(&errorLevel)->default_value(errorLevel),
                "set error message level")

            ("debug",
                po::value<int>(&debugLevel)->default_value(debugLevel),
                "set debug message level")
        ;

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("ippl-help")) {
            INFOMSG(desc << endl);
            std::exit(0);
        }

        if (vm.count("ippl-version")) {
            IpplInfo::printVersion();
            std::string options = IpplInfo::compileOptions();
            std::string header("Compile-time options: ");
            while (options.length() > 58) {
                std::string line = options.substr(0, 58);
                size_t n = line.find_last_of(' ');
                INFOMSG(header << line.substr(0, n) << "\n");

                header = std::string(22, ' ');
                options = options.substr(n + 1);
            }
            INFOMSG(header << options << endl);
            std::exit(0);
        }


        if (vm.count("summary")) {
            INFOMSG(*this << endl);
            std::exit(0);
        }

        if (vm.count("info")) {
            Info->setOutputLevel(infoLevel);
        }

        if (vm.count("warn")) {
            Warn->setOutputLevel(warnLevel);
        }

        if (vm.count("error")) {
            Error->setOutputLevel(errorLevel);
        }

        if (vm.count("debug")) {
            Debug->setOutputLevel(debugLevel);
        }

    } catch(const std::exception&) {
        /* do nothing here since a user might have passed Kokkos
         * arguments
         */
    }
}


/////////////////////////////////////////////////////////////////////
// Destructor: need to delete comm library if this is the last IpplInfo
Ippl::~Ippl() {
    Comm->deleteAllBuffers();
    Kokkos::finalize();
}


void Ippl::abort(const char *msg) {
    // print out message, if one was provided
    if (msg != 0) {
        ERRORMSG(msg << endl);
    }

    // that's it, folks this error will be propperly catched in the main
    throw std::runtime_error("Error form IpplInfo::abort");
}


void Ippl::fence() {
    Kokkos::fence();
}
