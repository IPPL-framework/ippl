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
#include <cstdlib>
#include <cstring>
#include <list>
#include "Utility/IpplInfo.h"

#include <Kokkos_Core.hpp>

// public static members of IpplInfo, initialized to default values
std::unique_ptr<ippl::Communicate> Ippl::Comm = 0;
std::unique_ptr<Inform> Ippl::Info            = 0;
std::unique_ptr<Inform> Ippl::Warn            = 0;
std::unique_ptr<Inform> Ippl::Error           = 0;

void Ippl::deleteGlobals() {
    Info.reset();
    Warn.reset();
    Error.reset();
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

Ippl::Ippl(int& argc, char**& argv, MPI_Comm mpicomm) {
    Info  = std::make_unique<Inform>("Ippl");
    Warn  = std::make_unique<Inform>("Warning", std::cerr);
    Error = std::make_unique<Inform>("Error", std::cerr, INFORM_ALL_NODES);

    Comm = std::make_unique<ippl::Communicate>(argc, argv, mpicomm);

    try {
        std::list<std::string> notparsed;
        int infoLevel = 0;
        int nargs     = 0;
        while (nargs < argc) {
            if (checkOption(argv[nargs], "--help", "-h")) {
                if (Comm->myNode() == 0) {
                    IpplInfo::printHelp(argv);
                }
                std::exit(0);
            } else if (checkOption(argv[nargs], "--info", "-i")) {
                ++nargs;
                if (nargs >= argc) {
                    throw std::runtime_error("Missing info level value!");
                }
                infoLevel = getIntOption(argv[nargs]);
            } else if (checkOption(argv[nargs], "--version", "-v")) {
                IpplInfo::printVersion();
                std::string options = IpplInfo::compileOptions();
                std::string header("Compile-time options: ");
                while (options.length() > 58) {
                    std::string line = options.substr(0, 58);
                    size_t n         = line.find_last_of(' ');
                    INFOMSG(header << line.substr(0, n) << "\n");

                    header  = std::string(22, ' ');
                    options = options.substr(n + 1);
                }
                INFOMSG(header << options << endl);
                std::exit(0);
            } else if (nargs > 0 && std::strstr(argv[nargs], "--kokkos") == nullptr) {
                notparsed.push_back(argv[nargs]);
            }
            ++nargs;
        }

        Info->setOutputLevel(infoLevel);
        Error->setOutputLevel(0);
        Warn->setOutputLevel(0);

        if (infoLevel > 0 && Comm->myNode() == 0) {
            for (auto& l : notparsed) {
                std::cout << "Warning: Option '" << l << "' is not parsed by Ippl." << std::endl;
            }
        }

    } catch (const std::exception& e) {
        if (Comm->myNode() == 0) {
            std::cerr << e.what() << std::endl;
        }
        std::exit(0);
    }

    Kokkos::initialize(argc, argv);
}

bool Ippl::checkOption(const char* arg, const char* lstr, const char* sstr) {
    return (std::strcmp(arg, lstr) == 0) || (std::strcmp(arg, sstr) == 0);
}

int Ippl::getIntOption(const char* arg) {
    std::string sarg = arg;

    // 21. Dec. 2021
    // https://stackoverflow.com/questions/8888748/how-to-check-if-given-c-string-or-char-contains-only-digits
    if (!std::all_of(sarg.begin(), sarg.end(), ::isdigit)) {
        throw std::runtime_error("Missing integer command line argument!");
    }
    return std::atoi(arg);
}

/////////////////////////////////////////////////////////////////////
// Destructor: need to delete comm library if this is the last IpplInfo
Ippl::~Ippl() {
    Comm->deleteAllBuffers();
    Kokkos::finalize();
}

void Ippl::abort(const char* msg) {
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
