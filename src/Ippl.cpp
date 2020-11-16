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

/////////////////////////////////////////////////////////////////////
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

// private static members of IpplInfo, initialized to default values
MPI_Comm Ippl::communicator_m = MPI_COMM_WORLD;
bool Ippl::CommInitialized = false;
//
/////////////////////////////////////////////////////////////////////
// print out current state to the given output stream
std::ostream& operator<<(std::ostream& o, const Ippl&) {
    o << "------------------------------------------\n";
    o << "IPPL Framework Application Summary:\n";
    o << "  Running on node " << Ippl::Comm->myNode();
    o << ", out of " << Ippl::Comm->getNodes() << " total.\n";
    o << "  Communication method: " << Ippl::Comm->name() << "\n";
    return o;
}


/////////////////////////////////////////////////////////////////////
// Constructor 1: parse argc, argv, and create proper Communicate object
// The second argument controls whether the IPPL-specific command line
// arguments are stripped out (the default) or left in (if the setting
// is IpplInfo::KEEP).
Ippl::Ippl(int& argc, char**& argv, int removeargs, MPI_Comm mpicomm)
: boost::mpi::environment(argc, argv)
{
    Kokkos::initialize(argc, argv);

    int i;			// loop variables
    int retargc;			// number of args to return to caller
    char **retargv;		// arguments to return
    bool printsummary = false;	// print summary at end of constructor

    // determine whether we should strip out ippl-specific arguments, or keep
    bool stripargs = (removeargs != KEEP);

    communicator_m = mpicomm;

    Info = std::make_unique<Inform>("Ippl");
    Warn = std::make_unique<Inform>("Warning", std::cerr);
    Error = std::make_unique<Inform>("Error", std::cerr, INFORM_ALL_NODES);
    Debug = std::make_unique<Inform>("**DEBUG**", std::cerr, INFORM_ALL_NODES);

    // You can only specify argc, argv once; if it is done again, print a warning
    // and continue as if we had not given argc, argv.
    if ( !CommInitialized ) {
        // first make a pass through the arguments, figure out whether we should
        // run in parallel, and start up the parallel environment.  After this,
        // process all the other cmdline args
        std::string commtype;
        bool startcomm = false;
//         bool comminit = true;         // do comm. system's init call
        int nprocs = (-1);		// num of processes to start; -1 means default


        /*
          if no argument is given, we assume mpi as
          communication method
         */
        commtype = std::string("mpi");
        startcomm = true;

        // create Communicate object now.
        // dbgmsg << "Setting up parallel environment ..." << endl;
        if (startcomm && nprocs != 0 && nprocs != 1) {
            Comm = std::make_unique<ippl::Communicate>(communicator_m);

        }

        // keep track of which arguments we do no use; these are returned
        retargc = 1;
        retargv = new char*[argc];
        retargv[0] = argv[0];	// we always return arg 0 (the exec. name)

        // if we're not stripping out arguments, just save all the args
        if (!stripargs)
            for (i=1; i < argc; ++i)
                retargv[retargc++] = argv[i];


        // Parse command-line options, looking for ippl options.  When found,
        // save their suggested values and use them at the end to create data, etc.
        for (i=1; i < argc; ++i) {
            if ( ( strcmp(argv[i], "--summary") == 0 ) ) {
                // set flag to print out summary of Ippl library settings at the
                // end of this constructor
                printsummary = true;

            } else if ( ( strcmp(argv[i], "--ipplversion") == 0 ) ) {
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
                exit(0);

            } else if ( ( strcmp(argv[i], "--ipplversionall") == 0 ) ||
                        ( strcmp(argv[i], "-vall") == 0 ) ) {
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
                exit(0);

            } else if ( ( strcmp(argv[i], "--info") == 0 ) ) {
                // Set the output level for informative messages.
                if ( (i + 1) < argc && argv[i+1][0] != '-' && atoi(argv[i+1]) >= 0 )
                    Info->setOutputLevel(atoi(argv[++i]));
                else
                    param_error(argv[i],
                            "Please specify an output level from 0 to 5", 0);

            } else if ( ( strcmp(argv[i], "--warn") == 0 ) ) {
                // Set the output level for warning messages.
                if ( (i + 1) < argc && argv[i+1][0] != '-' && atoi(argv[i+1]) >= 0 )
                    Warn->setOutputLevel(atoi(argv[++i]));
                else
                    param_error(argv[i],
                            "Please specify an output level from 0 to 5", 0);

            } else if ( ( strcmp(argv[i], "--error") == 0 ) ) {
                // Set the output level for error messages.
                if ( (i + 1) < argc && argv[i+1][0] != '-' && atoi(argv[i+1]) >= 0 )
                    Error->setOutputLevel(atoi(argv[++i]));
                else
                    param_error(argv[i],
                            "Please specify an output level from 0 to 5", 0);

            } else if ( ( strcmp(argv[i], "--debug") == 0 ) ) {
                // Set the output level for debug messages.
                if ( (i + 1) < argc && argv[i+1][0] != '-' && atoi(argv[i+1]) >= 0 )
                    Debug->setOutputLevel(atoi(argv[++i]));
                else
                    param_error(argv[i],
                            "Please specify an output level from 0 to 5", 0);

            } else if ( ( strcmp(argv[i], "--ipplhelp") == 0 ) ||
                    ( strcmp(argv[i], "-h") == 0 ) ||
                    ( strcmp(argv[i], "-?") == 0 ) ) {
                // print out summary of command line switches and exit
                IpplInfo::printHelp(argv);
                INFOMSG("   --ipplversion       : Print a brief version summary.\n");
                INFOMSG("   --ipplversionall    : Print a detailed version summary.\n");
                INFOMSG("   --ipplhelp          : Display this command-line summary.\n");
                INFOMSG(endl);
                exit(0);

            } else {
                // Unknown option; just ignore it.
                if (stripargs)
                    retargv[retargc++] = argv[i];
            }
        }

        // We can get on with creating and initializing all globally-used objects.

        // indicate back to the caller which arguments are left
        if (stripargs) {
            argc = retargc;
            argv = retargv;
        }
    }

    // indicate we've created one more Ippl object
    CommInitialized = true;

    // At the very end, print out a summary if requested
    if (printsummary)
        INFOMSG(*this << endl);
}


/////////////////////////////////////////////////////////////////////
// Destructor: need to delete comm library if this is the last IpplInfo
Ippl::~Ippl() {
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


/////////////////////////////////////////////////////////////////////
// param_error: print out an error message when an illegal cmd-line
// parameter is encountered.
// Arguments are: parameter, error message, bad value (if any)
void Ippl::param_error(const char *param, const char *msg,
        const char *bad) {
    if ( param != 0 )
        ERRORMSG(param << " ");
    if ( bad != 0 )
        ERRORMSG(bad << " ");
    if ( msg != 0 )
        ERRORMSG(": " << msg);
    ERRORMSG(endl);
    Ippl::abort(0);
}

void Ippl::param_error(const char *param, const char *msg1,
        const char *msg2, const char *bad) {
    if ( param != 0 )
        ERRORMSG(param << " ");
    if ( bad != 0 )
        ERRORMSG(bad << " ");
    if ( msg1 != 0 )
        ERRORMSG(": " << msg1);
    if ( msg2 != 0 )
        ERRORMSG(msg2);
    ERRORMSG(endl);
    Ippl::abort(0);
}