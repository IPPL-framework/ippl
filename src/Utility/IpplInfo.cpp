// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 * This program was prepared by PSI.
 * All rights in the program are reserved by PSI.
 * Neither PSI nor the author(s)
 * makes any warranty, express or implied, or assumes any liability or
 * responsibility for the use of this software
 *
 * Visit www.amas.web.psi for more details
 *
 ***************************************************************************/

// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

// include files
#include "Utility/IpplInfo.h"
#include "Utility/IpplStats.h"
#include "Utility/PAssert.h"
#include "Utility/vmap.h"
#include "Message/Message.h"
#include "Message/CommCreator.h"
#include "Message/Communicate.h"

#include "IpplVersions.h"

#include <unistd.h>
#include <cstdio>
#include <csignal>

/////////////////////////////////////////////////////////////////////
// public static members of IpplInfo, initialized to default values
std::unique_ptr<Communicate>  IpplInfo::Comm = 0;
std::unique_ptr<IpplStats> IpplInfo::Stats = 0;
std::unique_ptr<Inform> IpplInfo::Info = 0;
std::unique_ptr<Inform> IpplInfo::Warn = 0;
std::unique_ptr<Inform> IpplInfo::Error = 0;
std::unique_ptr<Inform> IpplInfo::Debug = 0;

void IpplInfo::instantiateGlobals() {
    if (Comm == 0)
        Comm = std::make_unique<Communicate>();
    if (Stats == 0)
        Stats = std::make_unique<IpplStats>();
    if (Info == 0)
        Info = std::make_unique<Inform>("Ippl");
    if (Warn == 0)
        Warn = std::make_unique<Inform>("Warning", std::cerr);
    if (Error == 0)
        Error = std::make_unique<Inform>("Error", std::cerr, INFORM_ALL_NODES);
    if (Debug == 0)
        Debug = std::make_unique<Inform>("**DEBUG**", std::cerr, INFORM_ALL_NODES);
}

void IpplInfo::deleteGlobals() {
    Comm.reset();
    Stats.reset();
    Info.reset();
    Warn.reset();
    Error.reset();
    Debug.reset();
}

// private static members of IpplInfo, initialized to default values
MPI_Comm IpplInfo::communicator_m = MPI_COMM_WORLD;
int  IpplInfo::NumCreated = 0;
bool IpplInfo::CommInitialized = false;
bool IpplInfo::PrintStats = false;
int  IpplInfo::MyArgc = 0;
char **IpplInfo::MyArgv = 0;
int  IpplInfo::MyNode = 0;
int  IpplInfo::TotalNodes = 1;
int  IpplInfo::MaxFFTNodes = 0;

/////////////////////////////////////////////////////////////////////
// print out current state to the given output stream
std::ostream& operator<<(std::ostream& o, const IpplInfo&) {
    o << "------------------------------------------\n";
    o << "IPPL Framework Application Summary:\n";
    o << "  Running on node " << IpplInfo::myNode();
    o << ", out of " << IpplInfo::getNodes() << " total.\n";
    o << "  Communication method: " << IpplInfo::Comm->name() << "\n";
    o << "  Elapsed wall-clock time (in seconds): ";
    o << IpplInfo::Stats->getTime().clock_time() << "\n";
    o << "  Elapsed CPU-clock time (in seconds) : ";
    o << IpplInfo::Stats->getTime().cpu_time() << "\n";
    o << "------------------------------------------\n";
    return o;
}


/////////////////////////////////////////////////////////////////////
// Constructor 1: parse argc, argv, and create proper Communicate object
// The second argument controls whether the IPPL-specific command line
// arguments are stripped out (the default) or left in (if the setting
// is IpplInfo::KEEP).
IpplInfo::IpplInfo(int& argc, char**& argv, int removeargs, MPI_Comm mpicomm) {

    int i;			// loop variables
    int retargc;			// number of args to return to caller
    char **retargv;		// arguments to return
    bool printsummary = false;	// print summary at end of constructor

    //Inform dbgmsg("IpplInfo(argc,argv)", INFORM_ALL_NODES);

    // determine whether we should strip out ippl-specific arguments, or keep
    bool stripargs = (removeargs != KEEP);

    communicator_m = mpicomm;

    if (NumCreated == 0) {
        Comm = std::make_unique<Communicate>();
        Stats = std::make_unique<IpplStats>();
        Info = std::make_unique<Inform>("Ippl");
        Warn = std::make_unique<Inform>("Warning", std::cerr);
        Error = std::make_unique<Inform>("Error", std::cerr, INFORM_ALL_NODES);
        Debug = std::make_unique<Inform>("**DEBUG**", std::cerr, INFORM_ALL_NODES);
    }
    // You can only specify argc, argv once; if it is done again, print a warning
    // and continue as if we had not given argc, argv.
    if ( CommInitialized ) {
      // ADA WARNMSG("Attempt to create IpplInfo with argc, argv again." << endl);
      //WARNMSG("Using previous argc,argv settings." << endl);
    } else {
        // dbgmsg << "Starting initialization: argc = " << argc << ", " << endl;
        // for (unsigned int dbgi=0; dbgi < argc; ++dbgi)
        //   dbgmsg << "  argv[" << dbgi << "] = '" << argv[dbgi] << "'" << endl;

        // first make a pass through the arguments, figure out whether we should
        // run in parallel, and start up the parallel environment.  After this,
        // process all the other cmdline args
        std::string commtype;
        bool startcomm = false;
        bool comminit = true;         // do comm. system's init call
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
            // dbgmsg << "  commlibarg=" << commtype << endl;
            // dbgmsg << ", nprocs=" << nprocs << endl;
            Communicate *newcomm = CommCreator::create(commtype.c_str(),
                    argc, argv,
                    nprocs, comminit, mpicomm);

            Comm = std::unique_ptr<Communicate>(newcomm);

                // cache our node number and node count
            MyNode = Comm->myNode();
            TotalNodes = Comm->getNodes();

                // advance the default random number generator
//                 IpplRandom.AdvanceSeed(Comm->myNode());

                // dbgmsg << "  Comm creation successful." << endl;
                // dbgmsg << *this << endl;
        }

        // dbgmsg << "After comm init: argc = " << argc << ", " << endl;
        // for (unsigned int dbgi=0; dbgi < argc; ++dbgi)
        //   dbgmsg << "  argv[" << dbgi << "] = '" << argv[dbgi] << "'" << endl;

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
                printVersion();
                std::string options = compileOptions();
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
                printVersion();
                std::string options = compileOptions();
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

            } else if ( ( strcmp(argv[i], "--time") == 0 ) ||
                    ( strcmp(argv[i], "-time") == 0 ) ||
                    ( strcmp(argv[i], "--statistics") == 0 ) ||
                    ( strcmp(argv[i], "-stats") == 0 ) ) {
                // The user specified that the program stats be printed at
                // the end of the program.
                PrintStats = true;

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

            } else if ( ( strcmp(argv[i], "--maxfftnodes") == 0 ) ) {
                // Limit the number of nodes that can participate in FFT operations
                if ( (i + 1) < argc && argv[i+1][0] != '-' && atoi(argv[i+1]) > 0 )
                    MaxFFTNodes = atoi(argv[++i]);
                else
                    param_error(argv[i],
                            "Please specify a maximum number of FFT nodes > 0", 0);

            } else if ( ( strcmp(argv[i], "--ipplhelp") == 0 ) ||
                    ( strcmp(argv[i], "-h") == 0 ) ||
                    ( strcmp(argv[i], "-?") == 0 ) ) {
                // print out summary of command line switches and exit
                printHelp(argv);
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
        MyArgc = retargc;
        MyArgv = retargv;
        if (stripargs) {
            argc = retargc;
            argv = retargv;
        }

        // Inform dbgmsg("IpplInfo::IpplInfo", INFORM_ALL_NODES);
        // dbgmsg << "Created IpplInfo.  node = " << MyNode << " out of ";
        // dbgmsg << TotalNodes << ", commlib = " << Comm->name() << endl;

        // now, at end, start the timer running, and print out a summary if asked
        Stats->getTime().stop();
        Stats->getTime().clear();
        Stats->getTime().start();
    }

    // indicate we've created one more Ippl object
    CommInitialized = true;
    NumCreated++;

    // At the very end, print out a summary if requested
    if (printsummary)
        INFOMSG(*this << endl);
}


/////////////////////////////////////////////////////////////////////
// Constructor 2: default constructor.
IpplInfo::IpplInfo() {
    if (NumCreated == 0) {
        Comm = std::make_unique<Communicate>();
        Stats = std::make_unique<IpplStats>();
        Info = std::make_unique<Inform>("Ippl");
        Warn = std::make_unique<Inform>("Warning", std::cerr);
        Error = std::make_unique<Inform>("Error", std::cerr, INFORM_ALL_NODES);
        Debug = std::make_unique<Inform>("**DEBUG**", std::cerr, INFORM_ALL_NODES);
    }

    // just indicate we've also been created
    NumCreated++;
}


/////////////////////////////////////////////////////////////////////
// Constructor 3: copy constructor.
IpplInfo::IpplInfo(const IpplInfo&) {
    if (NumCreated == 0) {
        Comm = std::make_unique<Communicate>();
        Stats = std::make_unique<IpplStats>();
        Info = std::make_unique<Inform>("Ippl");
        Warn = std::make_unique<Inform>("Warning", std::cerr);
        Error = std::make_unique<Inform>("Error", std::cerr, INFORM_ALL_NODES);
        Debug = std::make_unique<Inform>("**DEBUG**", std::cerr, INFORM_ALL_NODES);
    }

    // just indicate we've also been created
    NumCreated++;
}


/////////////////////////////////////////////////////////////////////
// Destructor: need to delete comm library if this is the last IpplInfo
IpplInfo::~IpplInfo() {
    // indicate we have one less instance; if this is the last one,
    // close communication and clean up
    // Inform dbgmsg("IpplInfo::~IpplInfo", INFORM_ALL_NODES);
    // dbgmsg << "In destructor: Current NumCreated = " << NumCreated << endl;

    if ((--NumCreated) == 0) {
        // at end of program, print statistics if requested to do so
        if (PrintStats) {
            Inform statsmsg("Stats", INFORM_ALL_NODES);
            statsmsg << *this;
            printStatistics(statsmsg);
        }

        CommInitialized = false;

        // delete other dynamically-allocated static objects
        delete [] MyArgv;
        MyArgv = 0;
        Stats.reset();
    }
}


/////////////////////////////////////////////////////////////////////
// equal operator
IpplInfo& IpplInfo::operator=(const IpplInfo&) {
    // nothing to do, we don't even need to indicate we've made another
    return *this;
}

void IpplInfo::abort(const char *msg) {
    // print out message, if one was provided
    if (msg != 0) {
        ERRORMSG(msg << endl);
    }

    // print out final stats, if necessary
    if (PrintStats) {
        Inform statsmsg("Stats", INFORM_ALL_NODES);
        statsmsg << IpplInfo();
        printStatistics(statsmsg);
    }

    // that's it, folks this error will be propperly catched in the main
    throw std::runtime_error("Error form IpplInfo::abort");
}


/////////////////////////////////////////////////////////////////////
// Signal to ALL the nodes that we should exit or abort.  If we abort,
// a core file will be produced.  If we exit, no core file will be made.
// The node which calls abortAllNodes will print out the given message;
// the other nodes will print out that they are aborting due to a message
// from this node.
void IpplInfo::abortAllNodes(const char *msg) {
    // print out message, if one was provided
    if (msg != 0) {
        ERRORMSG(msg << endl);
    }

    // print out final stats, if necessary
    if (PrintStats) {
        Inform statsmsg("Stats", INFORM_ALL_NODES);
        statsmsg << IpplInfo();
        printStatistics(statsmsg);
    }

    // broadcast out the kill message, if necessary
    if (getNodes() > 1)
        Comm->broadcast_others(new Message, IPPL_ABORT_TAG);

    throw std::runtime_error("Error form IpplInfo::abortAllNodes");

}

/////////////////////////////////////////////////////////////////////
// getNodes: return the number of 'Nodes' in use for the computation
int IpplInfo::getNodes() {
    return TotalNodes;
}


/////////////////////////////////////////////////////////////////////
// myNode: return which Node we are running on right now
int IpplInfo::myNode() {
    return MyNode;
}



/////////////////////////////////////////////////////////////////////
// printVersion: print out a version summary.  If the argument is true,
// print out a detailed listing, otherwise a summary.
void IpplInfo::printVersion(void) {
    INFOMSG("IPPL Framework version " << version() << endl);
    INFOMSG("Last build date: " << compileDate() << " by user ");
    INFOMSG(compileUser() << endl);
    INFOMSG("Built for machine: " << compileMachine() << endl);
}


void IpplInfo::printHelp(char** argv) {
    INFOMSG("Usage: " << argv[0] << " [<option> <option> ...]\n");
    INFOMSG("       The possible values for <option> are:\n");
    INFOMSG("   --summary           : Print IPPL lib summary at start.\n");
    INFOMSG("                         <x> = ");
    INFOMSG(CommCreator::getAllLibraryNames() << "\n");
    INFOMSG("                         initialization, assume already done.\n");
    INFOMSG("   --time              : Show total time used in execution.\n");
    INFOMSG("   --notime            : Do not show timing info (default).\n");
    INFOMSG("   --info <n>          : Set info message level.  0 = off.\n");
    INFOMSG("   --warn <n>          : Set warning message level.  0 = off.\n");
    INFOMSG("   --error <n>         : Set error message level.  0 = off.\n");
    INFOMSG("   --debug <n>         : Set debug message level.  0 = off.\n");
    INFOMSG("   --maxfftnodes <n>   : Limit the nodes that work on FFT's.\n");
}

/////////////////////////////////////////////////////////////////////
// here: as in stop in IpplInfo::here (in the debugger)
void IpplInfo::here()
{
}

/////////////////////////////////////////////////////////////////////
// print out statistics to the given Inform stream
void IpplInfo::printStatistics(Inform &o) { Stats->print(o); }


/////////////////////////////////////////////////////////////////////
// version: return the name of this version of Ippl, as a string
// (from Versions.h)
const char *IpplInfo::version() {
    return ippl_version_name;
}



/////////////////////////////////////////////////////////////////////
// compileArch: return the architecture on which this library was built
// (from IpplVersions.h)
const char *IpplInfo::compileArch() {
    return ippl_compile_arch;
}


/////////////////////////////////////////////////////////////////////
// compileDate: return the date on which this library was prepared for
// compilation (from IpplVersions.h)
const char *IpplInfo::compileDate() {
    return ippl_compile_date;
}


/////////////////////////////////////////////////////////////////////
// compileLine: return the compiler command used to compile each source file
// (from IpplVersions.h)
const char *IpplInfo::compileLine() {
    return ippl_compile_line;
}


/////////////////////////////////////////////////////////////////////
// compileMachine: return the machine on which this library was
// compiled (from IpplVersions.h)
const char *IpplInfo::compileMachine() {
    return ippl_compile_machine;
}


/////////////////////////////////////////////////////////////////////
// compileOptions: return the option list used to compile this library
// (from IpplVersions.h)
const char *IpplInfo::compileOptions() {
    return ippl_compile_options;
}


/////////////////////////////////////////////////////////////////////
// compileUser: return the username of the user who compiled this
// library (from IpplVersions.h)
const char *IpplInfo::compileUser() {
    return ippl_compile_user;
}


/////////////////////////////////////////////////////////////////////
// param_error: print out an error message when an illegal cmd-line
// parameter is encountered.
// Arguments are: parameter, error message, bad value (if any)
void IpplInfo::param_error(const char *param, const char *msg,
        const char *bad) {
    if ( param != 0 )
        ERRORMSG(param << " ");
    if ( bad != 0 )
        ERRORMSG(bad << " ");
    if ( msg != 0 )
        ERRORMSG(": " << msg);
    ERRORMSG(endl);
    IpplInfo::abort(0);
}

void IpplInfo::param_error(const char *param, const char *msg1,
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
    IpplInfo::abort(0);
}