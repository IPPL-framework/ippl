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
#include "Utility/RandomNumberGen.h"
#include "Utility/vmap.h"
#include "DataSource/DataConnectCreator.h"
#include "Message/CommCreator.h"
#include "Message/Communicate.h"

#include "IpplVersions.h"

#include <unistd.h>
#include <cstdio>
#include <csignal>

/////////////////////////////////////////////////////////////////////
// public static members of IpplInfo, initialized to default values
Communicate *IpplInfo::Comm = 0;
IpplStats  *IpplInfo::Stats = 0;
Inform *IpplInfo::Info = 0;
Inform *IpplInfo::Warn = 0;
Inform *IpplInfo::Error = 0;
Inform *IpplInfo::Debug = 0;

void IpplInfo::instantiateGlobals() {
    if (Comm == 0)
        Comm = new Communicate();
    if (Stats == 0)
        Stats = new IpplStats();
    if (Info == 0)
        Info = new Inform("Ippl");
    if (Warn == 0)
        Warn = new Inform("Warning", std::cerr);
    if (Error == 0)
        Error = new Inform("Error", std::cerr, INFORM_ALL_NODES);
    if (Debug == 0)
        Debug = new Inform("**DEBUG**", std::cerr, INFORM_ALL_NODES);
}

void IpplInfo::deleteGlobals() {
    delete Comm;
    delete Stats;
    delete Info;
    delete Warn;
    delete Error;
    delete Debug;

    Comm = 0;
    Stats = 0;
    Info = 0;
    Warn = 0;
    Error = 0;
    Debug = 0;
}

std::stack<StaticIpplInfo> IpplInfo::stashedStaticMembers;

// should we use the optimization of deferring guard cell fills until
// absolutely needed?  Can be changed to true by specifying the
// flag --defergcfill
bool IpplInfo::deferGuardCellFills = false;

// should we use the compression capabilities in {[Bare]Field,LField}? Can be
// changed to false by specifying the flag --nofieldcompression
bool IpplInfo::noFieldCompression = false;

// private static members of IpplInfo, initialized to default values
MPI_Comm IpplInfo::communicator_m = MPI_COMM_WORLD;
int  IpplInfo::NumCreated = 0;
bool IpplInfo::CommInitialized = false;
bool IpplInfo::PrintStats = false;
bool IpplInfo::NeedDeleteComm = false;
int  IpplInfo::MyArgc = 0;
char **IpplInfo::MyArgv = 0;
int  IpplInfo::MyNode = 0;
int  IpplInfo::TotalNodes = 1;
int  IpplInfo::NumSMPs = 1;
int* IpplInfo::SMPIDList = 0;
int* IpplInfo::SMPNodeList = 0;
bool IpplInfo::UseChecksums = false;
bool IpplInfo::Retransmit = false;
int  IpplInfo::MaxFFTNodes = 0;
int  IpplInfo::ChunkSize = 512*1024; // 512K == 64K doubles
bool IpplInfo::PerSMPParallelIO = false;
bool IpplInfo::offsetStorage = false;
bool IpplInfo::extraCompressChecks = false;

/////////////////////////////////////////////////////////////////////
// print out current state to the given output stream
std::ostream& operator<<(std::ostream& o, const IpplInfo&) {
    o << "------------------------------------------\n";
    o << "IPPL Framework Application Summary:\n";
    o << "  Running on node " << IpplInfo::myNode();
    o << ", out of " << IpplInfo::getNodes() << " total.\n";
    o << "  Number of SMPs: " << IpplInfo::getSMPs() << "\n";
    o << "  Relative SMP node: " << IpplInfo::mySMPNode();
    o << ", out of " << IpplInfo::getSMPNodes(IpplInfo::mySMP());
    o << " nodes.\n";
    o << "  Communication method: " << IpplInfo::Comm->name() << "\n";
    o << "  Disc read chunk size: " << IpplInfo::chunkSize() << " bytes.\n";
    o << "  Deferring guard cell fills? ";
    o << IpplInfo::deferGuardCellFills << "\n";
    o << "  Turning off Field compression? ";
    o << IpplInfo::noFieldCompression << "\n";
    o << "  Offsetting storage? ";
    o << IpplInfo::offsetStorage << "\n";
    o << "  Using extra compression checks in expressions? ";
    o << IpplInfo::extraCompressChecks << "\n";
    o << "  Use per-SMP parallel IO? ";
    o << IpplInfo::perSMPParallelIO() << "\n";
    o << "  Computing message CRC checksums? ";
    o << IpplInfo::useChecksums() << "\n";
    o << "  Retransmit messages on error (only if checkums on)? ";
    o << IpplInfo::retransmit() << "\n";

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
    int connectoption = (-1);     // for connection method option
    int retargc;			// number of args to return to caller
    char **retargv;		// arguments to return
    bool printsummary = false;	// print summary at end of constructor

    //Inform dbgmsg("IpplInfo(argc,argv)", INFORM_ALL_NODES);

    // determine whether we should strip out ippl-specific arguments, or keep
    bool stripargs = (removeargs != KEEP);

    communicator_m = mpicomm;

    if (NumCreated == 0) {
        Comm = new Communicate();
        Stats = new IpplStats();
        Info = new Inform("Ippl");
        Warn = new Inform("Warning", std::cerr);
        Error = new Inform("Error", std::cerr, INFORM_ALL_NODES);
        Debug = new Inform("**DEBUG**", std::cerr, INFORM_ALL_NODES);
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

        for (i=1; i < argc; ++i) {
            if ( ( strcmp(argv[i], "--processes") == 0 ) ||
                    ( strcmp(argv[i], "-procs") == 0 ) ) {
                // The user specified how many processes to use. This may not be useful
                // for all communication methods.
                if ( (i + 1) < argc && argv[i+1][0] != '-' && atoi(argv[i+1]) > 0 )
                    nprocs = atoi(argv[++i]);
                else
                    param_error(argv[i],
                            "Please specify a positive number of processes", 0);
            } else if ( ( strcmp(argv[i], "--commlib") == 0 ) ||
                    ( strcmp(argv[i], "-comm") == 0 ) ) {
                // The user specified what kind of comm library to use
                if ( (i + 1) < argc && argv[i+1][0] != '-' ) {
                    commtype = argv[++i];
                    startcomm = true;
                } else {
                    param_error(argv[i], "Please use one of: ",
                                CommCreator::getAllLibraryNames(), 0);
                    startcomm = false;
                }

            } else if ( strcmp(argv[i], "--nocomminit") == 0 ) {
                // The user requested that we do not let the run-time system call
                // whatever initialization routine it might have (like MPI_Init).
                // This is in case another agency has already done the initialization.
                comminit = false;
            }
        }

        // create Communicate object now.
        // dbgmsg << "Setting up parallel environment ..." << endl;
        if (startcomm && nprocs != 0 && nprocs != 1) {
            // dbgmsg << "  commlibarg=" << commtype << endl;
            // dbgmsg << ", nprocs=" << nprocs << endl;
            Communicate *newcomm = CommCreator::create(commtype.c_str(),
                    argc, argv,
                    nprocs, comminit, mpicomm);

            if (newcomm == 0) {
                if (CommCreator::supported(commtype.c_str()))
                    param_error("--commlib", "Could not initialize this ",
                            "communication library.", commtype.c_str());
                else if (CommCreator::known(commtype.c_str()))
                    param_error("--commlib", "This communication library is not ",
                            "available.", commtype.c_str());
                else
                    param_error("--commlib", "Please use one of: ",
                            CommCreator::getAllLibraryNames(), 0);
            } else {
                // success, we have a new comm object
                NeedDeleteComm = true;
                delete Comm;
                Comm = newcomm;

                // cache our node number and node count
                MyNode = Comm->myNode();
                TotalNodes = Comm->getNodes();
                find_smp_nodes();

                // advance the default random number generator
                IpplRandom.AdvanceSeed(Comm->myNode());

                // dbgmsg << "  Comm creation successful." << endl;
                // dbgmsg << *this << endl;
            }
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
            if ( ( strcmp(argv[i], "--processes") == 0 ) ||
                    ( strcmp(argv[i], "-procs") == 0 ) ) {
                // handled above
                if ( (i + 1) < argc && argv[i+1][0] != '-' && atoi(argv[i+1]) > 0 )
                    ++i;

            } else if ( ( strcmp(argv[i], "--nocomminit") == 0 ) ) {
                // handled above, nothing to do here but skip the arg

            } else if ( ( strcmp(argv[i], "--summary") == 0 ) ) {
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

            } else if ( ( strcmp(argv[i], "--checksums") == 0 ) ||
                    ( strcmp(argv[i], "--checksum") == 0 ) ) {
                UseChecksums = true;

            } else if ( ( strcmp(argv[i], "--retransmit") == 0 ) ) {
                Retransmit = true;

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

            } else if ( ( strcmp(argv[i], "--connect") == 0 ) ) {
                // Set the default external connection method
                if ( (i + 1) < argc && argv[i+1][0] != '-' )
                    connectoption = ++i;
                else
                    param_error(argv[i], "Please use one of: ",
                            DataConnectCreator::getAllMethodNames(), 0);

            } else if ( ( strcmp(argv[i], "--connectnodes") == 0 ) ) {
                // Set the number of nodes that are used in connections, by default
                if ( (i + 1) < argc && argv[i+1][0] != '-' && atoi(argv[i+1]) > 0 )
                    DataConnectCreator::setDefaultNodes(atoi(argv[++i]));
                else
                    param_error(argv[i],
                            "Please specify a number of nodes for connections > 0",
                            0);

            } else if ( ( strcmp(argv[i], "--commlib") == 0 ) ||
                    ( strcmp(argv[i], "-comm") == 0 ) ) {
                // handled above
                if ( (i + 1) < argc && argv[i+1][0] != '-' )
                    ++i;

            } else if   ( strcmp(argv[i], "--profile") == 0 )  {
                // handled above in
                if ( (i + 1) < argc && argv[i+1][0] != '-' )
                    ++i;

            } else if ( ( strcmp(argv[i], "--persmppario") == 0 ) ) {
                // Turn on the ability to use per-smp parallel IO
                PerSMPParallelIO = true;

            } else if ( ( strcmp(argv[i], "--nopersmppario") == 0 ) ) {
                // Turn off the ability to use per-smp parallel IO
                PerSMPParallelIO = false;

            } else if ( ( strcmp(argv[i], "--chunksize") == 0 ) ) {
                // Set the I/O chunk size, used to limit how many items
                // are read in or written to disk at one time.
                if ( (i + 1) < argc && argv[i+1][0] != '-' && atoi(argv[i+1]) >= 0 ) {
                    ChunkSize = atoi(argv[++i]);
                    char units = static_cast<char>(toupper(argv[i][strlen(argv[i])-1]));
                    if (units == 'K')
                        ChunkSize *= 1024;
                    else if (units == 'M')
                        ChunkSize *= 1024*1024;
                    else if (units == 'G')
                        ChunkSize *= 1024*1024*1024;
                } else {
                    param_error(argv[i],
                            "Please specify a timeout value (in seconds)", 0);
                }
            } else if ( ( strcmp(argv[i], "--defergcfill") == 0 ) ) {
                // Turn on the defer guard cell fill optimization
                deferGuardCellFills = true;

            } else if ( ( strcmp(argv[i], "--offsetstorage") == 0 ) ) {
                // Turn on the offset-storage modification to LFields
                offsetStorage = true;

            } else if ( ( strcmp(argv[i], "--extracompcheck") == 0 ) ) {
                // Turn on the extra compression checks in expressions
                extraCompressChecks = true;

            } else if ( ( strcmp(argv[i], "--nofieldcompression") == 0 ) ) {
                // Turn off compression in the Field classes
                noFieldCompression = true;

            } else if ( ( strcmp(argv[i], "--directio") == 0 ) ) {
                // Turn on the use of Direct-IO, if possible
                param_error(argv[i],
                        "Direct-IO is not available in this build of IPPL", 0);
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

        // Select the default connection method
        if ( connectoption >= 0 ) {
            if ( ! DataConnectCreator::setDefaultMethod(argv[connectoption]) ) {
                if (DataConnectCreator::supported(argv[connectoption]))
                    param_error(argv[connectoption - 1], "Could not initialize this ",
                            "connection.", argv[connectoption]);
                else if (DataConnectCreator::known(argv[connectoption]))
                    param_error(argv[connectoption - 1],"This connection method is not ",
                            "available.", argv[connectoption]);
                else
                    param_error(argv[connectoption - 1], "Please use one of: ",
                            DataConnectCreator::getAllMethodNames(), 0);
            }
        }

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
        Comm = new Communicate();
        Stats = new IpplStats();
        Info = new Inform("Ippl");
        Warn = new Inform("Warning", std::cerr);
        Error = new Inform("Error", std::cerr, INFORM_ALL_NODES);
        Debug = new Inform("**DEBUG**", std::cerr, INFORM_ALL_NODES);
    }

    // just indicate we've also been created
    NumCreated++;
}


/////////////////////////////////////////////////////////////////////
// Constructor 3: copy constructor.
IpplInfo::IpplInfo(const IpplInfo&) {
    if (NumCreated == 0) {
        Comm = new Communicate();
        Stats = new IpplStats();
        Info = new Inform("Ippl");
        Warn = new Inform("Warning", std::cerr);
        Error = new Inform("Error", std::cerr, INFORM_ALL_NODES);
        Debug = new Inform("**DEBUG**", std::cerr, INFORM_ALL_NODES);
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

        // Delete the communications object, if necessary, to shut down parallel
        // environment
        if (NeedDeleteComm) {
             // dbgmsg << "  Deleting comm object, since now NumCreated = ";
             // dbgmsg << NumCreated << endl;
             delete Comm;
             Comm = 0;
             NeedDeleteComm = false;
        }
        CommInitialized = false;

        // delete other dynamically-allocated static objects
        delete [] MyArgv;
        if (SMPIDList != 0) {
            delete [] SMPIDList;
        }
        if (SMPNodeList != 0) {
            delete [] SMPNodeList;
        }
        delete Stats;

        MyArgv = 0;
        SMPIDList = 0;
        SMPNodeList = 0;
        Stats = 0;
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

    // delete communication object, if necessary
    if (NeedDeleteComm) {
        NeedDeleteComm = false;
        delete Comm;
        Comm = 0;
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
// getContexts: return the number of 'Contexts' for the given node
int IpplInfo::getContexts(const int n) {
    return Comm->getContexts(n);
}


/////////////////////////////////////////////////////////////////////
// getProcesses: return the number of 'Processes' for the given Node and Context
int IpplInfo::getProcesses(const int n, const int c) {
    return Comm->getProcesses(n, c);
}


/////////////////////////////////////////////////////////////////////
// myNode: return which Node we are running on right now
int IpplInfo::myNode() {
    return MyNode;
}


/////////////////////////////////////////////////////////////////////
// getSMPs: return number of SMP's (each of which may be running
// several processes)
int IpplInfo::getSMPs() {
    return NumSMPs;
}


/////////////////////////////////////////////////////////////////////
// getSMPNodes: return number of nodes on the SMP with the given index
int IpplInfo::getSMPNodes(int smpindx) {
    int num = 0;
    if (SMPIDList == 0) {
        num = 1;
    } else {
        for (int i=0; i < TotalNodes; ++i)
            if (SMPIDList[i] == smpindx)
                num++;
    }
    return num;
}


/////////////////////////////////////////////////////////////////////
// mySMP: return ID of my SMP (numbered 0 ... getSMPs() - 1)
int IpplInfo::mySMP() {
    return (SMPIDList != 0 ? SMPIDList[MyNode] : 0);
}


/////////////////////////////////////////////////////////////////////
// mySMPNode: return relative node number within the nodes on our SMP
int IpplInfo::mySMPNode() {
    return (SMPNodeList != 0 ? SMPNodeList[MyNode] : 0);
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
    INFOMSG("   --processes <n>     : Number of parallel nodes to use.\n");
    INFOMSG("   --commlib <x>       : Selects a parallel comm. library.\n");
    INFOMSG("                         <x> = ");
    INFOMSG(CommCreator::getAllLibraryNames() << "\n");
    INFOMSG("   --nocomminit        : IPPL does not do communication\n");
    INFOMSG("                         initialization, assume already done.\n");
    INFOMSG("   --connect <x>       : Select external connection method.\n");
    INFOMSG("                         <x> = ");
    INFOMSG(DataConnectCreator::getAllMethodNames() << "\n");
    INFOMSG("   --time              : Show total time used in execution.\n");
    INFOMSG("   --notime            : Do not show timing info (default).\n");
    INFOMSG("   --info <n>          : Set info message level.  0 = off.\n");
    INFOMSG("   --warn <n>          : Set warning message level.  0 = off.\n");
    INFOMSG("   --error <n>         : Set error message level.  0 = off.\n");
    INFOMSG("   --debug <n>         : Set debug message level.  0 = off.\n");
    /*#ifdef PROFILING_ON
      INFOMSG("   --profile <gr>  : Enable profiling for groups (e.g., M+P+io) \n");
      INFOMSG("             M - Message, P - Pete, V - Viz, A - Assign, I - IO\n");
      INFOMSG("             F - Field, L - Layout, S - Sparse, D - Domainmap \n");
      INFOMSG("             Ut - Utility, R - Region, Ff - FFT \n");
      INFOMSG("             U - User, 1 - User1, 2 - User2, 3 - User3, 4 - User4\n");

      #endif*/ //PROFILING_ON
    INFOMSG("   --defergcfill       : Turn on deferred guard cell fills.\n");
    INFOMSG("   --nofieldcompression: Turn off compression in the Field classes.\n");
    INFOMSG("   --offsetstorage     : Turn on random LField storage offsets.\n");
    INFOMSG("   --extracompcheck    : Turn on extra compression checks in evaluator.\n");
    INFOMSG("   --checksums         : Turn on CRC checksums for messages.\n");
    INFOMSG("   --retransmit        : Resent messages if a CRC error occurs.\n");
    INFOMSG("   --maxfftnodes <n>   : Limit the nodes that work on FFT's.\n");
    INFOMSG("   --chunksize <n>     : Set I/O chunk size.  Can end w/K,M,G.\n");
    INFOMSG("   --persmppario       : Enable on-SMP parallel IO option.\n");
    INFOMSG("   --nopersmppario     : Disable on-SMP parallel IO option (default).\n");
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


/////////////////////////////////////////////////////////////////////
// find out how many SMP's there are, and which processor we are on
// our local SMP (e.g., if there are two SMP's with 4 nodes each,
// the process will have a node number from 0 ... 7, and an SMP node
// number from 0 ... 3
void IpplInfo::find_smp_nodes() {
    // Inform dbgmsg("IpplInfo::find_smp_nodes", INFORM_ALL_NODES);

    // create a tag for use in sending info to/from other nodes
    int tag = Comm->next_tag(IPPL_MAKE_HOST_MAP_TAG, IPPL_TAG_CYCLE);

    // create arrays to store the Node -> SMP mapping, and the relative
    // SMP node number
    if (SMPIDList != 0)
        delete [] SMPIDList;
    if (SMPNodeList != 0)
        delete [] SMPNodeList;
    SMPIDList   = new int[TotalNodes];
    SMPNodeList = new int[TotalNodes];

    // obtain the hostname and processor ID to send out
    char name[1024];
    if (gethostname(name, 1023) != 0) {
        ERRORMSG("Could not get hostname ... using localhost." << endl);
        strcpy(name, "localhost");
    }
    std::string NodeName(name,strlen(name));
    // dbgmsg << "My hostname is " << NodeName << endl;

    // all other nodes send their hostname to node 0; node 0 gets the names,
    // maps pnode ID's -> SMP ID's, then broadcasts all the necessary info to
    // all other nodes
    if (MyNode != 0) {
        // other nodes send their node name to node 0
        Message *msg = new Message;
        ::putMessage(*msg,NodeName);
        // dbgmsg << "Sending my name to node 0." << endl;
        Comm->send(msg, 0, tag);

        // receive back the SMPIDList mapping
        int node = 0;
        msg = Comm->receive_block(node, tag);
        PInsist(msg != 0 && node == 0,
                "SPMDList map not received from master in IpplInfo::find_smp_nodes!!");
        ::getMessage_iter(*msg, SMPIDList);
        ::getMessage_iter(*msg, SMPNodeList);
        delete msg;
    }
    else {
        // collect node names from everyone else, and then retransmit the collected
        // list.
        SMPIDList[0] = 0;
        vmap<std::string,int> smpMap;
        vmap<std::string,int>::iterator smpiter;
        smpMap.insert(vmap<std::string,int>::value_type(NodeName, 0));
        unsigned int unreceived = TotalNodes - 1;
        while (unreceived-- > 0) {
            // get the hostname from the remote node
            int node = COMM_ANY_NODE;
            Message *msg = Comm->receive_block(node, tag);
            PInsist(msg != 0,
                    "Hostname not received by master in IpplInfo::find_smp_nodes!!");
            std::string nodename;
            ::getMessage(*msg,nodename);
            delete msg;
            // dbgmsg <<"Received name '"<< nodename <<"' from node "<< node<<endl;

            // put it in the mapping from hostname -> SMP ID, if necessary
            smpiter = smpMap.find(nodename);
            if (smpiter == smpMap.end())
                smpMap.insert(vmap<std::string,int>::value_type(nodename,smpMap.size()));

            // from the hostname, get the SMP ID number and store it in SMPIDList
            SMPIDList[node] = smpMap[nodename];
        }

        // convert from SMPID mapping -> relative node number
        for (int smpindx = 0; (unsigned int) smpindx < smpMap.size(); ++smpindx) {
            int smpnodes = 0;
            for (int n=0; n < TotalNodes; ++n) {
                if (SMPIDList[n] == smpindx)
                    SMPNodeList[n] = smpnodes++;
            }
        }

        // broadcast SMP info to other nodes
        if (TotalNodes > 1) {
            Message *msg = new Message;
            ::putMessage(*msg, SMPIDList, SMPIDList + TotalNodes);
            ::putMessage(*msg, SMPNodeList, SMPNodeList + TotalNodes);
            Comm->broadcast_others(msg, tag);
        }
    }

    // compute number of SMP's ... necessary for all but node 0, but we'll do
    // it for all
    NumSMPs = 0;
    for (int ns=0; ns < TotalNodes; ++ns)
        if (SMPNodeList[ns] == 0)
            NumSMPs++;

    // dbgmsg << "Results of SMP mapping: NumSMPs = " << NumSMPs << endl;
    // for (unsigned int n=0; n < TotalNodes; ++n) {
    //   dbgmsg << "  n=" << n << ", SMPID=" << SMPIDList[n] << ", SMPNode=";
    //   dbgmsg << SMPNodeList[n] << endl;
    // }
}

void IpplInfo::stash() {
    PAssert_EQ(stashedStaticMembers.size(), 0);

    StaticIpplInfo obj;

    obj.Comm =                Comm;
    obj.Stats =               Stats;
    obj.Info =                Info;
    obj.Warn =                Warn;
    obj.Error =               Error;
    obj.Debug =               Debug;
    obj.deferGuardCellFills = deferGuardCellFills;
    obj.noFieldCompression =  noFieldCompression;
    obj.offsetStorage =       offsetStorage;
    obj.extraCompressChecks = extraCompressChecks;
    obj.communicator_m =      communicator_m;
    obj.NumCreated =          NumCreated;
    obj.CommInitialized =     CommInitialized;
    obj.PrintStats =          PrintStats;
    obj.NeedDeleteComm =      NeedDeleteComm;
    obj.UseChecksums =        UseChecksums;
    obj.Retransmit =          Retransmit;
    obj.MyArgc =              MyArgc;
    obj.MyArgv =              MyArgv;
    obj.MyNode =              MyNode;
    obj.TotalNodes =          TotalNodes;
    obj.NumSMPs =             NumSMPs;
    obj.SMPIDList =           SMPIDList;
    obj.SMPNodeList =         SMPNodeList;
    obj.MaxFFTNodes =         MaxFFTNodes;
    obj.ChunkSize =           ChunkSize;
    obj.PerSMPParallelIO =    PerSMPParallelIO;

    stashedStaticMembers.push(obj);

    Comm = 0;
    Stats = 0;
    Info = 0;
    Warn = 0;
    Error = 0;
    Debug = 0;

    deferGuardCellFills = false;
    noFieldCompression = false;
    offsetStorage = false;
    extraCompressChecks = false;
    communicator_m = MPI_COMM_WORLD;
    NumCreated = 0;
    CommInitialized = false;
    PrintStats = false;
    NeedDeleteComm = false;
    UseChecksums = false;
    Retransmit = false;
    MyArgc = 0;
    MyArgv = 0;
    MyNode = 0;
    TotalNodes = 1;
    NumSMPs = 1;
    SMPIDList = 0;
    SMPNodeList = 0;
    MaxFFTNodes = 0;
    ChunkSize = 512*1024; // 512K == 64K doubles
    PerSMPParallelIO = false;
}

void IpplInfo::pop() {
    PAssert_EQ(stashedStaticMembers.size(), 1);

    StaticIpplInfo obj = stashedStaticMembers.top();
    stashedStaticMembers.pop();
    // Delete the communications object, if necessary, to shut down parallel
    // environment
    // Comm is deleted in destructor
    delete [] MyArgv;
    delete [] SMPIDList;
    delete [] SMPNodeList;
    delete Info;
    delete Warn;
    delete Error;
    delete Debug;
    delete Stats;

    Comm =                obj.Comm;
    Stats =               obj.Stats;
    Info =                obj.Info;
    Warn =                obj.Warn;
    Error =               obj.Error;
    Debug =               obj.Debug;
    deferGuardCellFills = obj.deferGuardCellFills;
    noFieldCompression =  obj.noFieldCompression;
    offsetStorage =       obj.offsetStorage;
    extraCompressChecks = obj.extraCompressChecks;
    communicator_m =      obj.communicator_m;
    NumCreated =          obj.NumCreated;
    CommInitialized =     obj.CommInitialized;
    PrintStats =          obj.PrintStats;
    NeedDeleteComm =      obj.NeedDeleteComm;
    UseChecksums =        obj.UseChecksums;
    Retransmit =          obj.Retransmit;
    MyArgc =              obj.MyArgc;
    MyArgv =              obj.MyArgv;
    MyNode =              obj.MyNode;
    TotalNodes =          obj.TotalNodes;
    NumSMPs =             obj.NumSMPs;
    SMPIDList =           obj.SMPIDList;
    SMPNodeList =         obj.SMPNodeList;
    MaxFFTNodes =         obj.MaxFFTNodes;
    ChunkSize =           obj.ChunkSize;
    PerSMPParallelIO =    obj.PerSMPParallelIO;
}
