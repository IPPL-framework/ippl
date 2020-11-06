// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef IPPL_INFO_H
#define IPPL_INFO_H

/*
 * IpplInfo.has the following capabilities:
 *   1) It initializes all globally-used Ippl variables, such as the
 *      Communicate class and other manager classes;
 *   2) It parses command-line arguments to determine how to configure the
 *      global state of the Ippl application;
 *   3) It properly selects and configures the Communicate class, generally
 *      resulting in initialization of the parallel machine;
 *   4) It offers to the user a single class with access member functions to
 *      query for information about the Ippl application (such as what is
 *      the Communicate class instance to use, how many processors are there
 *      in the current run, etc.)
 *
 * The globally-available Ippl objects are (available via IpplInfo::variable)
 *   Communicate *Comm ....... parallel communication object
 *   Inform *Info ............ used to print out informative messages
 *   Inform *Warn ............ used to print out warning messages
 *   Inform *Error ........... used to print out error messages
 *   Inform *Debug ........... used to print out debugging messages
 *
 * Note that you should use the following macros to access the Inform objects,
 * so that these messages may be left out at compile time:
 *   INFOMSG("This is some information " << 34 << endl);
 *   WARNMSG("This is a warning " << 34 << endl);
 *   ERRORMSG("This is an error message " << 34 << endl);
 *
 * There is also a 'typedef IpplInfo Ippl' here, so you can simply use
 * the name 'Ippl' instead of the longer 'IpplInfo' to access this class.
 * 'Ippl' is in fact preferred, as it is shorter.
 */

// include files
#include "Utility/Inform.h"
#include "Message/Communicate.h"

#include <iostream>

//FIXME: Including this header here (regardless of the used commlib) here is
//necessary to enable IPPL to work on a user define communicator group
//(without further increasing the number of defines).
#include <mpi.h>

// forward declarations
class IpplStats;
class IpplInfo;
std::ostream& operator<<(std::ostream&, const IpplInfo&);


class IpplInfo {

public:
  // an enumeration used to indicate whether to KEEP command-line arguments
  // or REMOVE them
  enum { KEEP = 0, REMOVE = 1 };

  // Inform object to use to print messages to the console (or even to a
  // file if requested)
  static std::unique_ptr<Inform> Info;
  static std::unique_ptr<Inform> Warn;
  static std::unique_ptr<Inform> Error;
  static std::unique_ptr<Inform> Debug;

  // the parallel communication object
  static std::unique_ptr<Communicate> Comm;

  // the statistics collection object
  static std::unique_ptr<IpplStats> Stats;


  // Constructor 1: specify the argc, argv values from the cmd line.
  // The second argument controls whether the IPPL-specific command line
  // arguments are stripped out (the default) or left in (if the setting
  // is IpplInfo::KEEP).
  IpplInfo(int&, char** &, int removeargs = REMOVE, MPI_Comm mpicomm = MPI_COMM_WORLD);

  // Constructor 2: default constructor.  This will not change anything in
  // how the static data members are set up.  This is useful for declaring
  // automatic IpplInfo instances in functions after IpplInfo.has been
  // initially created in the main() routine.
  IpplInfo();

  // Constructor 3: copy constructor.  This will only copy non-static members
  // (obviously), if any exist.
  IpplInfo(const IpplInfo&);

  // Destructor.
  ~IpplInfo();

  // Overload the = operator; does the same thing as the copy constructor.
  IpplInfo& operator=(const IpplInfo&);


  /* NOTE: The following initialize/finalize methods have not yet been
     implemented.  Add them to IpplInfo.cpp if they are needed (bfh).
  //
  // Initialize and finalize routines ... initialize can be used if you
  // created IpplInfo with the default constructor, and finalize() can
  // be used to shut down IPPL and possibly exit the program
  //

  // initialize ourselves, if we have not yet done so, by parsing the
  // command-line args and creating the Communication object.  This should
  // be called by all the currently-running nodes.
  void initialize(int &, char ** &);

  // shut down the communication, and possibly exit.  This should be called
  // by all the nodes, it will not work if it is called by just one node
  // and you are running in parallel (in that case, the Communicate subclass
  // destructor will hang).
  void finalize();

  // a version of finalize that will also shut down all the machines via
  // a call to exit()
  void finalize(int exitcode);
  */


  //
  // Standard IPPL action methods (such as abort, etc)
  //

  // Kill the communication and throw runtime error exception.
  static void abort(const char * = 0);

  // Signal to ALL the nodes to abort and throw runtime error exception
  static void abortAllNodes(const char * = 0);

  //
  // Functions which return information about the current Ippl application.
  //

  static MPI_Comm getComm() {return communicator_m;}

  // Return the number of the 'current' node.
  static int myNode();

  // Return the current number of nodes working in parallel, where
  // each node may have more than one processor.  A 'Node' is basically
  // considered as an entity which has a single IP address.
  static int getNodes();

  // Return argc or argv as provided in the initialization
  static int getArgc() { return MyArgc; }
  static char **getArgv() { return MyArgv; }

  // Static data about a limit to the number of nodes that should be used
  // in FFT operations.  If this is <= 0 or > number of nodes, it is ignored.
  static int maxFFTNodes() { return MaxFFTNodes; }

  //
  // Functions which return information about the Ippl library
  //

  // printVersion: print out a version summary.  If the argument is true,
  // print out a detailed listing, otherwise a summary.
  static void printVersion(void);

  static void printHelp(char** argv);

  // version: return the name of this version of Ippl, as a string
  // (from IpplVersions.h)
  static const char *version();

  // compileArch: return the architecture on which this library was built
  // (from IpplVersions.h)
  static const char *compileArch();

  // compileDate: return the date on which this library was prepared for
  // compilation (from IpplVersions.h)
  static const char *compileDate();

  // compileLine: return the compiler command used to compile each source file
  // (from IpplVersions.h)
  static const char *compileLine();

  // compileMachine: return the machine on which this library was
  // compiled (from IpplVersions.h)
  static const char *compileMachine();

  // compileOptions: return the option list used to compile this library
  // (from IpplVersions.h)
  static const char *compileOptions();

  // compileUser: return the username of the user who compiled this
  // library (from IpplVersions.h)
  static const char *compileUser();

  // print out statistics to the given Inform stream
  static void printStatistics(Inform&);

  static void instantiateGlobals();
  static void deleteGlobals();
private:

  static MPI_Comm communicator_m;

  // Static counter indicating how many IpplInit objects have been created.
  // When this gets back to zero, it's time to delete the Comm and quit.
  static int NumCreated;

  // Static flag indicating whether this class has been created with
  // argc,argv specified ever.  This should only be done once.
  static bool CommInitialized;

  // Static flag indicating whether we should print out stats info at the
  // end of the program.
  static bool PrintStats;

  // Static data with argc and argv
  static int MyArgc;
  static char **MyArgv;

  // Static data with my node number and total number of nodes.  These are
  // only changed when a new Communicate object is created.
  static int MyNode;
  static int TotalNodes;

  // Static data about a limit to the number of nodes that should be used
  // in FFT operations.  If this is <= 0 or > number of nodes, it is ignored.
  static int MaxFFTNodes;

  // Indicate an error occurred while trying to parse the given command-line
  // option, and quit.  Arguments are: parameter, error message, bad value
  static void param_error(const char *, const char *, const char *);
  static void param_error(const char *, const char *, const char *, const char *);
};

// macros used to print out messages to the console or a directed file
#define INFOMSG(msg)  { *IpplInfo::Info << msg; }
#define WARNMSG(msg)  { *IpplInfo::Warn << msg; }
#define ERRORMSG(msg) { *IpplInfo::Error << msg; }

// typedef so that we can have a 'Ippl' class that's easier to manipulate
typedef IpplInfo Ippl;


#endif // IPPL_INFO_H

/***************************************************************************
 * $RCSfile: IpplInfo.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: IpplInfo.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $
 ***************************************************************************/
