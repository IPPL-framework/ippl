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

#include "Message/CommBoostMpi.h"

#include <iostream>

//FIXME: Including this header here (regardless of the used commlib) here is
//necessary to enable IPPL to work on a user define communicator group
//(without further increasing the number of defines).
#include <mpi.h>


class IpplInfo {

public:
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
};




#include <boost/mpi/environment.hpp>

class Ippl;
std::ostream& operator<<(std::ostream&, const Ippl&);

class Ippl : public boost::mpi::environment {

public:
    // an enumeration used to indicate whether to KEEP command-line arguments
    // or REMOVE them
    enum { KEEP = 0, REMOVE = 1 };

    // the parallel communication object
    static std::unique_ptr<ippl::Communicate> Comm;

    //   // Inform object to use to print messages to the console (or even to a
    //   // file if requested)
    static std::unique_ptr<Inform> Info;
    static std::unique_ptr<Inform> Warn;
    static std::unique_ptr<Inform> Error;
    static std::unique_ptr<Inform> Debug;

    // Constructor 1: specify the argc, argv values from the cmd line.
    // The second argument controls whether the IPPL-specific command line
    // arguments are stripped out (the default) or left in (if the setting
    // is IpplInfo::KEEP).
    Ippl(int&, char** &, int removeargs = REMOVE, MPI_Comm mpicomm = MPI_COMM_WORLD);

    // Constructor 2: default constructor.  This will not change anything in
    // how the static data members are set up.  This is useful for declaring
    // automatic IpplInfo instances in functions after IpplInfo.has been
    // initially created in the main() routine.
    Ippl() {};

    // Destructor.
    ~Ippl();

    static MPI_Comm getComm() {return communicator_m;}

    static MPI_Comm communicator_m;

    // Static flag indicating whether this class has been created with
    // argc,argv specified ever.  This should only be done once.
    static bool CommInitialized;


    // Kill the communication and throw runtime error exception.
    static void abort(const char * = 0);

    static void deleteGlobals();

private:
      // Indicate an error occurred while trying to parse the given command-line
    // option, and quit.  Arguments are: parameter, error message, bad value
    static void param_error(const char *, const char *, const char *);
    static void param_error(const char *, const char *, const char *, const char *);

    // Static flag indicating whether we should print out stats info at the
    // end of the program.
    static bool PrintStats;
};


// macros used to print out messages to the console or a directed file
#define INFOMSG(msg)  { *Ippl::Info << msg; }
#define WARNMSG(msg)  { *Ippl::Warn << msg; }
#define ERRORMSG(msg) { *Ippl::Error << msg; }


#endif // IPPL_INFO_H

/***************************************************************************
 * $RCSfile: IpplInfo.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: IpplInfo.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $
 ***************************************************************************/
