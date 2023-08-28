//
// Class IpplInfo
//   Global Ippl information.
//
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

class IpplInfo {
public:
    // printVersion: print out a version summary.  If the argument is true,
    // print out a detailed listing, otherwise a summary.
    static void printVersion(void);

    static void printHelp(char** argv);

    // version: return the name of this version of Ippl, as a string
    // (from IpplVersions.h)
    static const char* version();

    // compileArch: return the architecture on which this library was built
    // (from IpplVersions.h)
    static const char* compileArch();

    // compileDate: return the date on which this library was prepared for
    // compilation (from IpplVersions.h)
    static const char* compileDate();

    // compileLine: return the compiler command used to compile each source file
    // (from IpplVersions.h)
    static const char* compileLine();

    // compileMachine: return the machine on which this library was
    // compiled (from IpplVersions.h)
    static const char* compileMachine();

    // compileOptions: return the option list used to compile this library
    // (from IpplVersions.h)
    static const char* compileOptions();

    // compileUser: return the username of the user who compiled this
    // library (from IpplVersions.h)
    static const char* compileUser();
};

#endif
