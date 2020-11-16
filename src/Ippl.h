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
#ifndef IPPL_H
#define IPPL_H

#include <iostream>

#include "Utility/Inform.h"

#include "Message/CommBoostMpi.h"
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
};

// macros used to print out messages to the console or a directed file
#define INFOMSG(msg)  { *Ippl::Info << msg; }
#define WARNMSG(msg)  { *Ippl::Warn << msg; }
#define ERRORMSG(msg) { *Ippl::Error << msg; }


//FIMXE remove (only for backwards compatibility)
#include "IpplCore.h"

#endif
