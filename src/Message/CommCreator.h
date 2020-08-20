// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef COMM_CREATOR_H
#define COMM_CREATOR_H

/***********************************************************************
 *
 * CommCreator is a factory class which is used to create specific
 * subclasses of the base class Communicate.  To create a new library-
 * specific Communicate, the user can either instantiate such a class
 * directly, or they can call the static method
 *   CommCreator::create(CommCreator::CreateMethod, int argc, char *argv[],
 *                       int nodes)
 *
 ***********************************************************************/

#include <mpi.h>

// forward declarations
class Communicate;


// class definition
class CommCreator
{

public:

  // enumeration of communication libraries
  enum { MPI, SERIAL, COMMLIBRARIES };

public:
    // constructor and destructor
    CommCreator() { }
    ~CommCreator() { }

    //
    // informative methods
    //

    // return the number of libraries available
    static int getNumLibraries()
    {
        return COMMLIBRARIES;
    }

    // return the name of the Nth library
    static const char *getLibraryName(int);

    // return a list of all the libraries, as a single string
    static const char *getAllLibraryNames();

    // is the given comm library available here?
    static bool supported(int);
    static bool supported(const char *nm)
    {
        return supported(libindex(nm));
    }

    // is the given comm library name one we recognize?
    static bool known(const char *nm)
    {
        return (libindex(nm) >= 0);
    }

    //
    // Communicate create methods
    //

    // create a new Communicate object.  Arguments = type, argc, argv, num nodes,
    // do init
    static Communicate *create(int, int&, char** &, int = (-1), bool = true, 
                               MPI_Comm mpicomm = MPI_COMM_WORLD);

    // same as above, but specifying the type as a string instead of an int
    static Communicate *create(const char *nm, int& argc, char **& argv,
                               int nodes= (-1), bool doinit = true, 
                               MPI_Comm mpicomm = MPI_COMM_WORLD)
    {
        return create(libindex(nm), argc, argv, nodes, doinit, mpicomm);
    }

private:
    // return the index of the given named library, or (-1) if not found
    static int libindex(const char *);
};

#endif // COMM_CREATOR_H
