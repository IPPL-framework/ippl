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
 *
 ***************************************************************************/

/***************************************************************************
 *
 * A simple test program to test using IPPL with another MPI user.
 *
 ***************************************************************************/

#include "Ippl.h"
#include <mpi.h>

int main(int argc, char *argv[]) {
  int mynodes, myrank;

  // Initialize MPI directly, and use MPI_COMM_WORLD
  MPI_Init(&argc, &argv);
 
  // Get node and rank
  MPI_Comm_size(MPI_COMM_WORLD, &mynodes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  // Initialize IPPL now
  Ippl ippl(argc, argv);
  Inform pmsg(argv[0]);

  // print out info
  pmsg << "Started program." << endl;
  pmsg << "Application nodes = " << mynodes << endl;
  pmsg << "Applicaiton rank  = " << myrank << endl;
  pmsg << "IPPL nodes = " << Ippl::getNodes() << endl;
  pmsg << "IPPL rank  = " << Ippl::myNode() << endl;

  // call MPI_Finalize to clean up
  MPI_Finalize();

  return 0;
}

/***************************************************************************
 * $RCSfile: dualmpi.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:38 $
 * IPPL_VERSION_ID: $Id: dualmpi.cpp,v 1.1.1.1 2003/01/23 07:40:38 adelmann Exp $ 
 ***************************************************************************/
