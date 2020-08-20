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

// test program to demonstrate use of parallel reduction
#include "Ippl.h"

int main(int argc, char *argv[]) {
  Ippl ippl(argc,argv);
  int myN = Ippl::myNode();
  //
  // scalar reduction
  //  
  int localData = myN;
  int globalRes = 0;
  
  reduce(localData, globalRes, OpAddAssign());
  INFOMSG("Scalar reduction (compute sum) " << globalRes << endl);
  return 0;
}

/***************************************************************************
 * $RCSfile: reduce.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:38 $
 * IPPL_VERSION_ID: $Id: reduce.cpp,v 1.1.1.1 2003/01/23 07:40:38 adelmann Exp $ 
 ***************************************************************************/
