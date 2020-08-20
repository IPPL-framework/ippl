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
 ***************************************************************************/

// test program to demonstrate use of parallel reduction
#include "Ippl.h"


void report(const char *str, bool result) {
  
  Inform msg("Results");
  msg << "Test " << str << ": ";
  msg << (result ? "PASSED" : "FAILED") << endl;
}


int main(int argc, char *argv[]) {
  
  int i, j;
  Ippl ippl(argc,argv);
  int myN = Ippl::myNode();
  int N   = Ippl::getNodes();

  //
  // scalar reduction
  //

  int local = 1 + myN;
  int global = 999;
  int expected = N - (N > 1 ? 1 : 0);
  reduce_masked(local, global, OpMaxAssign(), (N == 1 || myN < (N-1)));
  report("Scalar reduction (compute max)", global==expected);

  //
  // array reduction
  //

  float localArray[3];
  float globalArray[3];
  float expectResult = 1.0;
  localArray[0] = localArray[1] = localArray[2] = local;
  for (i=1; i < Ippl::getNodes(); i++)
    expectResult += (float)(i + 1);
  reduce(localArray, localArray + 3, globalArray, OpAddAssign());
  bool correct = true;
  for (i=0; i<3; i++) {
    if (globalArray[i] != expectResult) correct = false;
  }
  report("Array reduction (compute sum)", correct);

  //
  // 2D array reduction
  //

  float localArray2[3][3];
  float globalArray2[3][3];
  float expectResult2 = 1.0;
  localArray2[0][0] = localArray2[1][0] = localArray2[2][0] = local;
  localArray2[0][1] = localArray2[1][1] = localArray2[2][1] = local;
  localArray2[0][2] = localArray2[1][2] = localArray2[2][2] = local;
  for (i=1; i < Ippl::getNodes(); i++)
    expectResult2 += (float)(i + 1);
  reduce(&(localArray2[0][0]), &(localArray2[0][0]) + 9,
         &(globalArray2[0][0]), OpAddAssign());
  correct = true;
  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) {
      if (globalArray2[i][j] != expectResult2) correct = false;
    }
  }
  report("2D Array reduction (compute sum)", correct);
//   for (int pe=0; pe < Ippl::getNodes(); pe++) {
//     if (pe == myN) {
//       cout << "(((((((((((((((((( PE " << myN << " ))))))))))))))))))" << endl;
//       for (i=0; i<3; i++) {
// 	for (j=0; j<3; j++) {
// 	  cout << "localArray2[" << i << "][" << j << "] = " 
// 	       << localArray2[i][j]
// 	       << " ; globalArray2[" << i << "][" << j << "] = " 
// 	       << globalArray2[i][j] << endl;
// 	}
//       }
//     }
//     Ippl::Comm->barrier();
//   }

  return 0;
}

/***************************************************************************
 * $RCSfile: reduce.cpp.org,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:38 $
 * IPPL_VERSION_ID: $Id: reduce.cpp.org,v 1.1.1.1 2003/01/23 07:40:38 adelmann Exp $ 
 ***************************************************************************/
