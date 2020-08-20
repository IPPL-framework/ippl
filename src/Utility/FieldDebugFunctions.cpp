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

#include "Utility/FieldDebugFunctions.h"

// forward class declarations
class Inform;

//----------------------------------------------------------------------
// Set up the I/O "stream" (IPPL Inform object) format where the output goes.
// This "sticks around" once set until re-set.
//----------------------------------------------------------------------
Inform* FldDbgInform;           // Pointer to active Inform object
bool FldDbgInformIsSet = false; // Flags whether pointer is set
void 
setInform(Inform& inform) {
  
  FldDbgInform = &inform;
  FldDbgInformIsSet = true;
}
//----------------------------------------------------------------------
// Variables to store print-formatting parameters. These "stick
// around" once set (set-functions provided) until re-set. Needed esp.
// in debugger (main intent of all these print functions) because of
// limitations with default arguments--fcns with extra format-parameter
// arguments there.
//----------------------------------------------------------------------
int elementsPerLine = 7;
int digitsPastDecimal = 3;
int widthOfElements = 0;
void 
setFormat(int ElementsPerLine, int DigitsPastDecimal, 
	       int WidthOfElements) {
  
  elementsPerLine = ElementsPerLine;
  digitsPastDecimal = DigitsPastDecimal;
  widthOfElements = WidthOfElements;
}

/***************************************************************************
 * $RCSfile: FieldDebugFunctions.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: FieldDebugFunctions.cpp,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
