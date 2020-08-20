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
#include "Utility/RNGXDiv.h"

// initialize static variables for RNGXDiv
const double RNGXDiv::SQR_RANMAX     = 16777216.0; // 2^24
const double RNGXDiv::RANDOM_MAX     = 16777216.0*16777216.0; // 2^48
const double RNGXDiv::INV_SQR_RANMAX = 1.0/RNGXDiv::SQR_RANMAX;
const double RNGXDiv::INV_RANMAX     = 1.0/RNGXDiv::RANDOM_MAX;
const double RNGXDiv::SeedMultUpper  = 13008944.0;
const double RNGXDiv::SeedMultLower  = 170125.0;
const double RNGXDiv::RandMultUpper  = 1136868.0;
const double RNGXDiv::RandMultLower  = 6328637.0;
const double RNGXDiv::FirstSeed      = 1953125.0*9765625.0; // 5^19


/***************************************************************************
 * $RCSfile: RNGXDiv.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: RNGXDiv.cpp,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
