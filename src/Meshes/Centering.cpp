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
#include "Meshes/Centering.h"

// Names for the centering classes Cell and Vert, which are being kept around
// for backwards compatibility and possibly use with noncartesian meshes:
const char* Cell::CenteringName = "Cell";
const char* Vert::CenteringName = "Vert";
const char* Edge::CenteringName = "Edge";

const char* Centering::CenteringEnum_Names[] = {"CELL  ","VERTEX","EDGE  "};

//CC chokes static void Cell::print_Centerings(ostream& out)
void Cell::print_Centerings(std::ostream& out)
{
  out << Cell::CenteringName << std::endl;
}
//CC chokes static void Vert::print_Centerings(ostream& out)
void Vert::print_Centerings(std::ostream& out)
{
  out << Vert::CenteringName << std::endl;
}

void Edge::print_Centerings(std::ostream& out)
{
    out << Edge::CenteringName << std::endl;
}

/***************************************************************************
 * $RCSfile: Centering.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: Centering.cpp,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $
 ***************************************************************************/