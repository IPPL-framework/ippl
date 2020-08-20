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

//---------------------------------------------------------------------------//
// Assert.cpp
// Geoffrey Furnish
// Fri Jul 25 08:41:38 1997
//---------------------------------------------------------------------------//
// @> Helper functions for the Assert facility.
//---------------------------------------------------------------------------//

// include files
#include "Utility/PAssert.h"

#include <iostream>
using namespace std;
#include <cstring>
#include <cstdio>
#include <cstdlib>

//---------------------------------------------------------------------------//

assertion::assertion( const char *cond, const char *file, int line ):
    std::runtime_error(cond)
{
    msg = new char[ strlen(cond) + strlen(file) + 500 ];
    sprintf( msg, "Assertion: %s, failed in %s, line %8d.",
	     cond, file, line );
}

assertion::assertion( const char *m ):
    std::runtime_error(m)
{
    msg = new char[ strlen(m)+1 ];
    strcpy( msg, m );
}

assertion::assertion( const assertion& a ):
    std::runtime_error(a.msg)
{
    msg = new char[ strlen(a.msg)+1 ];
    strcpy( msg, a.msg );
}

assertion& assertion::operator=( const assertion& a )
{
    msg = new char[ strlen(a.msg)+1 ];
    strcpy( msg, a.msg );
    return *this;
}

//---------------------------------------------------------------------------//
// Function to perform the task of actually throwing an assertion.
//---------------------------------------------------------------------------//

void toss_cookies( const char *cond, const char *file, int line )
{
    std::string what = "Assertion '" + std::string(cond) + "' failed. \n";
    what += "in \n";
    what += std::string(file) + ", line  " + std::to_string(line);

    throw std::runtime_error(what);
}

//---------------------------------------------------------------------------//
// Function to perform the task of actually throwing an isistion.
//---------------------------------------------------------------------------//

void insist( const char *cond, const char *msg, const char *file, int line )
{
    char* fullmsg = new char[ strlen(cond) + strlen(msg) + strlen(file) + 500 ];
    sprintf( fullmsg, "%s\nAssertion '%s' failed in \n%s on line %8d.",
	     msg, cond, file, line );

    throw assertion( fullmsg );

}

//---------------------------------------------------------------------------//
//                              end of Assert.cpp
//---------------------------------------------------------------------------//

/***************************************************************************
 * $RCSfile: PAssert.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: PAssert.cpp,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $
 ***************************************************************************/
