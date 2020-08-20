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

#include "Utility/DiscBuffer.h"
#include "Utility/PAssert.h"
#include <cstdlib>


////////////////////////////////////////////////////////////////////////////
// Method definitions for DiscBuffer
////////////////////////////////////////////////////////////////////////////

// Static storage

long  DiscBuffer::size_s   = 0;
char *DiscBuffer::buffer_s = 0;

// Some static variables used for statistics, these are really
// hacks so don't count on them.

double DiscBuffer::readtime   = 0.0;
double DiscBuffer::writetime  = 0.0;
long   DiscBuffer::readbytes  = 0;
long   DiscBuffer::writebytes = 0;


////////////////////////////////////////////////////////////////////////////
// The default constructor does nothing, this class is mainly an
// interface to the static data.

DiscBuffer::DiscBuffer()
{
}


////////////////////////////////////////////////////////////////////////////
// The destructor will delete the buffer storage if it has been
// previously created.

DiscBuffer::~DiscBuffer()
{
  if (buffer_s != 0)
    delete [] buffer_s;

  size_s = 0;
  buffer_s = 0;
}


////////////////////////////////////////////////////////////////////////////
// Make sure the current buffer is at least as large as the given
// size.  If it is not, reallocate (but do not copy).  In either case,
// return the buffer pointer.

void *DiscBuffer::resize(long sz)
{
  PAssert_GE(sz, 0);

  if (sz > size_s)
    {
      // Reset our existing size
      size_s = sz;

      // Free the old buffer, if necessary, and create a new one
      if (buffer_s != 0)
	{
	  delete [] buffer_s;
	  buffer_s = 0;
	}
      buffer_s = new char[size_s];
      PAssert(buffer_s);
    }

  return buffer();
}


////////////////////////////////////////////////////////////////////////////
// Create a single global instance of a DiscBuffer so that something
// will delete the storage when it is done.

DiscBuffer ipplGlobalDiscBuffer_g;


/***************************************************************************
 * $RCSfile: DiscBuffer.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: DiscBuffer.cpp,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $
 ***************************************************************************/
