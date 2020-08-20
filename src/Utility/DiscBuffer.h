// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef DISC_BUFFER_H
#define DISC_BUFFER_H

/***************************************************************************
 * DiscBuffer is a simple utility class that maintains a byte buffer
 * for use in I/O operations, one that will grow on demand and will
 * be reused until the program quits.  Users should just use the
 * static methods to grow and access the buffer.  An instance of this class
 * is created when a program starts and is deleted when a program is
 * deleted. Users can, but generally should not, create an instances
 * of this class themselves.
 ***************************************************************************/

// include files

#include "Utility/PAssert.h"


class DiscBuffer
{
public:
  // The default constructor does nothing, this class is mainly an
  // interface to the static data.

  DiscBuffer();

  // The destructor will delete the buffer storage if it has been
  // previously created.

  ~DiscBuffer();

  //
  // Accessor methods.
  //

  // Return the current size of the buffer, in bytes.

  static long size()
  {
    return size_s;
  }

  // Return the current buffer pointer, as a void *.

  static void *buffer()
  {
    return static_cast<void *>(buffer_s);
  }

  //
  // Modifier methods
  //

  // Make sure the current buffer is at least as large as the given
  // size.  If it is not, reallocate (but do not copy).  Return the
  // buffer pointer.

  static void *resize(long sz);

  // Grow the buffer to add in extra storage of the given amount.
  // Return the buffer pointer.

  static void *grow(long amt)
  {
    PAssert_GE(amt, 0);
    return DiscBuffer::resize(DiscBuffer::size() + amt);
  }

  // Some static variables used for statistics, these are really
  // hacks so don't count on them.

  static double readtime;
  static double writetime;
  static long readbytes;
  static long writebytes;

private:
  // Static storage for the size of the buffer.

  static long size_s;

  // Static storage for the buffer itself.

  static char *buffer_s;

};


#endif // DISC_BUFFER_H

/***************************************************************************
 * $RCSfile: DiscBuffer.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: DiscBuffer.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
