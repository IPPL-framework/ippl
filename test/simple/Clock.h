// -*- C++ -*-
// ACL:license
// ----------------------------------------------------------------------
// This software and ancillary information (herein called "SOFTWARE")
// called IPPL (Parallel Object-Oriented Methods and Applications) is
// without charge, provided that this Notice and any statement of
// authorship are reproduced on all copies.  Neither the Government nor
// the University makes any warranty, express or implied, or assumes any
// liability or responsibility for the use of this SOFTWARE.
// 
// If SOFTWARE is modified to produce derivative works, such modified
// SOFTWARE should be clearly marked, so as not to confuse it with the
// version available from LANL.
// 
// For more information about IPPL, send e-mail to ippl@acl.lanl.gov,
// or visit the IPPL web page at http://www.acl.lanl.gov/ippl/.
// ----------------------------------------------------------------------
// ACL:license

#ifndef IPPL_UTILITIES_CLOCK_H
#define IPPL_UTILITIES_CLOCK_H

#include <time.h>

//namespace Ippl {

//-----------------------------------------------------------------------
// Clock provides a running timer, utilizing high-speed SGI timers if 
// available.
//-----------------------------------------------------------------------

class Clock {
public:

  //---------------------------------------------------------------------
  // Set a static const that tells whether or not this class is utilizing
  // high-speed timers.
  
#if defined(CLOCK_SGI_CYCLE)
  static const bool highSpeed = true;
#else
  static const bool highSpeed = false;
#endif  

  //---------------------------------------------------------------------
  // Return the current value of the timer [sec].
  
  inline static double value()
  {
#if defined(CLOCK_SGI_CYCLE)
  timespec ts;
  clock_gettime(CLOCK_SGI_CYCLE, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
#else
  return double(clock()) / CLOCKS_PER_SEC;
#endif
  }
};

//}  namespace Ippl


//////////////////////////////////////////////////////////////////////

#endif // IPPL_UTILITIES_CLOCK_H

// ACL:rcsinfo
// ----------------------------------------------------------------------
// $RCSfile: Clock.h,v $   $Author: adelmann $
// $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:38 $
// ----------------------------------------------------------------------
// ACL:rcsinfo

/***************************************************************************
 * $RCSfile: addheaderfooter,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:17 $
 * IPPL_VERSION_ID: $Id: addheaderfooter,v 1.1.1.1 2003/01/23 07:40:17 adelmann Exp $ 
 ***************************************************************************/

