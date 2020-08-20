// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef UNIQUE_H
#define UNIQUE_H

/*

  This class is used to generate a series of unique id numbers.

  Each time you call Unique::get() you get a unique object of type
  Unique::type.

  Typically Unique::type will be an integer.

  A proper parallel implementation of this object will ensure that 
  the returned id's are unique across the whole machine.

 */

class Unique
{
public:
  typedef unsigned int type;		// An int is simple and quick for sorting.

  static type get()		// Get the next one.
  {				// 
    return Last++;		// return it.
  }

private:
  Unique();			// Don't actually build any of these.

  static type Last;		// The last one returned.
};

#endif // UNIQUE_H

/***************************************************************************
 * $RCSfile: Unique.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: Unique.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
