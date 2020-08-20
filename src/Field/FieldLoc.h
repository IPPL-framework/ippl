// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef FIELD_LOC_H
#define FIELD_LOC_H

/***************************************************************************
 *
 * This simple class stores information about where in a Field
 * iterator the user wants to refer to.  You can ask a Field::iterator
 * for its current FieldLoc, and set that iterator to point to the
 * location of a given FieldLoc.  It stores the current LField, and the
 * global index of the current point.
 *
 ***************************************************************************/

// include files
#include "Utility/Vec.h"


template <unsigned Dim>
class FieldLoc {

public:
  // constructor: provide the global index of the point, and an LField
  // 'index', which refers to the Nth LField in the relevant FieldLayout's
  // list of local vnodes.
  FieldLoc(const vec<int,Dim>& v, int f) : loc(v), LFIndex(f) { }

  // copy constructor
  FieldLoc(const FieldLoc<Dim>& fl) : loc(fl.loc), LFIndex(fl.LFIndex) { }
  
  // default constructor
  FieldLoc() : LFIndex(-1) { }

  // destructor
  ~FieldLoc() { }

  // equals operator
  FieldLoc<Dim>& operator=(const FieldLoc<Dim>& fl) {
    if (&fl != this) {
      loc = fl.loc;
      LFIndex = fl.LFIndex;
    }
    return *this;
  }

  // query for the current global location
  int &operator[](int n) { return loc[n]; }
  int operator[](int n) const { return loc[n]; }

  // query for the current LField index.  If -1, this does not point to any
  // LField.
  int getIndex() const { return LFIndex; }

  // change the index of the LField
  void setIndex(int newval) { LFIndex = newval; }

  // return an NDIndex with the point as one-element Index objects
  NDIndex<Dim> getDomain() const {
    NDIndex<Dim> retval;
    for (unsigned int d=0; d < Dim; ++d)
      retval[d] = Index(loc[d], loc[d]);
    return retval;
  }

private:
  // the global index for this point
  vec<int,Dim> loc;

  // the index of the LField in FieldLayout's list of local LField's
  int LFIndex;
};


#endif // FIELD_LOC_H

/***************************************************************************
 * $RCSfile: FieldLoc.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:26 $
 * IPPL_VERSION_ID: $Id: FieldLoc.h,v 1.1.1.1 2003/01/23 07:40:26 adelmann Exp $ 
 ***************************************************************************/
