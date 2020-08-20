// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 ***************************************************************************/

#ifndef FIELD_LAYOUT_USER_H
#define FIELD_LAYOUT_USER_H

/***********************************************************************
 * 
 * FieldLayoutUser is a base class for all classes which need to use
 * a FieldLayout - it is derived from User, which provides a virtual
 * function 'notifyUserOfDelete' which is called when the FieldLayout
 * is deleted, and the virtual function 'Repartition' which is called
 * when a Field needs to be redistributed between processors.
 *
 ***********************************************************************/

// include files
#include "Utility/User.h"
#include "Utility/UserList.h"


// class definition
class FieldLayoutUser : public User {

public:
  // constructor - the base class selects a unique ID value
  FieldLayoutUser() {};

  // destructor, nothing to do here
  virtual ~FieldLayoutUser() {};

  //
  // virtual functions for FieldLayoutUser's
  //

  // Repartition onto a new layout
  virtual void Repartition(UserList *) = 0;
};

#endif // FIELD_LAYOUT_USER_H

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
