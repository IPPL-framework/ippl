// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef USER_H
#define USER_H

/***********************************************************************
 * 
 * User is a base class for classes which need to be made part of a
 * UserList.  It provides a user ID value (obtained from the Unique
 * class) and a virtual function which is called when the UserList
 * needs the User to not be part of it anymore.
 *
 ***********************************************************************/

// include files
#include "Utility/Unique.h"

// forward declarations
class UserList;


// class definition
class User {

public:
  // useful typedefs
  typedef Unique::type ID_t;

public:
  // constructor
  User();

  // destructor ... nothing to do, subclasses are responsible for
  // doing checkin's and checkout's
  virtual ~User();

  // return the ID of this user
  ID_t get_Id() const { return Id; }

  //
  // virtual functions for User
  //

  // Tell the subclass that the FieldLayout is being deleted, so
  // don't use it anymore
  virtual void notifyUserOfDelete(UserList *) = 0;

protected: 
  // Each User has a unique Id.
  ID_t Id;

};

#endif // USER_H

/***************************************************************************
 * $RCSfile: User.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: User.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
