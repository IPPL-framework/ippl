// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef USER_LIST_H
#define USER_LIST_H

/***********************************************************************
 * 
 * UserList is a base class for classes which need to maintain a list
 * of users of that particular instance.  It provides 'checkinUser' and
 * 'checkoutUser' methods which are called by the users of an instance
 * of this class.  UserList is templated on a 'key' (used to uniquely
 * distinguish a particular user, provided in the checkin call) and on
 * the type of the user.  It stores a list of pointers to the users.
 *
 * When an instance of this class is deleted, it informs all users of
 * its demise via a call to 'notifyUserOfDelete', specifying the unique
 * ID of the UserList.  This ID is generated when the UserList instance
 * is created, and is returned to the user when they check in.
 * 
 ***********************************************************************/

// include files
#include "Utility/vmap.h"
#include "Utility/Unique.h"
#include "Utility/User.h"


// class definition
class UserList {

public:
  // useful typedefs
  typedef User::ID_t                 Key;
  typedef vmap<Key, User *>          UserList_t;
  typedef UserList_t::iterator       iterator_user;
  // typedef UserList_t::const_iterator const_iterator_user;
  typedef UserList_t::size_type      size_type_user;
  typedef User::ID_t                 ID_t;

public:
  // constructor: just get unique ID for this object
  UserList();

  // destructor: inform all users of our untimely demise
  virtual ~UserList();

  //
  // informative methods
  //

  // return the number of users
  size_type_user getNumUsers() const;

  // return the ID of this userlist
  ID_t getUserListID() const;

  // do we have a user with the given key?  Return true if we do.
  bool haveUser(Key key) const;

  // return the user with the given key
  User& getUser(Key key);

  // return begin/end iterators for the users
  iterator_user       begin_user();
  iterator_user       end_user();
  // const_iterator_user begin_user() const;
  // const_iterator_user end_user()   const;

  //
  // virtual checkin/checkout functions
  //

  // Check in a new user with the given key.  Check to make sure we do
  // not already have that key.  Return our UserList ID number.  Derived
  // classes can override this, but should call this base class version
  // to actually store the user.
  virtual ID_t checkinUser(User& user);

  // Check out a user, by removing them from our list.  Check to make sure
  // a user with the given key is in our list.  If the second argument is
  // true (by default it is not), the user is notified that they are
  // being checked out (this may be useful if the person calling checkoutUser
  // is not the user being checked out).
  virtual void checkoutUser(Key key, bool informuser = false);

  // also checkout a user, by specifying the user to checkout instead of
  // the user key.  The key is obtained from the user in this case.
  virtual void checkoutUser(const User& user, bool informuser = false);

private:
  // the list of users
  UserList_t userlist;

  // our unique ID
  ID_t userlistID;
};

#endif // USER_LIST_H

/***************************************************************************
 * $RCSfile: UserList.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:34 $
 * IPPL_VERSION_ID: $Id: UserList.h,v 1.1.1.1 2003/01/23 07:40:34 adelmann Exp $ 
 ***************************************************************************/
