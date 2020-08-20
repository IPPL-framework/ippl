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
#include "Utility/UserList.h"
#include "Utility/PAssert.h"



////////////////////////////////////////////////////////////////////////////
// constructor: just get unique ID for this object
UserList::UserList() : userlistID(Unique::get()) { }


////////////////////////////////////////////////////////////////////////////
// destructor: inform all users of our untimely demise
UserList::~UserList() {
  for (iterator_user a=begin_user(); a != end_user(); ++a)
    (*a).second->notifyUserOfDelete(this);
}


////////////////////////////////////////////////////////////////////////////
  // return the number of users
UserList::size_type_user UserList::getNumUsers() const {
  return userlist.size();
}


////////////////////////////////////////////////////////////////////////////
// return the ID of this userlist
UserList::ID_t UserList::getUserListID() const {
  return userlistID;
}


////////////////////////////////////////////////////////////////////////////
// do we have a user with the given key?  Return true if we do.
bool UserList::haveUser(Key key) const {
  return (userlist.count(key) == 1);
}


////////////////////////////////////////////////////////////////////////////
// return the user with the given key
User& UserList::getUser(Key key) {
  iterator_user user = userlist.find(key);
  PInsist(user != end_user(),"Failed to find key in UserList::getUser!!");
  return *((*user).second);
}


////////////////////////////////////////////////////////////////////////////
// return begin/end iterators for the users
UserList::iterator_user UserList::begin_user() {
  return userlist.begin();
}


////////////////////////////////////////////////////////////////////////////
// return begin/end iterators for the users
UserList::iterator_user UserList::end_user() {
  return userlist.end();
}


/*
////////////////////////////////////////////////////////////////////////////
// return begin/end iterators for the users
UserList::const_iterator_user UserList::begin_user() const {
  return userlist.begin();
}


////////////////////////////////////////////////////////////////////////////
// return begin/end iterators for the users
UserList::const_iterator_user UserList::end_user() const {
  return userlist.end();
}
*/

////////////////////////////////////////////////////////////////////////////
// Check in a new user with the given key.  Check to make sure we do
// not already have that key.  Return our UserList ID number.  Derived
// classes can override this, but should call this base class version
// to actually store the user.
UserList::ID_t UserList::checkinUser(User& user) {
  
  Key key = user.get_Id();
  if ( ! haveUser(key) )
    userlist[key] = &user;
  return userlistID;
}


////////////////////////////////////////////////////////////////////////////
// Check out a user, by removing them from our list.  Check to make sure
// a user with the given key is in our list.  If the second argument is
// true (by default it is not), the user is notified that they are
// being checked out (this may be useful if the person calling checkoutUser
// is not the user being checked out).
void UserList::checkoutUser(Key key, bool informuser) {
  
  iterator_user user = userlist.find(key);
  if (user != end_user()) {
    if (informuser)
      (*user).second->notifyUserOfDelete(this);
    userlist.erase(user);
  }
}


////////////////////////////////////////////////////////////////////////////
// also checkout a user, by specifying the user to checkout instead of
// the user key.  The key is obtained from the user in this case.
void UserList::checkoutUser(const User& user, bool informuser) {
  
  checkoutUser(user.get_Id(), informuser);
}


/***************************************************************************
 * $RCSfile: UserList.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:34 $
 * IPPL_VERSION_ID: $Id: UserList.cpp,v 1.1.1.1 2003/01/23 07:40:34 adelmann Exp $ 
 ***************************************************************************/
