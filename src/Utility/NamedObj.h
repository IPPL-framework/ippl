// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef NAMED_OBJ_H
#define NAMED_OBJ_H

/***************************************************************************
 *
 * NamedObj is a simple base class for objects which desire to have a name;
 * it provides one method 'name()' which returns the name of the object.
 *
 ***************************************************************************/

// include files
#include <string> 


class NamedObj {

public:
  NamedObj(const char *nm = 0) {
    if (nm != 0)
      MyName = nm;
  }
  virtual ~NamedObj() { }

  // return the current name
  const char *name() const { return MyName.c_str(); }

  // change the name
  const char *setName(const char *nm = 0) {
    if (nm != 0)
      MyName = nm;
    else
      MyName = "";
    return MyName.c_str();
  }

private:
  std::string MyName;
};

#endif // NAMED_OBJ_H

/***************************************************************************
 * $RCSfile: NamedObj.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: NamedObj.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
