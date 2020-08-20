// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef FILE_DATA_CONNECT_H
#define FILE_DATA_CONNECT_H

/***********************************************************************
 * 
 * FileDataConnect represents a single connection to a parallel datafile
 * object (such as DiskField)
 *
 ***********************************************************************/

// include files
#include "DataSource/DataConnect.h"


class FileDataConnect : public DataConnect {

public:
  // constructor: file name, items in the file, typestring, nodes
  FileDataConnect(const char *nm, unsigned int numobjs = 1, const char *ts = 0,
		  int n = 0)
    : DataConnect(nm, "file", DataSource::OUTPUT, n), NumObjects(numobjs) {
    if (ts == 0)
      TypeString = "unknown";
    else
      TypeString = ts;
  }

  // destructor: shut down the connection
  virtual ~FileDataConnect() {}

  //
  // methods specific to this type of DataConnect
  //

  // return number of objects that this file should hold
  unsigned int getNumObjects() const { return NumObjects; }

  // return the type string for this file
  const char* getTypeString() const { return TypeString.c_str(); }

  //
  // DataConnect virtual methods
  //

  // are we currently connected to a receiver?
  virtual bool connected() const { return true; }

private:
  // number of objects in the file, and a type string
  unsigned int NumObjects;
  std::string TypeString;
};

#endif // FILE_DATA_CONNECT_H

/***************************************************************************
 * $RCSfile: FileDataConnect.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:25 $
 * IPPL_VERSION_ID: $Id: FileDataConnect.h,v 1.1.1.1 2003/01/23 07:40:25 adelmann Exp $ 
 ***************************************************************************/
