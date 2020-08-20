// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef FILE_FIELD_DATA_SOURCE_H
#define FILE_FIELD_DATA_SOURCE_H

/***********************************************************************
 * 
 * class FileFieldDataSource
 *
 * FileFieldDataSource is a specific version of DataSourceObject which takes
 * the data for a given Field and writes it to a file using a DiscField
 * object.
 *
 ***********************************************************************/

// include files
#include "DataSource/DataSourceObject.h"
#include "Field/Field.h"
#include "Utility/DiscField.h"


template<class T, unsigned Dim, class M, class C>
class FileFieldDataSource : public DataSourceObject {

public:

  // constructor: the name, the connection, the transfer method,
  // the field to connect, and the parent node.
  FileFieldDataSource(const char *, DataConnect *, int, Field<T,Dim,M,C>&);

  // destructor
  virtual ~FileFieldDataSource();

  //
  // virtual function interface.
  //

  // Update the object, that is, make sure the receiver of the data has a
  // current and consistent snapshot of the current state.  Return success.
  virtual bool update();

  // Indicate to the receiver that we're allowing them time to manipulate the
  // data (e.g., for a viz program, to rotate it, change representation, etc.)
  // This should only return when the manipulation is done.
  virtual void interact(const char * = 0);

private:
  // the DiscField object, which read/writes the data
  DiscField<Dim> *DF;

  // the Field to read into (or write from)
  Field<T,Dim,M,C> &myField;

  // which field are we in the file?
  int FieldID;

  // the number of frames we have read or written (i.e. or current record)
  int counter;
};

#include "DataSource/FileFieldDataSource.hpp"

#endif // FILE_FIELD_DATA_SOURCE_H

/***************************************************************************
 * $RCSfile: FileFieldDataSource.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:25 $
 * IPPL_VERSION_ID: $Id: FileFieldDataSource.h,v 1.1.1.1 2003/01/23 07:40:25 adelmann Exp $ 
 ***************************************************************************/
