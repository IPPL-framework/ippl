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
#include "DataSource/FileFieldDataSource.h"
#include "DataSource/FileDataConnect.h"
#include "Utility/DiscField.h"
#include "Message/Communicate.h"
#include "Utility/IpplInfo.h"


////////////////////////////////////////////////////////////////////////////
// constructor: the name, the connection, the transfer method,
// the field to connect, and the parent node.
template<class T, unsigned Dim, class M, class C>
FileFieldDataSource<T,Dim,M,C>::FileFieldDataSource(const char *nm,
						    DataConnect *dc,
						    int tm,
						    Field<T,Dim,M,C>& F)
  : DataSourceObject(nm,&F,dc,tm), DF(0), myField(F), FieldID(0), counter(0) {

  
  

  std::string filestring = "file";
  if (std::string(dc->DSID()) != filestring) {
    ERRORMSG("Illegal DataConnect object for FILE Data Object." << endl);
    Connection = 0;
  } else if (tm != DataSource::OUTPUT && tm != DataSource::INPUT) {
    ERRORMSG("FILE connections may only be of type INPUT or OUTPUT." << endl);
    Connection = 0;
  } else {
    // find which DiscField to use ... first look for one in the DataConnect
    FieldID = dc->size();
    FileDataConnect *fdc = (FileDataConnect *)dc;
    if (dc->size() == 0) {
      // this is the first field in the FileDataConnect, so make a new
      // DiscField
      if (TransferMethod == DataSource::OUTPUT)
	DF = new DiscField<Dim>(nm, dc->name(), fdc->getNumObjects(),
				fdc->getTypeString());
      else
	DF = new DiscField<Dim>(nm, dc->name());
    }
    else {
      // use the DiscField in the first DataSource
      DataSourceObject *dso = (*(dc->begin()))->findDataSourceObject(dc);
      if (dso == 0) {
	ERRORMSG("Could not find proper DiscField while connecting " << nm << endl);
      }
      else {
	DF = ((FileFieldDataSource<T,Dim,M,C> *)dso)->DF;
      }
    }
  }
}


////////////////////////////////////////////////////////////////////////////
// destructor
template<class T, unsigned Dim, class M, class C>
FileFieldDataSource<T,Dim,M,C>::~FileFieldDataSource() {
  
  

  if (DF != 0 && getConnection()->size() == 0)
    delete DF;
}


////////////////////////////////////////////////////////////////////////////
// Update the object, that is, make sure the receiver of the data has a
// current and consistent snapshot of the current stat dc->name()e.  Return success.
template<class T, unsigned Dim, class M, class C>
bool FileFieldDataSource<T,Dim,M,C>::update() {
  
  

  if (TransferMethod == DataSource::OUTPUT)
    DF->write(myField, FieldID);
  else if (TransferMethod == DataSource::INPUT)
    DF->read(myField, FieldID, counter++);
  return true;
}


////////////////////////////////////////////////////////////////////////////
// Indicate to the receiver that we're allowing the FieldDisc time to
// manipulate the data (e.g., for a viz program, to rotate it, change
// representation, etc.).
// This should only return when the manipulation is done.
template<class T, unsigned Dim, class M, class C>
void FileFieldDataSource<T,Dim,M,C>::interact(const char *) {}


/***************************************************************************
 * $RCSfile: FileFieldDataSource.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:25 $
 * IPPL_VERSION_ID: $Id: FileFieldDataSource.cpp,v 1.1.1.1 2003/01/23 07:40:25 adelmann Exp $ 
 ***************************************************************************/
