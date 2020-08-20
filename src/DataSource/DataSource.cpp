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
#include "DataSource/DataSource.h"
#include "DataSource/DataSourceObject.h"
#include "DataSource/DataConnect.h"
#include "DataSource/DataConnectCreator.h"
#include "Utility/IpplInfo.h"


////////////////////////////////////////////////////////////////////////////
// constructor
DataSource::DataSource() {
  // nothing to do
}


////////////////////////////////////////////////////////////////////////////
// destructor
DataSource::~DataSource() {
  //Inform dbgmsg("DataSource::~DataSource", INFORM_ALL_NODES);
  //dbgmsg << "Calling disconnect() from DataSource destructor." << endl;

  // disconnect from all existing connections, if we still have any
  disconnect();
}


////////////////////////////////////////////////////////////////////////////
// find the first DataSourceObject which is connected to the given DataConnect,
// and return it; otherwise, return 0
DataSourceObject *DataSource::findDataSourceObject(DataConnect *dc) const {
  container_t::const_iterator a = ConnectionList.begin();
  for ( ; a != ConnectionList.end(); ++a)
    if (dc == (*a)->getConnection())
      return (*a);

  // if we're here, we're not connected
  return 0;
}


////////////////////////////////////////////////////////////////////////////
// Are we connected to the given DataConnect?  If not specified, just return
// whether we're connected to anything.
bool DataSource::connected(DataConnect *dataconn) const {
  if (dataconn == 0)
    return ( ! ConnectionList.empty() );
  else
    return (findDataSourceObject(dataconn) != 0);
}


////////////////////////////////////////////////////////////////////////////
// Register an object as something that can be a source of data.  Return
// success.  Arguments = name of item, connection to use, and type of
// connection (INPUT, OUTPUT, or BOTH).  If the connection
// has not been created yet (e.g., it is NULL), create a new default
// connection (or use the existing default one).  Return the connection.
DataConnect *DataSource::connect(const char *nm,DataConnect *dataconn,int tm) {
  //Inform dbgmsg("DataSource::connect", INFORM_ALL_NODES);
  //dbgmsg << "Connecting '" << nm << "'" << endl;
  //if (dataconn != 0)
  //  dbgmsg << " to '" << dataconn->name() << "'";
  //dbgmsg << endl;

  // figure out which connection to use ... if none given, we must get a
  // new one
  if (dataconn == 0) {
    //dbgmsg << "Creating new connection ..." << endl;
    dataconn = DataConnectCreator::create();
  }

  // make sure we're not already connected to this object
  if (connected(dataconn)) {
    ERRORMSG("Cannot connect '" << nm << "' to the same agency twice." <<endl);
    ERRORMSG("dataconn = " << (void *)dataconn << endl);
    return 0;
  }

  // set the transfer method properly, if we are requested to use the default
  if (tm == DataSource::DEFAULT)
    tm = dataconn->getDefaultTransferMethod();

  // OK, connect away ... create new DataSourceObject, which establishes
  // the connection.  If we fail to connect, delete the created Obj
  DataSourceObject *dso = createDataSourceObject(nm, dataconn, tm);
  if (dso != 0 && !connect(dso)) {
    delete dso;
    return 0;
  }

  // if we're here, everything was ok ... return the connection we used
  return dataconn;
}


////////////////////////////////////////////////////////////////////////////
// Register the given DataSourceObject directly.  This is simpler than
// the above version of connect, since the DataSourceObject has already
// been created.  It can be used to register most any type of connection,
// even one using DataConnect objects that are not part of IPPL itself.
// Return success.
bool DataSource::connect(DataSourceObject *dso) {

  // make sure the connection is OK
  if (dso != 0 && dso->connected() && dso->getSource() == this) {
    ConnectionList.push_back(dso);
    dso->getConnection()->checkin(this);
    return true;
  }

  // if we're here, there was a problem
  return false;
}


////////////////////////////////////////////////////////////////////////////
// Disconnect an object from the given receiver.  Return success.
bool DataSource::disconnect(DataConnect *dataconn) {
  //Inform dbgmsg("DataSource::disconnect", INFORM_ALL_NODES);
  //dbgmsg << "Disconnecting source from connection(s) '";
  //dbgmsg << (dataconn != 0 ? dataconn->name() : "(all)")<< "' ..." << endl;

  container_t::iterator a = ConnectionList.begin();
  for ( ; a != ConnectionList.end(); ++a) {
    DataConnect *dc = (*a)->getConnection();
    if (dataconn == 0 || dataconn == dc) {
      //dbgmsg << "Calling checkout from the DataSource ..." << endl;
      dc->checkout(this, false);
      //dbgmsg << "Deleting DSO '" << (*a)->name() << "' ..." << endl;
      delete (*a);
      if (dataconn == dc) {
	ConnectionList.erase(a);
	return true;
      }
    }
  }

  // if we've removed all, we can erase all
  if (dataconn == 0 && ConnectionList.size() > 0) {
    //dbgmsg<<"Erasing all " << ConnectionList.size() << " DSO's ..." << endl;
    ConnectionList.erase(ConnectionList.begin(), ConnectionList.end());
  }

  return (dataconn == 0);
}


////////////////////////////////////////////////////////////////////////////
// Update the object, that is, make sure the receiver of the data has a
// current and consistent snapshot of the current state.  Return success.
bool DataSource::updateConnection(DataConnect *dataconn) {
  container_t::iterator a = ConnectionList.begin();
  for ( ; a != ConnectionList.end(); ++a) {
    if (dataconn == 0)
      (*a)->update();
    else if (dataconn == (*a)->getConnection())
      return (*a)->update();
  }

  // if we're here, we were successful only if we tried to disconnect all
  return (dataconn == 0);
}


////////////////////////////////////////////////////////////////////////////
// Indicate to the receiver that we're allowing them time to manipulate the
// data (e.g., for a viz program, to rotate it, change representation, etc.)
// This should only return when the manipulation is done.  For some cases,
// this will be a no-op.
void DataSource::interact(DataConnect *dataconn) {
  container_t::iterator a = ConnectionList.begin();
  for ( ; a != ConnectionList.end(); ++a) {
    if (dataconn == 0 || dataconn == (*a)->getConnection())
      (*a)->interact();
  }
}


////////////////////////////////////////////////////////////////////////////
// Pass on a string to the connection, most likely to give it a
// command to do some action.  Similar to the above interact, except
// the request for interaction involves the given string
void DataSource::interact(const char *str, DataConnect *dataconn) {
  container_t::iterator a = ConnectionList.begin();
  for ( ; a != ConnectionList.end(); ++a) {
    if (dataconn == 0 || dataconn == (*a)->getConnection())
      (*a)->interact(str);
  }
}


/***************************************************************************
 * $RCSfile: DataSource.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:25 $
 * IPPL_VERSION_ID: $Id: DataSource.cpp,v 1.1.1.1 2003/01/23 07:40:25 adelmann Exp $ 
 ***************************************************************************/
