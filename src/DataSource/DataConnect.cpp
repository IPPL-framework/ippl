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
#include "DataSource/DataConnect.h"
#include "DataSource/DataSource.h"
#include "DataSource/DataSourceObject.h"
#include "DataSource/DataConnectCreator.h"
#include "Utility/IpplInfo.h"



/////////////////////////////////////////////////////////////////////////
// constructor
DataConnect::DataConnect(const char *nm, const char *id, int dtm, int n)
  : NamedObj(nm), MyID(id), nodes(n), DefTransMethod(dtm)
{
  if (n <= 0)
    nodes = DataConnectCreator::getDefaultNodes();
}


/////////////////////////////////////////////////////////////////////////
// destructor
DataConnect::~DataConnect()
{
  disconnectConnections();
}


/////////////////////////////////////////////////////////////////////////
// return true if our nodes is one of the connection nodes
bool DataConnect::onConnectNode() const {
  return (Ippl::myNode() < getNodes());
}


/////////////////////////////////////////////////////////////////////////
// are we currently connected to a receiver?  The base-class default
// behavior for this is to indicate that we're not connected.
bool DataConnect::connected() const {
  return false;
}


/////////////////////////////////////////////////////////////////////////
// Register an object as something that can be a source of data.
// Arguments = name of item, DataSource object, and transfer method
// (INPUT, OUTPUT, BOTH, or DEFAULT).  If this connection object is
// not actually connected, it is an error and this will return NULL.
// Otherwise, if the connection works, return the connection.
DataConnect *DataConnect::connect(const char *nm, DataSource *s, int tm) {
  DataConnect *conn = 0;
  if (connected() && s != 0)
    conn = s->connect(nm, this, tm);
  return conn;
}

DataConnect *DataConnect::connect(const char *nm, DataSource& s, int tm) {
  return connect(nm, &s, tm);
}


/////////////////////////////////////////////////////////////////////////
// Add a new single DataSourceObject connection.  It is added to the
// DataSource's list of single connections, and the DataSource will
// end up being added to our list of known sources.  Return success.
bool DataConnect::connect(DataSourceObject *dso) {
  

  // make sure the DataSourceObject has a source and the proper DataConnect
  if (dso == 0 || dso->getSource() == 0 || dso->getConnection() != this)
    return false;

  // tell the relevant DataSource it has a new DataSourceObject connection.
  // the DataSource will check itself in to our list of DataSources.
  return dso->getSource()->connect(dso);
}


/////////////////////////////////////////////////////////////////////////
// perform update for all registered DataSource's
void DataConnect::updateConnections(DataConnect *dc) {
  for (iterator a = begin(); a != end(); ++a)
    (*a)->updateConnection(dc);
}


/////////////////////////////////////////////////////////////////////////
// disconnect all the registered DataSource's
void DataConnect::disconnectConnections() {
  
  while (! SourceList.empty() )
    checkout(SourceList.front());
}


/////////////////////////////////////////////////////////////////////////
// allow all connections to perform an interactive action
void DataConnect::interact(const char *str, DataConnect *dc) {
  for (iterator a = begin(); a != end(); ++a)
    (*a)->interact(str, dc);
}


/////////////////////////////////////////////////////////////////////////
// Register a data object as connected here.  Return success.
bool DataConnect::checkin(DataSource *ds) {
  

  // make sure we do not have this DataSource already
  for (iterator a = begin(); a != end(); ++a) {
    if (*a == ds)
      return true;
  }

  // we don't have it, so add it to our list
  SourceList.push_back(ds);
  return true;
}


/////////////////////////////////////////////////////////////////////////
// remove a data object from our connected list.  Return success.
bool DataConnect::checkout(DataSource *ds, bool NeedDisconnect) {
  

  //Inform dbgmsg("DataConnect::checkout", INFORM_ALL_NODES);

  // make sure we have it ...
  for (iterator a = begin(); a != end(); ++a) {
    if (*a == ds) {
      // we do have it; make sure it is disconnected,
      // remove it and return success
      //dbgmsg << "Found DataSource ... ";
      SourceList.erase(a);
      if (NeedDisconnect) {
	//dbgmsg << "calling disconnect." << endl;
	ds->disconnect(this);
      } else {
	//dbgmsg << "SKIPPING disconnect." << endl;
      }
      return true;
    }
  }

  // if we're here, we did not find it
  //dbgmsg << "Could not find specified DataSource." << endl;
  return false;
}

/////////////////////////////////////////////////////////////////////////
// wait for synchronization from outside.
// default behavior is nothing.
void DataConnect::ready()
{
}

/***************************************************************************
 * $RCSfile: DataConnect.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:24 $
 * IPPL_VERSION_ID: $Id: DataConnect.cpp,v 1.1.1.1 2003/01/23 07:40:24 adelmann Exp $ 
 ***************************************************************************/
