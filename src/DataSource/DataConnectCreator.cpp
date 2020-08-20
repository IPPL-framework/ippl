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
#include "DataSource/DataConnectCreator.h"
#include "DataSource/DataConnect.h"
#include "DataSource/FileDataConnect.h"
#include "Utility/IpplInfo.h"
#include <cstring>


// static data for this file
static const int CONNECTMETHODS = 2;   // includes "no connection" method
static const char *ConnectMethodList  = "file, or none";
static const char *ConnectMethodNames[CONNECTMETHODS] = {"file", "none" };
static bool  ConnectMethodSupported[CONNECTMETHODS] = {true, true};


// a global instance of DataConnectCreator ... when this is destroyed
// at the end of the program, it will clean up the default connection
// object, if necessary
static DataConnectCreator GlobalDataConnectCreatorInstance;



/////////////////////////////////////////////////////////////////////////
// static member data for DataConnectCreator
int          DataConnectCreator::ConnectNodes      = 1;
int          DataConnectCreator::InstanceCount     = 0;
int          DataConnectCreator::DefaultMethod     = CONNECTMETHODS - 1;
DataConnect *DataConnectCreator::DefaultConnection = 0;



/////////////////////////////////////////////////////////////////////////
// constructor: increment the instance count
DataConnectCreator::DataConnectCreator() {
  
  InstanceCount++;
}


/////////////////////////////////////////////////////////////////////////
// destructor: deccrement the instance count, and if it goes to zero,
// delete any static object if necessary
DataConnectCreator::~DataConnectCreator() {
  
  if (--InstanceCount == 0)
    if (DefaultConnection != 0)
      delete DefaultConnection;
}


/////////////////////////////////////////////////////////////////////////
// return the name of the Nth method
int DataConnectCreator::getNumMethods() {
  
  return CONNECTMETHODS;
}


/////////////////////////////////////////////////////////////////////////
// return the name of the Nth method
const char *DataConnectCreator::getMethodName(int n) {
  
  if (n >= 0 && n < CONNECTMETHODS)
    return ConnectMethodNames[n];
  else
    return 0;
}


/////////////////////////////////////////////////////////////////////////
// return a list of all the methods, as a single string
const char *DataConnectCreator::getAllMethodNames() {
  
  return ConnectMethodList;
}


/////////////////////////////////////////////////////////////////////////
// check if the given connection method is supported
bool DataConnectCreator::supported(int cm) {
  
  return (known(cm) ? ConnectMethodSupported[cm] : false);
}


/////////////////////////////////////////////////////////////////////////
// check if the given connection method is supported
bool DataConnectCreator::supported(const char *nm) {
  
  return supported(libindex(nm));
}


/////////////////////////////////////////////////////////////////////////
// check if the given connection method is known at all
bool DataConnectCreator::known(int cm) {
  
  return (cm >= 0 && cm < CONNECTMETHODS);
}


/////////////////////////////////////////////////////////////////////////
// check if the given connection method is known at all
bool DataConnectCreator::known(const char *nm) {
  return known(libindex(nm));
}


/////////////////////////////////////////////////////////////////////////
// create a new connection.  Arguments = type, name, direction, nodes
// If n <= 0, use the "default" number of nodes set earlier
DataConnect *DataConnectCreator::create(int cm, const char *nm, int n) {
  
  // initially, we have a null pointer for the connection.  If everything
  // checks out, we'll have a non-null pointer at the end.  If we still
  // have a null pointer at the end of this routine, something went wrong
  // and we return NULL to indicate an error.
  DataConnect *dataconn = 0;

  // figure out how many nodes the connection should use, if it cares
  // at all about that.  
  int nodes = n;
  if (n <= 0)
    nodes = getDefaultNodes();

  if (cm == 0) {
    // transfer the data to/from a file using some form of parallel I/O
    dataconn = new FileDataConnect(nm, nodes);
  } else if (cm == 1) {
    // just make a dummy connect object, which does nothing
    dataconn = new DataConnect(nm, getMethodName(cm), DataSource::OUTPUT, nodes);
  }
  // make sure we have something
  if (dataconn == 0) {
    ERRORMSG("DataConnectCreator: unknown connection method." << endl);
  }
  return dataconn;
}


/////////////////////////////////////////////////////////////////////////
// a final method for creating objects; this one provides a default name,
// and if the default connection object has already been created, this just
// returns that one.
DataConnect *DataConnectCreator::create() {
  

  if (DefaultConnection == 0)
    DefaultConnection = create(getMethodName(DefaultMethod));
  return DefaultConnection;
}


/////////////////////////////////////////////////////////////////////////
// change the default connection method.  Return success.
bool DataConnectCreator::setDefaultMethod(int cm) {
  
  if (supported(cm)) {
    DefaultMethod = cm;
    return true;
  }
  return false;
}


/////////////////////////////////////////////////////////////////////////
// return the index of the given named method, or (-1) if not found
int DataConnectCreator::libindex(const char *nm) {
  
  for (int i=0; i < CONNECTMETHODS; ++i) {
    if (strcmp(nm, getMethodName(i)) == 0)
      return i;
  }

  // if here, it was not found
  return (-1);
}


/////////////////////////////////////////////////////////////////////////
// change the default number of nodes to use for the connection
void DataConnectCreator::setDefaultNodes(int n) {
  ConnectNodes = n;
}


/////////////////////////////////////////////////////////////////////////
// return the default number of nodes to use in a connection
int DataConnectCreator::getDefaultNodes() {
  return (ConnectNodes >= 0 && ConnectNodes <= Ippl::getNodes() ?
	  ConnectNodes :
	  Ippl::getNodes());
}


/***************************************************************************
 * $RCSfile: DataConnectCreator.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:24 $
 * IPPL_VERSION_ID: $Id: DataConnectCreator.cpp,v 1.1.1.1 2003/01/23 07:40:24 adelmann Exp $ 
 ***************************************************************************/
