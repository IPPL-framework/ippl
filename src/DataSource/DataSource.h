// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef DATA_SOURCE_H
#define DATA_SOURCE_H

/***********************************************************************
 * 
 * DataSource is the base class for all objects which desire to have their
 * contents available for another program or data processing API (e.g.,
 * visualization).  It provides an interface to do the following actions:
 * 1. CONNECT the object with the external program or API, by providing
 *    a name to use.
 *    Method:  DataConnect *connect(char *name, DataConnect * = 0)
 * 2. UPDATE the receiver end with the latest version of the data.
 *    Method:  bool update()
 * 3. INTERACT with the receiver, by handing off control to them until
 *    they are finished (for some receivers, this may be a no-op).
 *    Method:  bool interact()
 * 4. DISCONNECT from the receiver.  If not done explicitely, this will
 *    be done when the DataSource is deleted.
 *    Method:  bool disconnect()
 *
 * To construct a DataSource, a specific subclass of DataSourceObject
 * must be created, which provides the particular implementation of
 * these interface functions based on the type of data (ParticleAttrib
 * or Field) and the destination of the data
 *
 * Objects in Ippl which want to be a source for data to some other
 * agency should then do the following:
 * 1. Make your new class be derived from DataSource;
 * 2. Define a version of the protected virtual function
 *    'makeDataSourceObject'.  This function should create the proper
 *    subclass of DataSourceObject based on the new class' type and the
 *    method for connection.
 *
 ***********************************************************************/

// include files
#include <vector>

// forward declarations
class DataSourceObject;
class DataConnect;


// A base class for all objects in IPPL which we want to make accessible
// to external agencies (basically, ParticleAttrib's and Field's)
class DataSource {

public:
  // enumeration of data transfer directions:
  //   input   = obtain data from external agency
  //   output  = send data to external agency
  //   both    = can send and receive (meaning depends on context)
  //   default = get transfer direction from default setting in connect obj
  enum DsMode { INPUT, OUTPUT, BOTH, DEFAULT };

  // container type for connections
  typedef std::vector<DataSourceObject *> container_t;

public:
  // constructor
  DataSource();

  // destructor
  virtual ~DataSource();

  //
  // informative methods
  //

  // are we currently connected to a DataConnect object?  If an argument is
  // given, just check if we're connected to the specified DataConnect (else,
  // just report if we're connected to anyone).
  bool connected(DataConnect * = 0) const;

  // find the first DataSourceObject which is connected to the given
  // DataConnect, and return it; otherwise, return 0
  DataSourceObject *findDataSourceObject(DataConnect *) const;

  //
  // DataSource interface ... these mainly call the virtual functions in
  // the provided object
  //

  // Register an object as something that can be a source of data.
  // Arguments = name of item, connection to use, and type of
  // connection (INPUT, OUTPUT, BOTH, or DEFAULT).  If the connection
  // has not been created yet (e.g., it is NULL), create a new default
  // connection (or use the existing default one).  Return the connection.
  DataConnect *connect(const char *, DataConnect * = 0,
		       int = DataSource::DEFAULT);

  // Register the given DataSourceObject directly.  This is simpler than
  // the above version of connect, since the DataSourceObject has already
  // been created.  It can be used to register most any type of connection,
  // even one using DataConnect objects that are not part of IPPL itself.
  // Return success.
  bool connect(DataSourceObject *);

  // Disconnect an object from the DataConnect object.  Return success.
  // If no DataConnect is specified, disconnect from ALL connections.
  bool disconnect(DataConnect * = 0);

  // Update the object, that is, make sure the receiver of the data has a
  // current and consistent snapshot of the current state.  Return success.
  bool updateConnection(DataConnect * = 0);

  // Indicate to the receiver that we're allowing them time to manipulate the
  // data (e.g., for a viz program, to rotate it, change rep, etc.)
  // This should only return when the manipulation is done.  For some cases,
  // this will be a no-op.
  void interact(DataConnect * = 0);

  // Pass on a string to the connection, most likely to give it a
  // command to do some action.  Similar to the above interact, except
  // the request for interaction involves the given string
  void interact(const char *, DataConnect * = 0);

protected:
  // a virtual function which is called by this base class to get a
  // specific instance of DataSourceObject based on the type of data
  // and the connection method (the argument to the call).
  virtual DataSourceObject *createDataSourceObject(const char *,
						   DataConnect *,
						   int) = 0;

private:
  // the list of connected DataSourceObject's
  container_t ConnectionList;
};

#endif // DATA_SOURCE_H

/***************************************************************************
 * $RCSfile: DataSource.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:25 $
 * IPPL_VERSION_ID: $Id: DataSource.h,v 1.1.1.1 2003/01/23 07:40:25 adelmann Exp $ 
 ***************************************************************************/
