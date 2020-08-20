// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef DATA_SOURCE_OBJECT_H
#define DATA_SOURCE_OBJECT_H

/***********************************************************************
 * 
 * DataSourceObject is used basically as a type-independent interface
 * to store the specific object which is being connected with a receiver,
 * and to provide the actual implementations of the interface functions
 * described above.  These are implemented in DataSourceObject as virtual
 * functions; a specific, derived version of DataSourceObject for the
 * different types of objects to connect (e.g. Field or ParticleAttrib)
 * must be given to the constructor of DataSource.  The specific
 * implementation (a class derived from DataSourceObject) must also be
 * customized to the type of data receivers, or
 * another external program or an API within the same process.
 *
 * While a single DataSource may be connected to several DataConnect's, and
 * a DataConnect may have several DataSource objects connected to it, there
 * is exactly one DataSourceObject per DataSource <---> DataConnect pair.
 * This class carries out the actual work to tranfer the data for one data
 * object from a sender to a receiver.
 *
 * The def type of data source to use for a particular invocation of a IPPL
 * application is selected via a command-line option; the IpplInfo class
 * will save this information, which is used by the templated global
 * function 'make_DataSourceObject'.  There should be a version of this
 * function provided for each type of object in IPPL we want to connect,
 * basically Field's and ParticleAttrib's.
 *
 ***********************************************************************/

// include files
#include "DataSource/DataSource.h"
#include "Utility/NamedObj.h"

// forward declarations
class DataConnect;


// The interface class for objects we wish to connect to other agencies.
// The function 'make_DataSourceObject' in DataSource/MakeDataSource.h will
// create the proper subclass of this object, for use by DataSource.
class DataSourceObject : public NamedObj {

public:
  // Constructor
  DataSourceObject(const char *nm, DataSource *ds, DataConnect *dc, int tm)
    : NamedObj(nm), Connection(dc), Source(ds), TransferMethod(tm) { }

  // Default constructor
  DataSourceObject()
    : Connection(0), Source(0), TransferMethod(DataSource::OUTPUT) { }

  // Destructor: make it virtual, but it does nothing.
  virtual ~DataSourceObject() { }

  // are we currently connected?
  bool connected() const { return (Connection != 0 && Source != 0); }

  // who are we connected to?
  DataConnect *getConnection() { return Connection; }

  // who are we connected from?
  DataSource *getSource() { return Source; }

  // 
  //
  // virtual function interface. The default versions of these objects
  // do nothing, so that we can have a default behavior.  Note that the
  // 'connect' and 'disconnect' functionality should be implemented in the
  // derived classes constructor and destructor, respectively.
  //

  // Update the object, that is, make sure the receiver of the data has a
  // current and consistent snapshot of the current state.  Return success.
  virtual bool update() { return false; }

  // Indicate to the receiver that we're allowing them time to manipulate the
  // data (e.g., for a viz program, to rotate it, change representation, etc.)
  // This should only return when the manipulation is done.
  // Optionally, a string can be passed on to the connection, possibly for
  // use as an interactive command.
  virtual void interact(const char * = 0) { }

protected:
  // our current connection ... if the connection was not successful, this
  // should be set to 0
  DataConnect *Connection;

  // our current source ... if the connection was not successful, this
  // should be set to 0
  DataSource *Source;

  // our transfer method, as requested by the user ... some subclasses may
  // not support all types of transfer, if the user asks for one which is
  // not supported, the connection should fail
  int TransferMethod;
};

#endif // DATA_SOURCE_OBJECT_H

/***************************************************************************
 * $RCSfile: DataSourceObject.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:25 $
 * IPPL_VERSION_ID: $Id: DataSourceObject.h,v 1.1.1.1 2003/01/23 07:40:25 adelmann Exp $ 
 ***************************************************************************/
