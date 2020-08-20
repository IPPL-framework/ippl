// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef DATA_CONNECT_H
#define DATA_CONNECT_H

/***********************************************************************
 * 
 * DataConnect is the base class for all objects which maintain a connection
 * with another program or external agency.  When constructed, it initializes
 * and maintains the connection, and provides information for transferral
 * of individual data objects such as Particles and Fields.
 *
 ***********************************************************************/

// include files
#include "DataSource/DataSource.h"
#include "Utility/NamedObj.h"

// forward declarations
class DataSourceObject;



class DataConnect : public NamedObj {

public:
  // typedef for our list of DataSource's
  typedef std::vector<DataSource *>   container_t;
  typedef container_t::iterator       iterator;
  typedef container_t::const_iterator const_iterator;

public:
  // constructor
  DataConnect(const char *nm, const char *id, int dtm = DataSource::OUTPUT,
	      int n = 0);

  // destructor
  virtual ~DataConnect();

  //
  // informative methods
  //

  // return the ID for this object ... different types of DataConnect
  // subclasses have different ID's
  // ada change ID() to DSID() because of name clash
  const char *DSID() const { return MyID.c_str(); }

  // get or set our defalt data transfer method
  int getDefaultTransferMethod() const { return DefTransMethod; }
  void setDefaultTransferMethod(int m) { DefTransMethod = m; }

  // return the number of nodes that should be used for display
  int getNodes() const { return nodes; }

  // return true if our nodes is one of the connection nodes
  bool onConnectNode() const;

  //
  // iterators for our list of registered objects
  //

  iterator begin() { return SourceList.begin(); }
  iterator end()   { return SourceList.end(); }

  const_iterator begin() const { return SourceList.begin(); }
  const_iterator end() const   { return SourceList.end(); }

  //
  // other container-like routines
  //

  // return the number of registered DataSource's
  unsigned int size() const { return SourceList.size(); }
  unsigned int numDataSources() const { return SourceList.size(); }
  bool empty() const { return SourceList.empty(); }

  //
  // DataConnect virtual methods, which all have default behavior
  //

  // are we currently connected to a receiver?
  virtual bool connected() const;

  // Register an object as something that can be a source of data.
  // Arguments = name of item, DataSource object, and transfer method
  // (INPUT, OUTPUT, BOTH, or DEFAULT).  If this connection object is
  // not actually connected, it is an error and this will return NULL.
  // Otherwise, if the connection works, return the connection.
  virtual DataConnect *connect(const char *, DataSource *,
			       int=DataSource::DEFAULT);
  virtual DataConnect *connect(const char *, DataSource &,
			       int=DataSource::DEFAULT);

  // Add a new single DataSourceObject connection.  It is added to the
  // DataSource's list of single connections, and the DataSource will
  // end up being added to our list of known sources.  Return success.
  virtual bool connect(DataSourceObject *);

  // perform update for all registered DataSource's.  The optional
  // argument can be used to just update all things connected to the
  // current connector and any other connectors (if the pointer is 0),
  // to just those things that are connected to the current connector
  // AND are also connected to the provided connector.
  virtual void updateConnections(DataConnect * = 0);

  // disconnect all the registered DataSource's.
  virtual void disconnectConnections();

  // allow all connections to perform an interactive action.  An optional
  // command string can be supplied; if it is null, it will be ignored.
  virtual void interact(const char * = 0, DataConnect * = 0);

  // synchronization mechanism for waiting on some outside request.
  virtual void ready();

private:
  friend class DataSource;

  // ID for this object
  std::string MyID;

  // our list of connected data objects
  container_t SourceList;

  // the number of nodes to connect with
  int nodes;

  // default transfer method
  int DefTransMethod;

  // Register a data object as connected here.  Return success.
  bool checkin(DataSource *);

  // remove a data object from our connected list.  Return success.
  // Argument = whether we need to have the DataSource disconnect first.
  bool checkout(DataSource *, bool = true);
};

#endif // DATA_CONNECT_H

/***************************************************************************
 * $RCSfile: DataConnect.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:24 $
 * IPPL_VERSION_ID: $Id: DataConnect.h,v 1.1.1.1 2003/01/23 07:40:24 adelmann Exp $ 
 ***************************************************************************/
