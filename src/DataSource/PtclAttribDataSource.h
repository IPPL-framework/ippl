// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef PARTICLE_ATTRIB_DATA_SOURCE_H
#define PARTICLE_ATTRIB_DATA_SOURCE_H

/***********************************************************************
 *
 * class ParticleAttribDataSource
 *
 * ParticleAttribDataSource is a base class for classes which
 * provides functionality needed to have the attributes check to see if
 * their associated IpplParticleBase has been previously connected, and if so,
 * to add this object to the list of connected attributes for that
 * IpplParticleBase.  This also provides virtual functions for transmitting
 * and receiving attribute data from other nodes.
 *
 ***********************************************************************/

// include files
#include "DataSource/DataSourceObject.h"
#include "Message/Message.h"


// forward declarations
class IpplParticleBaseDataSource;
class ParticleAttribBase;
class DataSource;


class ParticleAttribDataSource : public DataSourceObject {

public:
  // constructor: the name, the connection, the transfer method, the attrib
  ParticleAttribDataSource(const char *, DataConnect *, int,
			   ParticleAttribBase *, DataSource *);

  // destructor
  virtual ~ParticleAttribDataSource();

  // tell this attribute we're disconnecting it
  void setDisconnected() { PBase = 0; }

  //
  // ParticleAttribDataSource virtual functions
  //

  // retrieve the agency-specific data structure
  virtual void *getConnectStorage() { return 0; }

  // copy attrib data on the local processor into the given Message.
  virtual void putMessage(Message *) = 0;

  // prepare the agency-specific data structures for update; this may
  // require reallocation of storage memory, etc.
  // Argument = are we at the start (true) or end (false) of the data update;
  //            # of particles to prepare for.
  virtual void prepare_data(bool, unsigned) = 0;

  // take data for N particles out of the given message, and put it
  // into the proper agency-specific structure.
  // If the Message pointer is 0, put in the locally-stored particles.
  // Arguments: num particles,
  // starting index for inserted particles, Message containing
  // particles
  virtual void insert_data(unsigned, unsigned, Message *) = 0;

protected:
  // container holding our particle base ... used for disconnection
  IpplParticleBaseDataSource *PBase;
};


#endif // PARTICLE_ATTRIB_DATA_SOURCE_H

/***************************************************************************
 * $RCSfile: PtclAttribDataSource.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:25 $
 * IPPL_VERSION_ID: $Id: PtclAttribDataSource.h,v 1.1.1.1 2003/01/23 07:40:25 adelmann Exp $ 
 ***************************************************************************/
