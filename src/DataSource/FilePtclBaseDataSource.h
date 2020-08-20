// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef FILE_PARTICLE_BASE_DATA_SOURCE_H
#define FILE_PARTICLE_BASE_DATA_SOURCE_H

/***********************************************************************
 *
 * class FileIpplParticleBaseDataSource
 *
 * A specific version of DataSourceObject which takes the data for
 * a given IpplParticleBase and writes it to a file using a DiscParticle
 * object.
 *
 ***********************************************************************/

// include files
#include "DataSource/DataSourceObject.h"
#include "Particle/IpplParticleBase.h"
#include "Utility/DiscParticle.h"


template<class T>
class FileIpplParticleBaseDataSource : public DataSourceObject {

public:
  // constructor: the name, the connection, the transfer method,
  // the IpplParticleBase to connect
  FileIpplParticleBaseDataSource(const char *, DataConnect *, int,
                                 IpplParticleBase<T> &);

  // destructor
  virtual ~FileIpplParticleBaseDataSource();

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
  // the DiscParticle object, which read/writes the data
  DiscParticle *DP;

  // the IpplParticleBase to read into (or write from)
  IpplParticleBase<T> &MyParticles;

  // the number of frames we have read or written (i.e. or current record)
  int counter;
};

#include "DataSource/FilePtclBaseDataSource.hpp"

#endif // FILE_PARTICLE_BASE_DATA_SOURCE_H

/***************************************************************************
 * $RCSfile: FilePtclBaseDataSource.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:25 $
 * IPPL_VERSION_ID: $Id: FilePtclBaseDataSource.h,v 1.1.1.1 2003/01/23 07:40:25 adelmann Exp $
 ***************************************************************************/