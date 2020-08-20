// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef PARTICLE_BASE_DATA_SOURCE_H
#define PARTICLE_BASE_DATA_SOURCE_H

/***********************************************************************
 *
 * class IpplParticleBaseDataSource
 *
 * IpplParticleBaseDataSource is a base class which
 * stores a static list of all IpplParticleBaseDataSource's (stored as
 * IpplParticleBaseDataSource pointers) which are currently connected.  This
 * is needed so that ParticleAttrib's can determine if their parent
 * IpplParticleBase has been previously connected, and if so they can then
 * put themselves in the list for that IpplParticleBase as an attribute to be
 * transmitted along with the particle positions.  Subclasses provide
 * specific functionality to connect to external agencies such as viz
 * programs.
 *
 ***********************************************************************/

// include files 
#include "DataSource/DataSourceObject.h"

#include <vector>

// forward declarations
class ParticleAttribDataSource;
class ParticleAttribBase;


class IpplParticleBaseDataSource : public DataSourceObject {

public:
  // some useful typedefs
  typedef std::vector<ParticleAttribDataSource *> AttribList_t;
  typedef std::vector<IpplParticleBaseDataSource *>   BaseList_t;

public:
  // constructor: name, connection method, transfer method
  IpplParticleBaseDataSource(const char *, DataConnect *, int, DataSource *);

  // destructor
  virtual ~IpplParticleBaseDataSource();

  // get the begin/end iterators for the list of attributes
  AttribList_t::iterator begin_attrib() { return AttribList.begin(); }
  AttribList_t::iterator end_attrib()   { return AttribList.end(); }

  // return begin/end iterators for the list of particle base holders
  static BaseList_t::iterator begin_base() { return BaseList.begin(); }
  static BaseList_t::iterator end_base()   { return BaseList.end(); }

  // try to add a new ParticleAttrib (stored in a ParticleAttribDataSource
  // object) to our list of connected attributes.  This will check through
  // the list of registered IpplParticleBase's, and add it to the proper one.
  // If none are found, this returns NULL, otherwise this method returns
  // a pointer to the IpplParticleBaseDataSource to which the attrib was added.
  // This function is static, so that it may be called without a specific
  // IpplParticleBaseDataSource instance.
  static
  IpplParticleBaseDataSource* find_particle_base(ParticleAttribDataSource *,
					     ParticleAttribBase *);

  //
  // IpplParticleBaseDataSource public virtual function interface
  //

  // make a connection using the given attribute.  Return success.
  virtual bool connect_attrib(ParticleAttribDataSource *);

  // disconnect from the external agency the connection involving this
  // particle base and the given attribute.  Return success.
  virtual bool disconnect_attrib(ParticleAttribDataSource *);

  // check to see if the given ParticleAttribBase is in this IpplParticleBase's
  // list of registered attributes.  Return true if this is so.
  virtual bool has_attrib(ParticleAttribBase *) = 0;

protected:
  // register ourselves as a properly-connected IpplParticleBase holder.  This
  // should be called by the connect method in subclasses after a successful
  // checkin.
  void checkin();

  // unregister ourselves ... generally called by the disconnect method
  // of subclasses.
  void checkout();

private:
  // a non-static list of the ParticleAttrib's which have been requested
  // to connect to the same receiver as this object's IpplParticleBase
  AttribList_t AttribList;

  // a static list of the currently-connected IpplParticleBaseDataSource's.
  static BaseList_t BaseList;
};


#endif // PARTICLE_BASE_DATA_SOURCE_H

/***************************************************************************
 * $RCSfile: PtclBaseDataSource.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:25 $
 * IPPL_VERSION_ID: $Id: PtclBaseDataSource.h,v 1.1.1.1 2003/01/23 07:40:25 adelmann Exp $ 
 ***************************************************************************/
