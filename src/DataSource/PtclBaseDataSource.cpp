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
#include "DataSource/PtclBaseDataSource.h"
#include "DataSource/PtclAttribDataSource.h"



// static objects for this class
IpplParticleBaseDataSource::BaseList_t IpplParticleBaseDataSource::BaseList;


///////////////////////////////////////////////////////////////////////////
// constructor: name, connection method, transfer method
IpplParticleBaseDataSource::IpplParticleBaseDataSource(const char *nm,
					       DataConnect *dc,
					       int tm,
					       DataSource *ds)
  : DataSourceObject(nm, ds, dc, tm){ }


///////////////////////////////////////////////////////////////////////////
// destructor ... unregister ourselves if this has not already been done
IpplParticleBaseDataSource::~IpplParticleBaseDataSource() {

  // disconnect all our currently connected attributes
  while (AttribList.size() > 0)
    disconnect_attrib(AttribList.front());

  // remove ourselves from the list of available IpplParticleBase containers
  checkout();
}


///////////////////////////////////////////////////////////////////////////
// try to add a new ParticleAttrib (stored in a ParticleAttribDataSource
// object) to our list of connected attributes.  This will check through
// the list of registered IpplParticleBase's, and add it to the proper one.
// If none are found, this returns NULL, otherwise this method returns
// a pointer to the IpplParticleBaseDataSource to which the attrib was added.
// This function is static, so that it may be called without a specific
// IpplParticleBaseDataSource instance.
IpplParticleBaseDataSource*
IpplParticleBaseDataSource::find_particle_base(ParticleAttribDataSource *pa,
					   ParticleAttribBase *pabase) {

  // search through the particle base holders, and check for matching
  // connection method, and if pa is in currbase
  BaseList_t::iterator currbase = IpplParticleBaseDataSource::begin_base();
  BaseList_t::iterator endbase = IpplParticleBaseDataSource::end_base();
  for ( ; currbase != endbase; ++currbase ) {
    IpplParticleBaseDataSource *list = *currbase;
    if (pa->getConnection()==list->getConnection() && list->has_attrib(pabase))
      return list;
  }

  // if we're here, we did not find the attribute
  return 0;
}


///////////////////////////////////////////////////////////////////////////
// register ourselves as a properly-connected IpplParticleBase holder.  This
// should be called by the constructors of subclasses after a successful
// connect.  Argument = name of this particle base
void IpplParticleBaseDataSource::checkin() {


  // first see if we're already here ...
  BaseList_t::iterator currbase = IpplParticleBaseDataSource::begin_base();
  BaseList_t::iterator endbase = IpplParticleBaseDataSource::end_base();
  for ( ; currbase != endbase; ++currbase )
    if (*currbase == this)
      return;			// we're already checked in

  // add to the end of the list
  BaseList.push_back(this);
}


///////////////////////////////////////////////////////////////////////////
// unregister ourselves ... generally called by subclass destructors.
void IpplParticleBaseDataSource::checkout() {


  for (unsigned i=0; i < BaseList.size(); ++i) {
    if (BaseList[i] == this) {
      BaseList[i] = BaseList.back(); // move last element into this spot
      BaseList.pop_back();
      return;
    }
  }
}


////////////////////////////////////////////////////////////////////////////
// make a connection using the given attribute.  Return success.
bool IpplParticleBaseDataSource::connect_attrib(ParticleAttribDataSource *pa) {
  AttribList.push_back(pa);
  return true;
}


////////////////////////////////////////////////////////////////////////////
// disconnect from the external agency the connection involving this
// particle base and the given attribute.  Return success.
bool IpplParticleBaseDataSource::disconnect_attrib(ParticleAttribDataSource *pa) {

  // remove the attribute from our list
  int i, size = AttribList.size();
  for (i=0; i < size; ++i) {
    if (pa == AttribList[i]) {
      AttribList[i] = AttribList.back();
      AttribList.pop_back();
      break;
    }
  }

  // tell the attribute we're disconnecting it
  pa->setDisconnected();
  return true;
}


/***************************************************************************
 * $RCSfile: PtclBaseDataSource.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:25 $
 * IPPL_VERSION_ID: $Id: PtclBaseDataSource.cpp,v 1.1.1.1 2003/01/23 07:40:25 adelmann Exp $
 ***************************************************************************/