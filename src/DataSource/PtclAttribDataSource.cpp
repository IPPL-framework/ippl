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
#include "DataSource/PtclAttribDataSource.h"
#include "DataSource/PtclBaseDataSource.h"

///////////////////////////////////////////////////////////////////////////
// constructor: the name, the connection, the transfer method, the attrib
ParticleAttribDataSource::ParticleAttribDataSource(const char *nm,
        DataConnect *dc, int tm, ParticleAttribBase *pa, DataSource *ds)
  : DataSourceObject(nm, ds, dc, tm) {

  // find a particlebase object which contains this attribute; if none
  // found, it is an error
  PBase = IpplParticleBaseDataSource::find_particle_base(this, pa);

  // if we did find it ...
  if (PBase != 0) {
    // change our name to include the particle base's name
    std::string newname = PBase->name();
    newname += ":";
    newname += name();
    setName(newname.c_str());
  }
}


///////////////////////////////////////////////////////////////////////////
// destructor
ParticleAttribDataSource::~ParticleAttribDataSource() { }



/***************************************************************************
 * $RCSfile: PtclAttribDataSource.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:25 $
 * IPPL_VERSION_ID: $Id: PtclAttribDataSource.cpp,v 1.1.1.1 2003/01/23 07:40:25 adelmann Exp $ 
 ***************************************************************************/
