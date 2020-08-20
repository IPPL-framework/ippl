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
#include "DataSource/FilePtclBaseDataSource.h"
#include "DataSource/FileDataConnect.h"
#include "Utility/DiscParticle.h"
#include "Utility/IpplInfo.h"

////////////////////////////////////////////////////////////////////////////
// constructor: the name, the connection, the transfer method,
// the particlebase to connect
template<class T>
FileIpplParticleBaseDataSource<T>::FileIpplParticleBaseDataSource(const char *nm,
							  DataConnect *dc,
							  int tm,
							  IpplParticleBase<T>& P)
    : DataSourceObject(nm,&P,dc,tm), DP(0), MyParticles(P), counter(0) {

  std::string filestring = "file";
  if (std::string(dc->DSID()) != filestring) {
    ERRORMSG("Illegal DataConnect object for FILE Data Object." << endl);
    Connection = 0;
  } else if (tm != DataSource::OUTPUT && tm != DataSource::INPUT) {
    ERRORMSG("FILE connections may only be of type INPUT or OUTPUT." << endl);
    Connection = 0;
  } else {
    FileDataConnect *fdc = (FileDataConnect *)dc;
    int dptm = (TransferMethod == DataSource::OUTPUT ?
                DiscParticle::OUTPUT : DiscParticle::INPUT);
    DP = new DiscParticle(nm, dc->name(), dptm, fdc->getTypeString());
  }
}


////////////////////////////////////////////////////////////////////////////
// destructor
template<class T>
FileIpplParticleBaseDataSource<T>::~FileIpplParticleBaseDataSource() {

  if (DP != 0)
    delete DP;
}


////////////////////////////////////////////////////////////////////////////
// Update the object, that is, make sure the receiver of the data has a
// current and consistent snapshot of the current state.  Return success.
template<class T>
bool FileIpplParticleBaseDataSource<T>::update() {

  if (TransferMethod == DataSource::OUTPUT)
    return DP->write(MyParticles);
  else if (TransferMethod == DataSource::INPUT)
    return DP->read(MyParticles, counter++);
  else
    return false;
}


////////////////////////////////////////////////////////////////////////////
// Indicate to the receiver that we're allowing the connection time to
// manipulate the data (e.g., for a viz program, to rotate it, change
// representation, etc.).
// This should only return when the manipulation is done.
template<class T>
void FileIpplParticleBaseDataSource<T>::interact(const char *) {}


/***************************************************************************
 * $RCSfile: FilePtclBaseDataSource.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:25 $
 * IPPL_VERSION_ID: $Id: FilePtclBaseDataSource.cpp,v 1.1.1.1 2003/01/23 07:40:25 adelmann Exp $ 
 ***************************************************************************/
